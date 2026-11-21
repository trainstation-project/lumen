//! What the Triton code generators share: the backend type, element types, constants, the fill kernel,
//! template `BODY` functions and text helpers.
//!
//! Triton codegen: each planned step's loop kernel, generated once when the plan compiles (its `BODY`, for a
//! templated kernel, generated with its tensors' layouts at launch).

use crate::DType;
use crate::autotune::Autotuner;
use crate::codegen::attention::triton::FlashAttention;
use crate::codegen::dot_general::triton::Gemm;
use crate::codegen::elementwise::triton::Loop;
use crate::codegen::fusion::triton::{COMBINE, fusible};
use crate::codegen::kernel::Runner;
use crate::codegen::kinds::{combine_root, gemm_dot, reduction_root};
use crate::codegen::layout::{TILE, transpose_tiling};
use crate::codegen::reduction::ReductionCodegen;
use crate::codegen::row::RowCodegen;
use crate::codegen::scan::triton::Scan;
use crate::codegen::scan::triton::scan;
use crate::codegen::scatter_add::triton::atomic_add;
use crate::codegen::scatter_add::triton::launch::Scatter;
use crate::codegen::store::triton::{load, store};
use crate::graph::{Graph, Node, Primitive, UPDATES};
use crate::graph_compiler::device::{Device, DeviceModule};
use crate::graph_compiler::passes::horizontal::segments;
use crate::graph_compiler::{Options, plan_options, positions};
use crate::plan::step::triton::{Collective, TritonTimer, Unsupported, Wait};
use crate::plan::{Buffer, CpuPlan, Plan, Step as _, StepBase, TritonPlan, TritonStep};
use crate::tensor::dtype::dispatch_dtype;
use crate::tensor::storage::as_bytes;
use crate::{Element, Scalar};
use indoc::formatdoc;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

/// The Triton code generator.
pub(crate) struct Triton;

/// Triton element type for `dtype`, or `None` when Triton can't hold it.
pub(crate) fn triton_type(dtype: DType) -> Option<&'static str> {
    use DType::*;

    Some(match dtype {
        Bool => "tl.int1",
        U8 => "tl.uint8",
        U16 => "tl.uint16",
        U32 => "tl.uint32",
        U64 => "tl.uint64",
        I8 => "tl.int8",
        I16 => "tl.int16",
        I32 => "tl.int32",
        I64 => "tl.int64",
        F16 => "tl.float16",
        BF16 => "tl.bfloat16",
        F32 => "tl.float32",
        F64 => "tl.float64",
        _ => return None,
    })
}

/// Kernel parameter list: inputs (`in{k}`), optional `extra`, `out`, `n`, a `LOAD{k}` helper per
/// by-reference input, `stores` store helpers (`STORE`, then `STORE{k}` from 1) and `BLOCK_SIZE`,
/// one per line.
pub(crate) fn params(by_value: &[bool], extra: Option<&str>, stores: usize) -> String {
    let inputs = (0..by_value.len())
        .map(|k| format!("in{k}"))
        .chain(extra.map(String::from));

    let loads = (0..by_value.len())
        .filter(|&k| !by_value[k])
        .map(|k| format!("LOAD{k}: tl.constexpr"));

    inputs
        .chain(["out".into(), "n".into()])
        .chain(loads)
        .chain((0..stores).map(|k| match k {
            0 => "STORE: tl.constexpr".into(),
            k => format!("STORE{k}: tl.constexpr"),
        }))
        .chain(["BLOCK_SIZE: tl.constexpr".into()])
        .map(|p| format!("    {p},\n"))
        .collect()
}

/// Generates a kernel filling `out` with a constant passed as its raw bits (int64) at launch.
pub(crate) fn fill(dtype: DType) -> (String, String) {
    let value = match dtype {
        DType::Bool => "bits != 0".to_string(),
        _ => format!(
            "tl.full([], bits, {}).to({}, bitcast=True)",
            raw_type(dtype),
            triton_type(dtype).expect("a Triton dtype")
        ),
    };

    let source = formatdoc!(
        "
        @triton.jit
        def NAME(bits, out, n, STORE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            j = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            STORE(out, j, {value}, j < n)
        "
    );

    named("fill", source)
}

/// A tensor's shape and strides, in elements.
pub(crate) type Owned = (Vec<usize>, Vec<usize>);

/// The layouts a template kernel's `BODY` is generated for: each input's (`None` when it is passed by
/// value), then each output's.
pub(crate) struct Tensors {
    pub(crate) inputs: Vec<Option<Owned>>,
    pub(crate) outputs: Vec<Owned>,
}

/// A template kernel's generated `BODY(EXTRA, {params})` running `code`: it binds the step's inputs `in{k}`
/// and outputs `out` and `out{k}` from `EXTRA` (the inputs, then the outputs after the first, then the
/// first), and defines the `LOAD{k}` and `STORE`/`STORE{k}` helpers `code` calls for their `layouts`.
/// Returns `(name, source)`, the name `kind` plus a hash of the source.
pub(crate) fn body_function(
    kind: &str,
    layouts: &Tensors,
    params: &str,
    code: &str,
) -> (String, String) {
    let (inputs, outputs) = (layouts.inputs.len(), layouts.outputs.len());
    let mut helpers = String::new();

    for (k, layout) in layouts.inputs.iter().enumerate() {
        if let Some((shape, strides)) = layout {
            let (name, source) = load(shape, strides);
            helpers += &source.replace(&format!("def {name}("), &format!("def LOAD{k}("));
            helpers += "\n";
        }
    }

    for (k, (shape, strides)) in layouts.outputs.iter().enumerate() {
        let (name, source) = store(shape, strides);

        let param = match k {
            0 => "STORE".to_string(),
            k => format!("STORE{k}"),
        };

        helpers += &source.replace(&format!("def {name}("), &format!("def {param}("));
        helpers += "\n";
    }

    let mut bind: String = (0..inputs)
        .map(|k| format!("    in{k} = EXTRA[{k}]\n"))
        .collect();

    for k in 1..outputs {
        writeln!(bind, "    out{k} = EXTRA[{}]", inputs + k - 1).unwrap();
    }

    writeln!(bind, "    out = EXTRA[{}]", inputs + outputs - 1).unwrap();

    named(
        kind,
        format!("{helpers}@triton.jit\ndef NAME(EXTRA, {params}):\n{bind}{code}"),
    )
}

/// Bit pattern of `value` cast to `dtype`, zero-extended into a `u64` (little-endian bytes).
pub(crate) fn bits(dtype: DType, value: Scalar) -> u64 {
    let bytes = dispatch_dtype!(dtype, T => as_bytes(std::slice::from_ref(&T::from_scalar(value))).to_vec());

    bytes
        .iter()
        .rev()
        .fold(0u64, |acc, &b| (acc << 8) | b as u64)
}

/// Substitutes `NAME` in `source` with `kind` plus a hash of the source, so identical kernels share
/// a name and compile cache entry. Returns `(name, source)`.
pub(crate) fn named(kind: &str, source: String) -> (String, String) {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    let name = format!("{kind}_{:016x}", hasher.finish());
    (name.clone(), source.replace("NAME", &name))
}

/// Unsigned Triton type as wide as `dtype`, used to bitcast constants.
fn raw_type(dtype: DType) -> &'static str {
    match dtype.size_of() {
        1 => "tl.uint8",
        2 => "tl.uint16",
        4 => "tl.uint32",
        _ => "tl.uint64",
    }
}

/// Triton expression for the scalar `value` of `dtype`, built bit-exactly from its raw bits.
pub(crate) fn constant(dtype: DType, value: Scalar) -> String {
    let (bits, raw) = (bits(dtype, value), raw_type(dtype));

    match dtype {
        DType::Bool => format!("tl.full([], {}, tl.int1)", (bits != 0) as u8),
        _ => format!(
            "tl.full([], {bits:#x}, {raw}).to({}, bitcast=True)",
            triton_type(dtype).expect("a Triton dtype")
        ),
    }
}

/// Indents every line of `lines` by four spaces.
pub(crate) fn indented(lines: &str) -> String {
    lines.lines().map(|l| format!("    {l}\n")).collect()
}

/// A generated Triton loop kernel for one plan step.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Kernel {
    pub name: String,
    /// Python source defining the kernel (and its helpers).
    pub source: String,
    /// Per input, whether it is passed by value as a scalar.
    pub by_value: Vec<bool>,
    /// Reduced elements each output processes per iteration; 1 for elementwise kernels.
    pub lanes: usize,
    /// Programs per output of a split reduction; 1 otherwise.
    pub splits: usize,
    /// Element counts of a horizontal fusion's grid segments, each on its own programs; empty
    /// when every output has one element count.
    pub segments: Vec<usize>,
    /// The fixed kernel the step's generated `BODY` runs in, its source then empty: the `BODY` is generated
    /// with its tensors' layouts, among its helpers. `None` for a kernel generated whole.
    pub template: Option<Template>,
}

/// A fixed Triton kernel a step's generated `BODY` runs in (a scatter-add's runs without one).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub enum Template {
    /// `elementwise_kernel` (lumen.codegen.elementwise.triton.kernel): blocks of each grid segment's elements.
    Elementwise,
    /// `transpose_kernel` (lumen.codegen.layout.triton.kernel): `TILE` x `TILE` tiles of an `a` x `b` output, per
    /// batch.
    Transpose { a: usize, b: usize },
    /// `dynamic_update_slice_kernel` (lumen.codegen.dynamic_slice.triton.kernel): blocks of the update's elements.
    DynamicUpdateSlice,
    /// `cross_entropy_combine_kernel` (lumen.codegen.dot_general.triton.linear_cross_entropy_kernel): one row a
    /// program, its `tiles` partials `BLOCK_SIZE` at a time.
    CrossEntropyCombine { tiles: usize },
    /// `kernel`, one of the scatter-add kernels (lumen.codegen.scatter_add.triton.kernel), of a scatter-add
    /// of `picks` updates along an operand viewed as `[outer, length, after]`.
    ScatterAdd {
        outer: usize,
        length: usize,
        after: usize,
        picks: usize,
        kernel: Scatter,
    },
    /// `scan_kernel` (lumen.codegen.scan.triton.kernel): a tile of the lines a program, scanned as `Scan`
    /// tiles them.
    Scan(Scan),
}

#[cfg_attr(not(feature = "python"), allow(dead_code))]
impl Template {
    /// The Python module and name of the kernel.
    pub fn kernel(self) -> (&'static str, &'static str) {
        match self {
            Template::Elementwise => (
                "lumen.codegen.elementwise.triton.kernel",
                "elementwise_kernel",
            ),
            Template::Transpose { .. } => {
                ("lumen.codegen.layout.triton.kernel", "transpose_kernel")
            }
            Template::DynamicUpdateSlice => (
                "lumen.codegen.dynamic_slice.triton.kernel",
                "dynamic_update_slice_kernel",
            ),
            Template::CrossEntropyCombine { .. } => (
                "lumen.codegen.dot_general.triton.linear_cross_entropy_kernel",
                "cross_entropy_combine_kernel",
            ),
            Template::ScatterAdd { kernel, .. } => {
                ("lumen.codegen.scatter_add.triton.kernel", kernel.name())
            }
            Template::Scan(_) => ("lumen.codegen.scan.triton.kernel", "scan_kernel"),
        }
    }

    /// The kernel's constexprs for a launch over `n` outputs in `segments` (empty for one segment): a
    /// loop's segments' element counts, a transpose's output sides and tile, an update's element count, a
    /// cross entropy's rows and tiles, a scatter-add's count (update elements, picks, positions or rows
    /// programs) and sizes, a scan's lines and tiling.
    pub fn constexprs(self, n: usize, segments: &[usize]) -> Vec<(&'static str, Constexpr)> {
        match self {
            Template::Elementwise => {
                let segments = match segments {
                    [] => vec![n],
                    segments => segments.to_vec(),
                };

                vec![("SEGMENTS", Constexpr::Ints(segments))]
            }
            Template::Transpose { a, b } => vec![
                ("A", Constexpr::Int(a)),
                ("B", Constexpr::Int(b)),
                ("TILE", Constexpr::Int(TILE)),
            ],
            Template::DynamicUpdateSlice => vec![("N", Constexpr::Int(n))],
            Template::CrossEntropyCombine { tiles } => {
                vec![("N", Constexpr::Int(n)), ("TILES", Constexpr::Int(tiles))]
            }
            Template::ScatterAdd {
                outer,
                length,
                after,
                picks,
                ..
            } => vec![
                ("N", Constexpr::Int(n)),
                ("OUTER", Constexpr::Int(outer)),
                ("LENGTH", Constexpr::Int(length)),
                ("AFTER", Constexpr::Int(after)),
                ("PICKS", Constexpr::Int(picks)),
            ],
            Template::Scan(scan) => vec![
                ("N", Constexpr::Int(n)),
                ("LENGTH", Constexpr::Int(scan.length)),
                ("INNER", Constexpr::Int(scan.inner)),
                ("LANES", Constexpr::Int(scan.lanes)),
                ("ROWS", Constexpr::Int(scan.rows.into())),
                ("REVERSE", Constexpr::Int(scan.reverse.into())),
                ("ACCUM_DTYPE", Constexpr::DType(scan.accum_dtype)),
            ],
        }
    }
}

/// A template kernel's constexpr value.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub enum Constexpr {
    Int(usize),
    /// A tuple of ints.
    Ints(Vec<usize>),
    /// A `triton.language` dtype.
    DType(DType),
}

/// Name of a normalization row fusion.
pub(crate) const ROW: &str = "row";

/// Triton's plan of `plan`: each step with its loop kernel (none for a step another kernel runs).
pub(crate) fn codegen(plan: CpuPlan, options: &Options) -> Result<TritonPlan, String> {
    let deterministic = options.config.deterministic;

    plan.map(|step| {
        let code = loop_kernel(&step, deterministic)?;
        let base = step.into_base();
        let kernel = kernel_of(&base, code.is_some());
        Ok(TritonStep::new(base, code, kernel))
    })
}

/// The kernel `step` runs: flash attention, the GEMM, a collective or a wait, else its generated loop kernel (when
/// codegen generated one: `looped`), else none (it fails when it runs).
fn kernel_of(
    step: &StepBase,
    looped: bool,
) -> Arc<dyn Runner<TritonStep, crate::plan::step::Tensors>> {
    match &step.primitive {
        p if crate::codegen::attention::triton::launch::is_attention(p) => Arc::new(FlashAttention),
        _ if crate::codegen::dot_general::triton::launch::is_gemm(step) => Arc::new(Gemm),
        p if p.is_collective() => Arc::new(Collective),
        Primitive::DeviceWait | Primitive::HostWait => Arc::new(Wait),
        Primitive::Fusion { .. }
        | Primitive::Cumsum { .. }
        | Primitive::DynamicUpdateSlice
        | Primitive::ScatterAdd { .. }
            if looped =>
        {
            Arc::new(Loop)
        }
        _ => Arc::new(Unsupported),
    }
}

/// The name of a templated kernel: `kind` plus a hash of `key`, what its `BODY` is generated from (the
/// `BODY`, generated per layout at launch, is named after its own source).
fn templated(kind: &str, key: &str) -> String {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    format!("{kind}_{:016x}", hasher.finish())
}

/// Generates the loop kernel for `step`, or `None` for GEMMs and steps without a Triton loop kernel;
/// unless `deterministic`, a scatter-add of a dtype with atomic adds adds each update element atomically,
/// otherwise it sorts its picks by position and adds a row's in order. Fails on a body no kernel takes.
pub(crate) fn loop_kernel(step: &StepBase, deterministic: bool) -> Result<Option<Kernel>, String> {
    let by_value: Vec<bool> = step
        .inputs
        .iter()
        .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
        .collect();

    // Whether a scatter-add adds atomically; any other step ignores it.
    let atomic = !deterministic && atomic_add(step.output.1.dtype);

    let (name, source, lanes, splits) = match &step.primitive {
        // A chain of updates runs in the dynamic-update-slice template, its `BODY` generated at launch.
        Primitive::Fusion { name, body, .. } if name == UPDATES => (
            templated("dynamic_update_slice", &body.to_string()),
            String::new(),
            1,
            1,
        ),
        Primitive::Fusion { body, .. } if gemm_dot(body).is_some() => return Ok(None),
        Primitive::Fusion { name, body, .. } if name == ROW => {
            let r =
                Triton::row(body, &by_value, &()).map_err(|e| format!("{}: {e}", step.label))?;

            (r.name, r.source, r.lanes, r.splits)
        }
        Primitive::Fusion { name, body, .. } if name == COMBINE => {
            let root = combine_root(body).expect("a split-K combine's sum");
            let r = crate::codegen::reduction::triton::split_k_combine(body, &by_value, root)
                .map_err(|e| format!("{}: {e}", step.label))?;

            (r.name, r.source, r.lanes, r.splits)
        }
        Primitive::Fusion { body, .. } => match reduction_root(body) {
            Some(root) => {
                let r = Triton::reduction(body, &by_value, root)
                    .map_err(|e| format!("{}: {e}", step.label))?;

                (r.name, r.source, r.lanes, r.splits)
            }
            // An elementwise fusion runs in a template kernel, its `BODY` generated with its tensors' layouts at launch.
            None => (
                templated("elementwise", &body.to_string()),
                String::new(),
                1,
                1,
            ),
        },
        // A cumsum runs in the scan template, loading and storing through helpers for its layouts.
        Primitive::Cumsum { .. } => (
            templated(
                "scan",
                &format!("{:?} {}", step.primitive, step.inputs[0].1),
            ),
            String::new(),
            scan(step)?.lanes,
            1,
        ),
        Primitive::DynamicUpdateSlice => {
            let types: Vec<String> = step.inputs.iter().map(|(_, t)| t.to_string()).collect();

            (
                templated("dynamic_update_slice", &format!("{types:?} {by_value:?}")),
                String::new(),
                1,
                1,
            )
        }
        Primitive::ScatterAdd { .. } => {
            let types: Vec<String> = step.inputs.iter().map(|(_, t)| t.to_string()).collect();

            let kind = match atomic {
                true => "scatter_add_atomic",
                false => "scatter_add_sorted",
            };

            (
                templated(kind, &format!("{:?} {types:?}", step.primitive)),
                String::new(),
                1,
                1,
            )
        }
        _ => return Ok(None),
    };

    // Only an elementwise fusion's outputs (a horizontal fusion's) split the grid; a row's are one row each.
    let segments = match &step.primitive {
        Primitive::Fusion { name, body, .. }
            if name != ROW && name != COMBINE && reduction_root(body).is_none() =>
        {
            segments(body)
        }
        _ => Vec::new(),
    };

    let segments = match segments.len() {
        0 | 1 => Vec::new(),
        _ => segments.into_iter().map(|(numel, _)| numel).collect(),
    };

    // An elementwise fusion computed from transposes it can tile runs a tile of the output a program.
    let template = match &step.primitive {
        Primitive::Cumsum { .. } => Some(Template::Scan(scan(step)?)),
        Primitive::Fusion { name, .. } if name == UPDATES => Some(Template::DynamicUpdateSlice),
        Primitive::DynamicUpdateSlice => Some(Template::DynamicUpdateSlice),
        Primitive::ScatterAdd { axis } => {
            let x = &step.inputs[0].1.shape;

            Some(Template::ScatterAdd {
                outer: x[..*axis].iter().product(),
                length: x[*axis],
                after: x[axis + 1..].iter().product(),
                picks: step.inputs[1].1.numel(),
                kernel: match atomic {
                    true => Scatter::Atomic,
                    false => Scatter::Rows,
                },
            })
        }
        Primitive::Fusion { name, body, .. }
            if name != UPDATES
                && name != ROW
                && name != COMBINE
                && reduction_root(body).is_none() =>
        {
            match transpose_tiling(body) {
                Some(tiling) => Some(Template::Transpose {
                    a: tiling.a,
                    b: tiling.b,
                }),
                None => Some(Template::Elementwise),
            }
        }
        _ => None,
    };

    Ok(Some(Kernel {
        name,
        source,
        by_value,
        lanes,
        splits,
        segments,
        template,
    }))
}

/// Triton rejects half-precision transcendentals; it plans the passed graph, generates its loop kernels and
/// autotunes GEMM and loop configs when enabled.
impl Device for Triton {
    const NAME: &'static str = "triton";

    /// Plans `module`'s graph with its packed blocks, generates its kernels and autotunes them.
    fn lower(
        module: DeviceModule<'_, Self>,
        device: crate::Device,
    ) -> Result<Box<dyn Plan>, String> {
        let DeviceModule {
            graph: fused,
            options,
            packed,
            interleaved,
            ..
        } = module;

        let config = &options.config;
        let scalars = positions(&options.scalars);
        let plan = plan_options(&fused, options, config, &packed, scalars);
        let mut plan = CpuPlan::compile_with(&fused, &plan);
        plan.packed = packed;
        plan.interleaved = interleaved;
        let mut plan = codegen(plan, options)?;

        if config.autotune || crate::autotune::forcing() {
            let tuner = Autotuner {
                timer: TritonTimer { device },
                autotune: config.autotune,
            };

            tuner.tune_steps(&mut plan)?;
        }

        Ok(Box::new(plan))
    }
}

impl crate::codegen::Lone for DeviceModule<'_, Triton> {
    /// Whether lone `n` is one a Triton loop kernel computes (Triton has no other kernel for it): every
    /// fusible node but views of their operand (reshapes, slices placed as views) and in-place
    /// concatenations.
    fn alone(graph: &Graph, n: &Node) -> bool {
        fusible(graph, n)
            && !matches!(
                n.primitive,
                Primitive::Reshape { .. }
                    | Primitive::Slice { view: true, .. }
                    | Primitive::Concatenate { in_place: true, .. }
            )
    }

    /// Triton passes no inputs by value.
    fn scalars(&mut self) -> Option<&mut Vec<usize>> {
        None
    }

    /// Names the kernel by the body's kind.
    fn kernel(&mut self, body: &Graph, _: &[bool]) -> String {
        crate::codegen::fusion::triton::named(body)
    }
}

impl crate::codegen::Supports for Triton {
    /// Rejects graphs with half-precision transcendentals, which Triton's libdevice lacks.
    fn check(graph: &Graph) -> Result<(), String> {
        for node in graph.nodes() {
            let x = graph.type_of(node.inputs.first().copied().unwrap_or(node.output));

            if node.primitive.is_transcendental() && matches!(x.dtype, DType::F16 | DType::BF16) {
                return Err(format!(
                    "{}: has no {} kernel on Triton, got {x}: convert to float32 first (x.to(dtype=float32))",
                    node.primitive.name(),
                    x.dtype.name()
                ));
            }
        }

        Ok(())
    }
}
