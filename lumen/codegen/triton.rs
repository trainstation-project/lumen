//! Triton codegen: each planned step's loop kernel, generated once when the plan compiles (its `BODY`, for a
//! templated kernel, generated with its tensors' layouts at launch).

use crate::DType;
use crate::autotune::Autotuner;
use crate::codegen::attention::triton::FlashAttention;
use crate::codegen::dot_general::triton::Gemm;
use crate::codegen::elementwise::triton::Loop;
use crate::codegen::kernel::Runner;
use crate::codegen::kinds::{combine_root, gemm_dot, reduction_root};
use crate::codegen::layout::{TILE, transpose_tiling};
use crate::codegen::reduction::ReductionCodegen;
use crate::codegen::row::RowCodegen;
use crate::codegen::scan::ScanCodegen;
use crate::codegen::scan::triton::Scan;
use crate::codegen::scatter_add::triton::launch::Scatter;
use crate::codegen::utils::triton as codegen;
use crate::graph::{Graph, Node, Primitive, UPDATES};
use crate::graph_compiler::device::{Device, DeviceModule, Triton};
use crate::graph_compiler::passes::fusion::triton::{COMBINE, fusible, named};
use crate::graph_compiler::passes::horizontal::segments;
use crate::graph_compiler::{Options, plan_options, positions};
use crate::plan::step::Tensors;
use crate::plan::step::triton::{Collective, TritonTimer, Unsupported, Wait};
use crate::plan::{Buffer, CpuPlan, Plan, Step as _, StepBase, TritonPlan, TritonStep};
use std::sync::Arc;

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
fn kernel_of(step: &StepBase, looped: bool) -> Arc<dyn Runner<TritonStep, Tensors>> {
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

/// Whether Triton's `tl.atomic_add` adds elements of `dtype`: floats from 16 to 64 bits and 32- or 64-bit
/// integers, not bool, float8 or narrower integers.
fn atomic_add(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F16
            | DType::BF16
            | DType::F32
            | DType::F64
            | DType::I32
            | DType::I64
            | DType::U32
            | DType::U64
    )
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
            let r = codegen::Triton::row(body, &by_value, &())
                .map_err(|e| format!("{}: {e}", step.label))?;

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
                let r = codegen::Triton::reduction(body, &by_value, root)
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

/// How the scan template tiles cumsum `step`.
fn scan(step: &StepBase) -> Result<Scan, String> {
    let Primitive::Cumsum {
        axis,
        reverse,
        accum_dtype,
    } = step.primitive
    else {
        unreachable!("a cumsum")
    };

    codegen::Triton::scan(&step.inputs[0].1, axis, reverse, accum_dtype)
        .map_err(|e| format!("{}: {e}", step.label))
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
        let mut plan = crate::codegen::triton::codegen(plan, options)?;

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
        named(body)
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

/// Triton kernels read each element's partner at `j ^ 1`, or swap a GEMM accumulator tile's adjacent lanes.
impl crate::codegen::SwapsPairs for Triton {}

/// Triton runs batched inner products as elementwise and reduction fusions.
impl crate::codegen::InnerProducts for Triton {}

/// Triton's fusion fuses rows of what it fuses (the rows MPS finds), column sums riding along within
/// `row::triton::ROW_PARTIALS`.
impl crate::codegen::NormalizationRows for Triton {
    const PARTIALS: crate::graph_compiler::passes::diamonds::Partials =
        crate::codegen::row::triton::ROW_PARTIALS;

    /// `fusion::triton::fusible`.
    fn fusible(graph: &Graph, node: &Node) -> bool {
        crate::graph_compiler::passes::fusion::triton::fusible(graph, node)
    }
}

/// Tiles a Triton split aims for: on an H100, outputs of 64 or more 128x128 tiles already keep the GPU busy (a
/// 1024x1024 bfloat16 matmul is slower split), while 16 tiles gain about 2x.
const SPLIT_TARGET: usize = 128;

/// Triton splits with the untuned GEMM config's tiles, aiming at `SPLIT_TARGET`, keeping matrix-vector products whole
/// for the GEMV kernel (its lanes already split K, as MPS keeps its GEMVs). Its partials are always summed by a
/// reduction (no atomics), so results stay deterministic.
impl crate::codegen::SplitsK for Triton {
    const TARGET: usize = SPLIT_TARGET;

    /// Tiles of the untuned GEMM config.
    fn tiles(m: usize, n: usize) -> usize {
        use crate::codegen::dot_general::triton::{CONFIGS, UNTUNED};
        let tile = CONFIGS[UNTUNED];
        m.div_ceil(tile.block_m) * n.div_ceil(tile.block_n)
    }

    /// Matrix-vector products (`dot_general::triton::gemv`).
    fn unsplit(
        dot: &Primitive,
        lhs: &crate::graph::TensorType,
        rhs: &crate::graph::TensorType,
        _: &crate::graph::TensorType,
    ) -> bool {
        crate::codegen::dot_general::triton::gemv(dot, lhs, rhs)
    }
}

/// Triton's GEMM writes a wide row's partials over `partial_tile`-column blocks, except on the GEMV kernel.
impl crate::codegen::PartialRows for Triton {
    /// `dot_general::triton::partial_tile`.
    fn partial_tile(n: usize) -> Option<usize> {
        crate::codegen::dot_general::triton::partial_tile(n)
    }

    /// `dot_general::triton::gemv_body`.
    fn gemv(body: &Graph) -> bool {
        crate::codegen::dot_general::triton::gemv_body(body)
    }
}

/// Flash-attention matching for Triton, as on MPS: elementwise epilogues reading any leaf, runtime causal
/// offsets (decode).
impl crate::graph_compiler::passes::attention::Attentions for DeviceModule<'_, Triton> {
    /// Elementwise epilogues are fused into the forward kernel's output store.
    fn contraction_epilogues(&self) -> bool {
        true
    }

    /// Graph values of the by-value scalar inputs.
    fn scalars(&self) -> Vec<crate::graph::Var> {
        crate::graph_compiler::positions(&self.options.scalars)
            .into_iter()
            .map(|i| self.graph.inputs()[i])
            .collect()
    }

    /// Every forward attention runs on the one forward kernel.
    fn kernel(&mut self) -> String {
        crate::graph_compiler::passes::attention::FORWARD.into()
    }
}

/// Each attention backward runs as one kernel adding dQ atomically as it computes dK and dV, or, when
/// `deterministic`, a dK/dV kernel and a dQ kernel.
impl crate::graph_compiler::passes::attention::AttentionBackwards for DeviceModule<'_, Triton> {
    /// Every backward runs as one kernel unless deterministic, so dQ is summed over key blocks in a fixed order.
    fn together(
        &self,
    ) -> impl Fn(&crate::graph_compiler::passes::attention::Backward) -> bool + 'static {
        let deterministic = self.options.config.deterministic;
        move |_| !deterministic
    }

    /// Every backward fusion runs on the backward kernels.
    fn backward_kernel(&mut self) -> String {
        crate::graph_compiler::passes::attention::BACKWARD.into()
    }
}

/// Triton merges, with `fuse`, non-trivial `elementwise` fusions without reductions or matmuls, grouped by tiling.
impl crate::codegen::HorizontallyFuses for Triton {
    const GROUPING: crate::graph_compiler::passes::horizontal::Grouping =
        crate::graph_compiler::passes::horizontal::Grouping::Tiling;
    const BUFFERS: usize = crate::graph_compiler::passes::fusion::triton::HORIZONTAL_BUFFERS;

    /// `fusion::triton::horizontal_candidate`.
    fn candidate(graph: &Graph, node: &Node) -> bool {
        crate::graph_compiler::passes::fusion::triton::horizontal_candidate(graph, node)
    }
}

/// Triton runs a linear cross entropy as its GEMM with a per-tile max/sum epilogue, then a combine
/// (`dot_general::triton::cross_entropy`).
impl crate::codegen::LinearCrossEntropies for Triton {}

/// Triton merges with `fuse`.
impl crate::codegen::MergesDots for Triton {
    /// What Triton fusion can fuse, read in place; views; lanes 1; 8 columns.
    fn target() -> crate::graph_compiler::passes::merge_dots::Target {
        crate::graph_compiler::passes::merge_dots::Target {
            reads: crate::graph_compiler::passes::fusion::triton::fusible,
            views: true,
            lanes: 1,
            columns: 8,
        }
    }
}

/// Triton's kernels writing a concatenation's operands into their slots, with `fuse`: kernels of their own writing
/// fresh memory (`fusion::triton::slot_writes`).
impl crate::codegen::Slots for DeviceModule<'_, Triton> {
    /// What `fusion::triton::slot_writes` accepts.
    fn writes(&self, _: &Node, _: usize, producer: &Node, _: usize) -> bool {
        crate::graph_compiler::passes::fusion::triton::slot_writes(&self.graph, producer)
    }
}

/// Triton's GEMM reads its operands at any strides (its prologues are generated for their layouts).
impl crate::codegen::StridedOperands for Triton {
    /// Any layout.
    fn reads_strided(
        _: &Primitive,
        _: [&crate::graph::TensorType; 2],
        _: usize,
        _: &[usize],
    ) -> bool {
        true
    }
}

/// Triton kernels gather (a fused `Gather`).
impl crate::codegen::Gathers for Triton {}
