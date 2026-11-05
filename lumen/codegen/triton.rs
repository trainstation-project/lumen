//! Triton codegen: each planned step's loop kernel, generated once when the plan compiles (its `BODY`, for a
//! templated kernel, generated with its tensors' layouts at launch).

use std::sync::Arc;

use crate::DType;
use crate::codegen::kinds::{gemm_dot, reduction_root};
use crate::codegen::layout::{TILE, transpose_tiling};
use crate::codegen::reduction::ReductionCodegen;
use crate::codegen::row::RowCodegen;
use crate::codegen::scan::ScanCodegen;
use crate::codegen::scan::triton::Scan;
use crate::codegen::utils::triton as codegen;
use crate::graph::{Primitive, UPDATES};
use crate::graph_compiler::Options;
use crate::graph_compiler::passes::horizontal::segments;
use crate::graph_compiler::passes::scatter_add::triton::Scatter;
use crate::plan::{Buffer, Plan, Step};

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

/// Generates each step's loop kernel, kept on the step (none for a step another kernel runs), and marks `plan`
/// as Triton's.
pub(crate) fn codegen(plan: &mut Plan, options: &Options) -> Result<(), String> {
    let deterministic = options.config.deterministic;

    for step in plan.steps_mut() {
        if let Some(kernel) = loop_kernel(step, deterministic)? {
            step.kernel = Some(Arc::new(kernel));
        }
    }

    plan.triton = true;
    Ok(())
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
pub(crate) fn loop_kernel(step: &Step, deterministic: bool) -> Result<Option<Kernel>, String> {
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
        Primitive::Fusion { name, body, .. } if name != ROW && reduction_root(body).is_none() => {
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
            if name != UPDATES && name != ROW && reduction_root(body).is_none() =>
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
fn scan(step: &Step) -> Result<Scan, String> {
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
