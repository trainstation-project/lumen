//! The Triton backend for CUDA: pass pipeline, plan compilation, autotuning, and running steps
//! through Triton kernels launched from Python.

mod attention;
mod cross_entropy;
mod diamonds;
#[cfg(all(feature = "python", lumen_cuda_linked))]
pub(crate) mod direct;
mod fusion;
mod gemm;
mod horizontal;
mod place_concats;
mod split_k;
mod wide_rows;

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

#[cfg(feature = "python")]
pub(crate) use crate::codegen::dot_general::triton::Config;
use attention::Tiles;
#[cfg(feature = "python")]
pub(crate) use attention::{Launch, set_attention};
use diamonds::Diamonds;
use fusion::{Fusion, PlaceViews};
#[cfg(feature = "python")]
pub(crate) use gemm::{Epilogue, set_gemm};
use horizontal::HorizontalFusion;
use place_concats::PlaceConcats;
use split_k::SplitK;
use wide_rows::WideRows;

use super::common::alone::Alone;
use super::common::autotune::{Autotuner, Kernels, Timer};
use super::common::chunked_cross_entropy::ChunkedCrossEntropy;
use super::common::clamp_indices::ClampIndices;
use super::common::dot_strength::DotStrength;
use super::common::flash_attention::{FlashAttention, FlashAttentionBackward};
use super::common::linear_cross_entropy::LinearCrossEntropy;
use super::common::merge_dots::{Blocks, MergeDots, Packs, Target};
use super::common::merge_dots_backward::MergeDotsBackward;
use super::common::simplify::Simplify;
use super::common::take_along_axis::TakeAlongAxis;
use super::common::updates::Updates;
use super::pass::{Module, Pipeline};
use super::plan::{Buffer, Step, Tuned};
use super::{CompilerConfig, Options, Plan, plan_options, positions};
use crate::codegen::dynamic_slice::DynamicSliceCodegen;
use crate::codegen::elementwise::ElementwiseCodegen;
use crate::codegen::elementwise::triton::{
    LOOP_CONFIGS, LoopConfig, UNTUNED_LOOP, loop_configs, untuned_loop,
};
use crate::codegen::kinds::{gemm_dot, reduction_root};
use crate::codegen::layout::{LayoutCodegen, TILE, transpose_tiling};
use crate::codegen::reduction::ReductionCodegen;
use crate::codegen::row::RowCodegen;
use crate::codegen::row::triton::{row_count, row_groups, row_looped};
use crate::codegen::scan::ScanCodegen;
use crate::codegen::scan::triton::Scan;
use crate::codegen::store::triton::{load, store};
use crate::codegen::utils::triton as codegen;
use crate::graph::{Graph, Primitive, TensorType, UPDATES};
use crate::graph_compiler::common::diamonds::Row;
use crate::graph_compiler::common::horizontal::segments;
use crate::{DType, Device, Scalar, Tensor, TensorOptions};

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
    /// `scatter_add_kernel` (lumen.codegen.scatter_add.triton): blocks of the positions off the `length`-long
    /// axis (`after` inner ones per outer), adding `picks` updates at each.
    ScatterAdd {
        length: usize,
        after: usize,
        picks: usize,
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
            Template::ScatterAdd { .. } => {
                ("lumen.codegen.scatter_add.triton", "scatter_add_kernel")
            }
            Template::Scan(_) => ("lumen.codegen.scan.triton.kernel", "scan_kernel"),
        }
    }

    /// The kernel's constexprs for a launch over `n` outputs in `segments` (empty for one segment): a
    /// loop's segments' element counts, a transpose's output sides and tile, an update's element count, a
    /// cross entropy's rows and tiles, a scatter-add's positions off the axis and its sizes, a scan's lines
    /// and tiling.
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
                length,
                after,
                picks,
            } => vec![
                ("N", Constexpr::Int(n)),
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

/// A jit helper bound to a kernel's constexpr parameter.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Function {
    /// The kernel parameter it is passed as (e.g. `LOAD0`, `STORE`).
    pub param: String,
    pub name: String,
    pub source: String,
}

impl Function {
    /// Binds the generated `(name, source)` to parameter `param`.
    fn new(param: impl Into<String>, (name, source): (String, String)) -> Self {
        Function {
            param: param.into(),
            name,
            source,
        }
    }
}

/// The kernels a step's launches were compiled for, kept with its generated helpers: the launcher finds
/// a run's kernel here by its config and argument classes instead of building a key each run.
#[derive(Default)]
#[cfg_attr(not(lumen_cuda_linked), allow(dead_code))]
pub struct Launches(pub(crate) Mutex<Vec<Arc<dyn Any + Send + Sync>>>);

/// Launches a loop kernel over `n` outputs on `programs` programs with the given helper functions,
/// finding or recording its compiled kernel in `launches`.
pub type Loop = fn(
    kernel: &Kernel,
    config: &LoopConfig,
    functions: &[Function],
    launches: &Launches,
    inputs: &[Tensor],
    out: &Tensor,
    n: usize,
    programs: usize,
) -> Result<(), String>;

/// The loop launcher installed from Python.
static LOOP: OnceLock<Loop> = OnceLock::new();

/// Installs the loop launcher; later calls are ignored.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_loop(launch: Loop) {
    let _ = LOOP.set(launch);
}

/// Name of a normalization row fusion.
const ROW: &str = "row";

/// Rejects graphs with half-precision transcendentals, which Triton's libdevice lacks.
fn unsupported(graph: &Graph) -> Result<(), String> {
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

/// A graph compiled for Triton, with the packs merge_dots creates.
struct TritonModule<'a> {
    graph: Graph,
    options: &'a Options,
    /// Inputs concatenated into each packed block.
    packed: Packs,
    /// Interleaved region sizes of each packed block.
    interleaved: Vec<Vec<usize>>,
    /// Normalization rows the diamonds pass found, consumed by fusion.
    rows: Vec<Row>,
}

impl Module for TritonModule<'_> {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    fn options(&self) -> &Options {
        self.options
    }

    fn config(&self) -> &dyn CompilerConfig {
        &self.options.config.cuda
    }
}

impl Blocks for TritonModule<'_> {
    fn blocks(&mut self) -> (&mut Packs, &mut Vec<Vec<usize>>) {
        (&mut self.packed, &mut self.interleaved)
    }
}

/// Compiles `graph` for `device` with the Triton backend: runs the passes, plans the steps, generates
/// loop kernels and autotunes GEMM and loop configs when enabled.
pub(crate) fn compile(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    unsupported(graph)?;
    let config = &options.config.cuda;

    let mut module = TritonModule {
        graph: graph.clone(),
        options,
        packed: Vec::new(),
        interleaved: Vec::new(),
        rows: Vec::new(),
    };

    let mut pipeline = Pipeline::new("triton");
    pipeline.add(TakeAlongAxis);

    // With fusion off (`fuse`), as on MPS: no pass merges ops, each runs as its own kernel.
    if config.fuse && config.fused_linear_cross_entropy {
        pipeline.add(ChunkedCrossEntropy);
    }

    if config.fuse && config.merge_dots {
        pipeline.add(MergeDotsBackward).add(MergeDots {
            target: Target {
                reads: fusion::fusible,
                views: true,
                lanes: 1,
                columns: 8,
            },
        });
    }

    if config.fuse && config.flash_attention {
        pipeline.add(FlashAttention).add(FlashAttentionBackward);
    }

    if config.fuse && config.fused_linear_cross_entropy {
        pipeline.add(LinearCrossEntropy);
    }

    if config.safe_kernels {
        pipeline.add(ClampIndices);
    }

    // Batched inner products (attention backward's delta, read through transposes) run as elementwise fusions.
    pipeline.add(Simplify { matched: true }).add(DotStrength);

    if config.split_k {
        pipeline.add(SplitK);
    }

    // As on MPS: update chains, normalization rows, in-place concatenations, then fusion.
    if config.fuse {
        pipeline.add(Updates {
            least: 2,
            most: usize::MAX,
        });

        if config.normalization_diamonds {
            pipeline.add(Diamonds);
        }

        pipeline.add(PlaceConcats);
        pipeline.add(Fusion);

        // Rows too wide for one GEMM tile: the GEMM feeding them writes their reductions' partials too.
        if config.normalization_diamonds {
            pipeline.add(WideRows);
        }
    }

    pipeline.add(PlaceViews);
    pipeline.add(Alone);

    if config.fuse && config.horizontal_fusion {
        pipeline.add(HorizontalFusion);
    }

    pipeline.run(&mut module)?;

    let TritonModule {
        graph: fused,
        packed,
        interleaved,
        ..
    } = module;

    let scalars = positions(&options.scalars);
    let plan = plan_options(&fused, options, config, &packed, scalars);
    let mut plan = Plan::compile_with(&fused, &plan);
    plan.packed = packed;
    plan.interleaved = interleaved;
    let kernels = plan.steps().iter().map(loop_kernel);
    plan.triton = Some(kernels.collect());

    if options.config.cuda.autotune {
        let tuner = Autotuner {
            timer: Events { device },
        };

        tuner.tune(&mut gemm::Gemms, &mut plan)?;
        tuner.tune(&mut Tiles, &mut plan)?;
        tuner.tune(&mut Loops, &mut plan)?;
    }

    Ok(plan)
}

/// The name of a templated kernel: `kind` plus a hash of `key`, what its `BODY` is generated from (the
/// `BODY`, generated per layout at launch, is named after its own source).
fn templated(kind: &str, key: &str) -> String {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    format!("{kind}_{:016x}", hasher.finish())
}

/// Generates the loop kernel for `step`, or `None` for GEMMs and steps without a Triton loop kernel.
fn loop_kernel(step: &Step) -> Option<Kernel> {
    let by_value: Vec<bool> = step
        .inputs
        .iter()
        .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
        .collect();

    let (name, source, lanes, splits) = match &step.primitive {
        // A chain of updates runs in the dynamic-update-slice template, its `BODY` generated at launch.
        Primitive::Fusion { name, body, .. } if name == UPDATES => (
            templated("dynamic_update_slice", &body.to_string()),
            String::new(),
            1,
            1,
        ),
        Primitive::Fusion { body, .. } if gemm_dot(body).is_some() => return None,
        Primitive::Fusion { name, body, .. } if name == ROW => {
            let r = codegen::Triton::row(body, &by_value, &())
                .expect("a Triton row kernel of every row body");

            (r.name, r.source, r.lanes, r.splits)
        }
        Primitive::Fusion { body, .. } => match reduction_root(body) {
            Some(root) => {
                let r = codegen::Triton::reduction(body, &by_value, root)
                    .expect("a Triton reduction of every reduction body");

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

            (
                templated("scatter_add", &format!("{:?} {types:?}", step.primitive)),
                String::new(),
                1,
                1,
            )
        }
        _ => return None,
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
                length: x[*axis],
                after: x[axis + 1..].iter().product(),
                picks: step.inputs[1].1.numel(),
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

    Some(Kernel {
        name,
        source,
        by_value,
        lanes,
        splits,
        segments,
        template,
    })
}

/// How the scan template tiles cumsum `step`, or `None` for a step that is not one.
fn scan(step: &Step) -> Option<Scan> {
    let Primitive::Cumsum {
        axis,
        reverse,
        accum_dtype,
    } = step.primitive
    else {
        return None;
    };

    let scan = codegen::Triton::scan(&step.inputs[0].1, axis, reverse, accum_dtype);

    Some(scan.expect("a Triton scan of every cumsum"))
}

/// Autotuning candidates for loop kernels.
struct Loops;

impl Kernels for Loops {
    type Config = LoopConfig;

    /// Loop configs whose block covers the kernel's lanes and splits.
    fn configs(&self, step: &Step) -> Vec<LoopConfig> {
        let Some(kernel) = loop_kernel(step) else {
            return Vec::new();
        };

        loop_configs(kernel.lanes.max(kernel.splits.next_power_of_two()))
    }

    /// The step launched with `config`.
    fn build(&mut self, _: &Step, config: LoopConfig) -> Tuned {
        Tuned::new(format!("{config:?}"), config)
    }
}

/// Times steps on `device` with CUDA events.
struct Events {
    device: Device,
}

impl Timer for Events {
    type Buffers = (Option<Kernel>, Vec<Tensor>, Tensor, Vec<Tensor>);

    /// The device steps run on.
    fn device(&self) -> Device {
        self.device
    }

    /// Zeroed inputs and outputs for `step`; scalars live on the CPU, and in-place ops write their operand.
    fn buffers(&self, step: &Step) -> Result<Option<Self::Buffers>, String> {
        let zeros = |ty: &TensorType, device| {
            Tensor::zeros(
                &ty.shape,
                TensorOptions::new().dtype(ty.dtype).device(device),
            )
        };

        let inputs: Vec<Tensor> = step
            .inputs
            .iter()
            .map(|(b, ty)| match b {
                Buffer::Scalar(_) => zeros(ty, Device::Cpu(0)),
                _ => zeros(ty, self.device),
            })
            .collect();

        let out = match step.primitive {
            Primitive::DynamicUpdateSlice | Primitive::ScatterAdd { .. } => inputs[0].clone(),
            _ => zeros(&step.output.1, self.device),
        };

        let outs = step
            .extra_outputs
            .iter()
            .map(|(_, ty)| zeros(ty, self.device))
            .collect();

        Ok(Some((loop_kernel(step), inputs, out, outs)))
    }

    /// One run of the step.
    fn run(
        &self,
        (kernel, inputs, out, outs): &Self::Buffers,
        step: &Step,
        _: usize,
    ) -> Result<(), String> {
        run(kernel.as_ref(), step, inputs, out, outs)
    }
}

/// Runs `step` on the given buffers: GEMMs through the GEMM kernel, everything else through `kernel`.
pub(crate) fn run(
    kernel: Option<&Kernel>,
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
) -> Result<(), String> {
    match (&step.primitive, kernel) {
        (p, _) if attention::is_attention(p) => attention::run(step, inputs, out, outs),
        _ if gemm::is_gemm(step) => gemm::run(step, inputs, out, outs),
        #[cfg(lumen_cuda_linked)]
        (
            Primitive::AllReduce { .. }
            | Primitive::AllGather { .. }
            | Primitive::ReduceScatter { .. },
            _,
        ) => crate::distributed::nccl::run(&step.primitive, &inputs[0], out),
        (Primitive::DeviceWait, _) => {
            #[cfg(lumen_cuda_linked)]
            crate::distributed::nccl::device_wait(&inputs[0])?;

            match out.data_ptr() == inputs[0].data_ptr() {
                true => Ok(()),
                false => copy(out, &inputs[0]),
            }
        }
        (Primitive::HostWait, _) => {
            crate::stream::stream_for(out.device())?.synchronize();

            match out.data_ptr() == inputs[0].data_ptr() {
                true => Ok(()),
                false => copy(out, &inputs[0]),
            }
        }
        (
            Primitive::Fusion { .. }
            | Primitive::Cumsum { .. }
            | Primitive::DynamicUpdateSlice
            | Primitive::ScatterAdd { .. },
            Some(kernel),
        ) => {
            // A fusion's other outputs follow its inputs, each with its own store.
            let outs: &[Tensor] = match &step.primitive {
                Primitive::Fusion { .. } => outs,
                _ => &[],
            };

            if let Primitive::ScatterAdd { .. } = step.primitive
                && out.dtype() == DType::Bool
            {
                return Err(format!(
                    "{}: booleans have no sum to scatter-add",
                    step.label
                ));
            }
            let key = layouts(inputs.iter().chain([out]).chain(outs));

            let helpers = step
                .memo
                .get(key, || Ok(loop_helpers(step, kernel, inputs, out, outs)))?;

            let (functions, launches) = (&helpers.functions, &helpers.launches);
            let inputs: &[Tensor] = &[inputs, outs].concat();
            let untuned = untuned_loop(kernel.lanes);
            let config = step.launch::<LoopConfig>().unwrap_or(&untuned);
            // A looped row kernel walks one row a program, a block of it at a time; one with column
            // partials one group of rows a program; a tiled transpose one tile a program.
            let looped;

            let kernel = match &step.primitive {
                Primitive::Fusion { name, body, .. }
                    if name == ROW && (row_looped(body) || row_groups(body).is_some())
                        || name != ROW && transpose_tiling(body).is_some() =>
                {
                    looped = Kernel {
                        lanes: config.block_size,
                        ..kernel.clone()
                    };

                    &looped
                }
                _ => kernel,
            };

            // Grid size: outputs, or a row fusion's rows (its row groups with partials), or a tiled
            // transpose's tiles, or lines for a scan, or picks for a scatter.
            let n = match &step.primitive {
                Primitive::Fusion { name, body, .. } if name == ROW => {
                    Some(row_groups(body).unwrap_or(row_count(body)))
                }
                Primitive::Fusion { body, .. } if transpose_tiling(body).is_some() => {
                    transpose_tiling(body).map(|tiling| tiling.tiles())
                }
                Primitive::Cumsum { axis, .. } => out.numel().checked_div(out.shape()[*axis]),
                Primitive::DynamicUpdateSlice => Some(inputs[1].numel()),
                Primitive::Fusion { name, .. } if name == UPDATES => Some(inputs[1].numel()),
                Primitive::ScatterAdd { axis } => out.numel().checked_div(out.shape()[*axis]),
                _ => Some(out.numel()),
            };

            // In-place ops start from a copy of their operand.
            let updates =
                matches!(&step.primitive, Primitive::Fusion { name, .. } if name == UPDATES);

            if (updates
                || matches!(
                    step.primitive,
                    Primitive::DynamicUpdateSlice | Primitive::ScatterAdd { .. }
                ))
                && out.data_ptr() != inputs[0].data_ptr()
            {
                copy(out, &inputs[0])?;
            }

            let n = n.unwrap_or(0);

            if kernel.splits == 1 {
                return launch(kernel, config, functions, launches, inputs, out, n);
            }

            // A split reduction writes per-split partials, then a second kernel combines them into `out`.
            let Primitive::Fusion { body, .. } = &step.primitive else {
                unreachable!("a split reduction")
            };

            let reduced = reduction_root(body).expect("a reduction").output;

            let options = TensorOptions::new()
                .dtype(body.type_of(reduced).dtype)
                .device(out.device());

            let partials = unsafe { Tensor::empty(&[n * kernel.splits], options) };
            // The first pass also writes the fusion's other outputs; the second only the result.
            launch(kernel, config, functions, launches, inputs, &partials, n)?;

            let last = Kernel {
                name: format!("{}_final", kernel.name),
                source: kernel.source.clone(),
                by_value: [kernel.by_value.as_slice(), &[false]].concat(),
                lanes: kernel.splits.next_power_of_two(),
                splits: 1,
                segments: Vec::new(),
                template: None,
            };

            let inputs = [inputs, std::slice::from_ref(&partials)].concat();

            let finals: Vec<Function> = functions
                .iter()
                .filter(|f| f.param == "STORE" || !f.param.starts_with("STORE"))
                .cloned()
                .collect();

            launch(&last, config, &finals, launches, &inputs, out, n)
        }
        _ => Err(format!("{}: has no Triton kernel on CUDA yet", step.label)),
    }
}

/// A fill or copy kernel with its helpers for one layout of its tensors, and the kernels compiled for it.
struct Utility {
    kernel: Kernel,
    functions: Vec<Function>,
    launches: Launches,
}

/// The fill or copy kernel `make` builds for the tensors' dtypes, shapes and strides in `layouts`,
/// made once per layout.
fn utility(layouts: &[&Tensor], make: impl FnOnce() -> (Kernel, Vec<Function>)) -> Arc<Utility> {
    type Layout = (DType, Vec<usize>, Vec<usize>);
    static UTILITIES: Mutex<Option<HashMap<Vec<Layout>, Arc<Utility>>>> = Mutex::new(None);

    let key: Vec<Layout> = layouts
        .iter()
        .map(|t| (t.dtype(), t.shape().to_vec(), t.strides().to_vec()))
        .collect();

    let mut cache = UTILITIES.lock().unwrap_or_else(PoisonError::into_inner);
    let cache = cache.get_or_insert_with(HashMap::new);

    cache
        .entry(key)
        .or_insert_with(|| {
            let (kernel, functions) = make();

            Arc::new(Utility {
                kernel,
                functions,
                launches: Launches::default(),
            })
        })
        .clone()
}

/// Fills `t` with `value` using a Triton kernel.
pub(crate) fn fill(t: &Tensor, value: Scalar) -> Result<(), String> {
    let fill = utility(&[t], || {
        let (name, source) = codegen::fill(t.dtype());

        let kernel = Kernel {
            name,
            source,
            by_value: vec![true],
            lanes: 1,
            splits: 1,
            segments: Vec::new(),
            template: None,
        };

        let store = Function::new("STORE", store(t.shape(), t.strides()));
        (kernel, vec![store])
    });

    let bits = Tensor::from_slice(&[codegen::bits(t.dtype(), value)], DType::U64);

    launch(
        &fill.kernel,
        &LOOP_CONFIGS[UNTUNED_LOOP],
        &fill.functions,
        &fill.launches,
        &[bits],
        t,
        t.numel(),
    )
}

/// Copies `src` into `dst`, which may have different strides, using a Triton kernel.
pub(crate) fn copy(dst: &Tensor, src: &Tensor) -> Result<(), String> {
    let copy = utility(&[dst, src], || {
        let mut body = Graph::new();
        let x = body.input(TensorType::new(src.dtype(), src.shape()));
        body.set_outputs(&[x]).expect("the copied value");

        let layouts = codegen::Tensors {
            inputs: vec![Some((src.shape().to_vec(), src.strides().to_vec()))],
            outputs: vec![(dst.shape().to_vec(), dst.strides().to_vec())],
        };

        let generated = codegen::Triton::elementwise(&body, &[false], &HashMap::new(), &layouts)
            .expect("a Triton loop of a copy");

        let kernel = Kernel {
            name: generated.0.clone(),
            source: String::new(),
            by_value: vec![false],
            lanes: 1,
            splits: 1,
            segments: Vec::new(),
            template: Some(Template::Elementwise),
        };

        let functions = vec![Function::new("BODY", generated)];

        (kernel, functions)
    });

    launch(
        &copy.kernel,
        &LOOP_CONFIGS[UNTUNED_LOOP],
        &copy.functions,
        &copy.launches,
        std::slice::from_ref(src),
        dst,
        dst.numel(),
    )
}

/// A loop step's generated helpers for its tensors' layouts, and the kernels compiled for them.
struct Helpers {
    /// The loads of its tensor inputs, then the stores of its outputs; a templated step's `BODY` instead.
    functions: Vec<Function>,
    launches: Launches,
}

/// A loop step's helpers for its tensors' layouts: the loads of its tensor inputs, the stores of `out`
/// and `outs`, or a templated step's `BODY`.
fn loop_helpers(
    step: &Step,
    kernel: &Kernel,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
) -> Helpers {
    // A scatter-add's kernel is fixed: its tensors are contiguous, as it reads no strided inputs.
    if let Some(Template::ScatterAdd { .. }) = kernel.template {
        return Helpers {
            functions: Vec::new(),
            launches: Launches::default(),
        };
    }

    // A scan's kernel is fixed too, its tensors loaded and stored through the helpers below.
    if let Some(template) = kernel.template.filter(|t| !matches!(t, Template::Scan(_))) {
        let owned = |t: &Tensor| (t.shape().to_vec(), t.strides().to_vec());

        let layouts = codegen::Tensors {
            inputs: inputs
                .iter()
                .zip(&kernel.by_value)
                .map(|(t, &scalar)| (!scalar).then(|| owned(t)))
                .collect(),
            outputs: std::iter::once(out).chain(outs).map(owned).collect(),
        };

        let generated = match (template, &step.primitive) {
            (Template::Elementwise, Primitive::Fusion { body, .. }) => {
                codegen::Triton::elementwise(body, &kernel.by_value, &HashMap::new(), &layouts)
                    .expect("a Triton kernel of every elementwise body")
            }
            (Template::Transpose { .. }, Primitive::Fusion { body, .. }) => {
                codegen::Triton::transpose(body, &kernel.by_value, &layouts)
                    .expect("a Triton kernel of every tiled transpose")
            }
            (Template::DynamicUpdateSlice, Primitive::Fusion { body, .. }) => {
                codegen::Triton::updates(body, &layouts)
                    .expect("a Triton kernel of every updates body")
            }
            (Template::DynamicUpdateSlice, _) => codegen::Triton::update(
                &step.inputs[0].1,
                &step.inputs[1].1,
                &kernel.by_value,
                &layouts,
            )
            .expect("a Triton kernel of every dynamic_update_slice"),
            _ => unreachable!("a templated kernel's fusion"),
        };

        return Helpers {
            functions: vec![Function::new("BODY", generated)],
            launches: Launches::default(),
        };
    }

    let mut functions: Vec<Function> = inputs
        .iter()
        .enumerate()
        .filter(|&(k, _)| !kernel.by_value[k])
        .map(|(k, t)| Function::new(format!("LOAD{k}"), load(t.shape(), t.strides())))
        .collect();

    functions.push(Function::new("STORE", store(out.shape(), out.strides())));

    for (k, t) in outs.iter().enumerate() {
        functions.push(Function::new(
            format!("STORE{}", k + 1),
            store(t.shape(), t.strides()),
        ));
    }

    Helpers {
        functions,
        launches: Launches::default(),
    }
}

/// A hash of `tensors`' dtypes, shapes and strides: the key of what a step generated for those layouts.
pub(crate) fn layouts<'a>(tensors: impl IntoIterator<Item = &'a Tensor>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    for t in tensors {
        (t.dtype(), t.shape(), t.strides()).hash(&mut hasher);
    }

    hasher.finish()
}

/// Whether the Triton kernel for `step` reads strided inputs directly (attention any layout with a
/// contiguous last dim, copying others itself).
pub(crate) fn reads_strided(step: &Step) -> bool {
    matches!(
        step.primitive,
        Primitive::DotGeneral { .. }
            | Primitive::Fusion { .. }
            | Primitive::Cumsum { .. }
            | Primitive::LinearCrossEntropy
    )
}

/// Launches `kernel` over `n` outputs, with enough programs for the block size, lanes and splits; its
/// compiled kernels are kept in `launches`.
fn launch(
    kernel: &Kernel,
    config: &LoopConfig,
    functions: &[Function],
    launches: &Launches,
    inputs: &[Tensor],
    out: &Tensor,
    n: usize,
) -> Result<(), String> {
    let launch = LOOP.get().ok_or(
        "the Triton backend's loops launch from Python (lumen.codegen.utils.triton): none is installed",
    )?;

    let programs = match kernel.segments.as_slice() {
        [] => n.div_ceil(config.block_size / kernel.lanes) * kernel.splits,
        segments => segments.iter().map(|s| s.div_ceil(config.block_size)).sum(),
    };

    launch(
        kernel, config, functions, launches, inputs, out, n, programs,
    )
}
