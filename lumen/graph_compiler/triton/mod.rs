//! The Triton backend for CUDA: pass pipeline, plan compilation, autotuning, and running steps
//! through Triton kernels launched from Python.

mod attention;
mod diamonds;
mod fusion;
mod gemm;
mod horizontal;
mod place_concats;
mod split_k;

use std::sync::OnceLock;

#[cfg(feature = "python")]
pub(crate) use crate::codegen::triton::Config;
use attention::{FlashAttentionBackward, Tiles};
#[cfg(feature = "python")]
pub(crate) use attention::{Launch, set_attention};
use diamonds::{Diamonds, DotRows};
use fusion::LoopFusion;
#[cfg(feature = "python")]
pub(crate) use gemm::{Epilogue, set_gemm};
use horizontal::HorizontalFusion;
use place_concats::PlaceConcats;
use split_k::SplitK;

use super::common::autotune::{Autotuner, Kernels, Timer};
use super::common::clamp_indices::ClampIndices;
use super::common::flash_attention::FlashAttention;
use super::common::merge_dots::{Blocks, MergeDots, Packs, Target};
use super::common::merge_dots_backward::MergeDotsBackward;
use super::common::simplify::Simplify;
use super::common::updates::Updates;
use super::pass::{Module, Pipeline};
use super::plan::{Buffer, Step, Tuned};
use super::{CompilerConfig, Options, Plan, PlanOptions, positions};
use crate::codegen::triton::{self as codegen, LOOP_CONFIGS, LoopConfig, UNTUNED_LOOP};
use crate::graph::{Graph, Primitive, TensorType, UPDATES};
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

/// Launches a loop kernel over `n` outputs on `programs` programs with the given helper functions.
pub type Loop = fn(
    kernel: &Kernel,
    config: &LoopConfig,
    functions: &[Function],
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
    };
    let mut pipeline = Pipeline::new("triton");
    if config.merge_dots {
        pipeline.add(MergeDotsBackward).add(MergeDots {
            target: Target {
                reads: codegen::fusible,
                views: true,
                lanes: 1,
                columns: 8,
            },
        });
    }
    if config.flash_attention {
        pipeline.add(FlashAttention).add(FlashAttentionBackward);
    }
    if config.safe_kernels {
        pipeline.add(ClampIndices);
    }
    pipeline.add(Simplify { matched: true });
    if config.split_k {
        pipeline.add(SplitK);
    }
    if config.normalization_diamonds {
        pipeline.add(Diamonds).add(DotRows);
    }
    pipeline.add(Updates {
        least: 2,
        most: usize::MAX,
    });
    pipeline.add(PlaceConcats);
    pipeline.add(LoopFusion {
        multi_output: config.multi_output_fusion,
    });
    if config.horizontal_fusion {
        pipeline.add(HorizontalFusion);
    }
    pipeline.run(&mut module)?;
    let TritonModule {
        graph: fused,
        packed,
        interleaved,
        ..
    } = module;
    // Packed blocks are appended as inputs after the original ones.
    let parameters = options.parameters.clone().map(|mut p| {
        p.resize(fused.inputs().len(), true);
        p
    });
    let first_block = fused.inputs().len() - packed.len();
    let blocks = packed
        .iter()
        .enumerate()
        .map(|(k, (members, _))| (first_block + k, members.clone()))
        .collect();
    let mut plan = Plan::compile_with(
        &fused,
        &PlanOptions {
            scratch: None,
            donate: options.donate.clone(),
            donate_into: options.donate_into.clone(),
            parameters,
            scalars: positions(&options.scalars),
            memory_limit: config.memory_limit(),
            blocks,
            layout: None,
        },
    );
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

/// Generates the loop kernel for `step`, or `None` for GEMMs and steps without a Triton loop kernel.
fn loop_kernel(step: &Step) -> Option<Kernel> {
    let by_value: Vec<bool> = step
        .inputs
        .iter()
        .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
        .collect();
    let (name, source, lanes, splits) = match &step.primitive {
        Primitive::Fusion { name, body, .. } if name == UPDATES => {
            let chain = crate::graph_compiler::common::updates::starts(body);
            let starts: Vec<Vec<i64>> = chain.iter().map(|(_, s)| s.clone()).collect();
            let (name, source) = codegen::updates(&step.inputs[0].1, &step.inputs[1].1, &starts);
            (name, source, 1, 1)
        }
        Primitive::Fusion { body, .. } if codegen::gemm_dot(body).is_some() => return None,
        Primitive::Fusion { name, body, .. } if name == ROW => {
            let r = codegen::row(body, &by_value);
            (r.name, r.source, r.lanes, r.splits)
        }
        Primitive::Fusion { body, .. } => match codegen::reduction_root(body) {
            Some(root) => {
                let r = codegen::reduction(body, &by_value, root);
                (r.name, r.source, r.lanes, r.splits)
            }
            None => {
                let (name, source) = codegen::kernel(body, &by_value);
                (name, source, 1, 1)
            }
        },
        Primitive::Cumsum {
            axis,
            reverse,
            accum_dtype,
        } => {
            let (name, source, lanes) =
                codegen::scan(&step.inputs[0].1, *axis, *reverse, *accum_dtype);
            (name, source, lanes, 1)
        }
        Primitive::DynamicUpdateSlice => {
            let (name, source) = codegen::update(&step.inputs[0].1, &step.inputs[1].1, &by_value);
            (name, source, 1, 1)
        }
        Primitive::ScatterAdd { axis } => {
            let (name, source) =
                codegen::scatter_add(&step.inputs[0].1, *axis, step.inputs[1].1.numel());
            (name, source, 1, 1)
        }
        _ => return None,
    };
    let segments = match &step.primitive {
        Primitive::Fusion { body, .. } if codegen::reduction_root(body).is_none() => segments(body),
        _ => Vec::new(),
    };
    let segments = match segments.len() {
        0 | 1 => Vec::new(),
        _ => segments.into_iter().map(|(numel, _)| numel).collect(),
    };
    Some(Kernel {
        name,
        source,
        by_value,
        lanes,
        splits,
        segments,
    })
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
        codegen::loop_configs(kernel.lanes.max(kernel.splits.next_power_of_two()))
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
            let loads: Vec<Function> = inputs
                .iter()
                .enumerate()
                .filter(|&(k, _)| !kernel.by_value[k])
                .map(|(k, t)| {
                    Function::new(format!("LOAD{k}"), codegen::load(t.shape(), t.strides()))
                })
                .collect();
            let store = Function::new("STORE", codegen::store(out.shape(), out.strides()));
            let mut functions = [loads.as_slice(), std::slice::from_ref(&store)].concat();
            // An elementwise or row fusion's other outputs follow its inputs, each with its own store.
            let outs: &[Tensor] = match &step.primitive {
                Primitive::Fusion { name, body, .. }
                    if name == ROW || codegen::reduction_root(body).is_none() =>
                {
                    outs
                }
                _ => &[],
            };
            for (k, t) in outs.iter().enumerate() {
                functions.push(Function::new(
                    format!("STORE{}", k + 1),
                    codegen::store(t.shape(), t.strides()),
                ));
            }
            let inputs: &[Tensor] = &[inputs, outs].concat();
            let untuned = codegen::untuned_loop(kernel.lanes);
            let config = step.launch::<LoopConfig>().unwrap_or(&untuned);
            // A looped row kernel walks one row a program, a block of it at a time.
            let looped;
            let kernel = match &step.primitive {
                Primitive::Fusion { name, body, .. }
                    if name == ROW && codegen::row_looped(body) =>
                {
                    looped = Kernel {
                        lanes: config.block_size,
                        ..kernel.clone()
                    };
                    &looped
                }
                _ => kernel,
            };
            // Grid size: outputs, or rows for a row fusion or lines for a scan, or picks for a scatter.
            let n = match &step.primitive {
                Primitive::Fusion { name, .. } if name == ROW => out
                    .numel()
                    .checked_div(out.shape().last().copied().unwrap_or(1)),
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
            if let Primitive::ScatterAdd { .. } = step.primitive {
                if out.dtype() == DType::Bool {
                    return Err(format!(
                        "{}: booleans have no sum to scatter-add",
                        step.label
                    ));
                }
                functions.push(Function::new(
                    "OUTLOAD",
                    codegen::load(out.shape(), out.strides()),
                ));
            }
            let n = n.unwrap_or(0);
            if kernel.splits == 1 {
                return launch(kernel, config, &functions, inputs, out, n);
            }
            // A split reduction writes per-split partials, then a second kernel combines them into `out`.
            let Primitive::Fusion { body, .. } = &step.primitive else {
                unreachable!("a split reduction")
            };
            let reduced = codegen::reduction_root(body).expect("a reduction").output;
            let options = TensorOptions::new()
                .dtype(body.type_of(reduced).dtype)
                .device(out.device());
            let partials = unsafe { Tensor::empty(&[n * kernel.splits], options) };
            launch(kernel, config, &loads, inputs, &partials, n)?;
            let last = Kernel {
                name: format!("{}_final", kernel.name),
                source: kernel.source.clone(),
                by_value: [kernel.by_value.as_slice(), &[false]].concat(),
                lanes: kernel.splits.next_power_of_two(),
                splits: 1,
                segments: Vec::new(),
            };
            let inputs = [inputs, std::slice::from_ref(&partials)].concat();
            launch(&last, config, &functions, &inputs, out, n)
        }
        _ => Err(format!("{}: has no Triton kernel on CUDA yet", step.label)),
    }
}

/// Fills `t` with `value` using a Triton kernel.
pub(crate) fn fill(t: &Tensor, value: Scalar) -> Result<(), String> {
    let (name, source) = codegen::fill(t.dtype());
    let kernel = Kernel {
        name,
        source,
        by_value: vec![true],
        lanes: 1,
        splits: 1,
        segments: Vec::new(),
    };
    let store = Function::new("STORE", codegen::store(t.shape(), t.strides()));
    let bits = Tensor::from_slice(&[codegen::bits(t.dtype(), value)], DType::U64);
    launch(
        &kernel,
        &LOOP_CONFIGS[UNTUNED_LOOP],
        &[store],
        &[bits],
        t,
        t.numel(),
    )
}

/// Copies `src` into `dst`, which may have different strides, using a Triton kernel.
pub(crate) fn copy(dst: &Tensor, src: &Tensor) -> Result<(), String> {
    let mut body = Graph::new();
    let x = body.input(TensorType::new(src.dtype(), src.shape()));
    body.set_outputs(&[x]).expect("the copied value");
    let (name, source) = codegen::kernel(&body, &[false]);
    let kernel = Kernel {
        name,
        source,
        by_value: vec![false],
        lanes: 1,
        splits: 1,
        segments: Vec::new(),
    };
    let functions = [
        Function::new("LOAD0", codegen::load(src.shape(), src.strides())),
        Function::new("STORE", codegen::store(dst.shape(), dst.strides())),
    ];
    launch(
        &kernel,
        &LOOP_CONFIGS[UNTUNED_LOOP],
        &functions,
        std::slice::from_ref(src),
        dst,
        dst.numel(),
    )
}

/// Whether the Triton kernel for `step` reads strided inputs directly.
pub(crate) fn reads_strided(step: &Step) -> bool {
    !attention::is_attention(&step.primitive)
        && matches!(
            step.primitive,
            Primitive::DotGeneral { .. } | Primitive::Fusion { .. } | Primitive::Cumsum { .. }
        )
}

/// Launches `kernel` over `n` outputs, with enough programs for the block size, lanes and splits.
fn launch(
    kernel: &Kernel,
    config: &LoopConfig,
    functions: &[Function],
    inputs: &[Tensor],
    out: &Tensor,
    n: usize,
) -> Result<(), String> {
    let launch = LOOP.get().ok_or(
        "the Triton backend's loops launch from Python (lumen.codegen.triton): none is installed",
    )?;
    let programs = match kernel.segments.as_slice() {
        [] => n.div_ceil(config.block_size / kernel.lanes) * kernel.splits,
        segments => segments.iter().map(|s| s.div_ceil(config.block_size)).sum(),
    };
    launch(kernel, config, functions, inputs, out, n, programs)
}
