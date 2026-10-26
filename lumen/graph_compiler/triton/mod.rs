mod fusion;
mod gemm;

use std::sync::OnceLock;

#[cfg(feature = "python")]
pub(crate) use crate::codegen::triton::Config;
use fusion::LoopFusion;
#[cfg(feature = "python")]
pub(crate) use gemm::set_gemm;

use super::common::autotune::{Autotuner, Kernels, Timer};
use super::common::clamp_indices::ClampIndices;
use super::common::simplify::Simplify;
use super::pass::{GraphModule, Pipeline};
use super::plan::{Buffer, Step, Tuned};
use super::{Options, Plan, cpu};
use crate::codegen::triton::{self as codegen, LOOP_CONFIGS, LoopConfig, UNTUNED_LOOP};
use crate::graph::{Graph, Primitive, TensorType};
use crate::{DType, Device, Scalar, Tensor, TensorOptions};

#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Kernel {
    pub name: String,
    pub source: String,
    pub by_value: Vec<bool>,
    pub lanes: usize,
    pub splits: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Function {
    pub param: String,
    pub name: String,
    pub source: String,
}

impl Function {
    fn new(param: impl Into<String>, (name, source): (String, String)) -> Self {
        Function {
            param: param.into(),
            name,
            source,
        }
    }
}

pub type Loop = fn(
    kernel: &Kernel,
    config: &LoopConfig,
    functions: &[Function],
    inputs: &[Tensor],
    out: &Tensor,
    n: usize,
    programs: usize,
) -> Result<(), String>;

static LOOP: OnceLock<Loop> = OnceLock::new();

#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_loop(launch: Loop) {
    let _ = LOOP.set(launch);
}

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

pub(crate) fn compile(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    unsupported(graph)?;
    let mut module = GraphModule {
        graph: graph.clone(),
        device,
        options,
    };
    let mut pipeline = Pipeline::new("triton");
    if options.config.cuda.safe_kernels {
        pipeline.add(ClampIndices);
    }
    pipeline
        .add(Simplify { matched: true })
        .add(LoopFusion)
        .run(&mut module)?;
    let mut plan = cpu::plan(&module.graph, device, options);
    let kernels = plan.steps().iter().map(loop_kernel);
    plan.triton = Some(kernels.collect());
    if options.config.cuda.autotune {
        let tuner = Autotuner {
            timer: Events { device },
        };
        tuner.tune(&mut gemm::Gemms, &mut plan)?;
        tuner.tune(&mut Loops, &mut plan)?;
    }
    Ok(plan)
}

fn loop_kernel(step: &Step) -> Option<Kernel> {
    let by_value: Vec<bool> = step
        .inputs
        .iter()
        .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
        .collect();
    let (name, source, lanes, splits) = match &step.primitive {
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
    Some(Kernel {
        name,
        source,
        by_value,
        lanes,
        splits,
    })
}

struct Loops;

impl Kernels for Loops {
    type Config = LoopConfig;

    fn configs(&self, step: &Step) -> Vec<LoopConfig> {
        let Some(kernel) = loop_kernel(step) else {
            return Vec::new();
        };
        let widest = kernel.lanes.max(kernel.splits.next_power_of_two());
        LOOP_CONFIGS
            .into_iter()
            .filter(|c| c.block_size >= widest)
            .collect()
    }

    fn build(&mut self, _: &Step, config: LoopConfig) -> Tuned {
        Tuned::new(format!("{config:?}"), config)
    }
}

struct Events {
    device: Device,
}

impl Timer for Events {
    type Buffers = (Option<Kernel>, Vec<Tensor>, Tensor);

    fn device(&self) -> Device {
        self.device
    }

    fn buffers(&self, step: &Step) -> Result<Option<Self::Buffers>, String> {
        let zeros = |ty: &TensorType, device| {
            Tensor::zeros(
                &ty.shape,
                TensorOptions::new().dtype(ty.dtype).device(device),
            )
        };
        let inputs = step
            .inputs
            .iter()
            .map(|(b, ty)| match b {
                Buffer::Scalar(_) => zeros(ty, Device::Cpu(0)),
                _ => zeros(ty, self.device),
            })
            .collect();
        Ok(Some((
            loop_kernel(step),
            inputs,
            zeros(&step.output.1, self.device),
        )))
    }

    fn run(
        &self,
        (kernel, inputs, out): &Self::Buffers,
        step: &Step,
        _: usize,
    ) -> Result<(), String> {
        run(kernel.as_ref(), step, inputs, out)
    }
}

pub(crate) fn run(
    kernel: Option<&Kernel>,
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
) -> Result<(), String> {
    match (&step.primitive, kernel) {
        (Primitive::DotGeneral { .. }, _) => gemm::run(step, &inputs[0], &inputs[1], out),
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
            let config = step
                .launch::<LoopConfig>()
                .unwrap_or(&LOOP_CONFIGS[UNTUNED_LOOP]);
            let n = match step.primitive {
                Primitive::Cumsum { axis, .. } => out.numel().checked_div(out.shape()[axis]),
                Primitive::DynamicUpdateSlice => Some(inputs[1].numel()),
                Primitive::ScatterAdd { axis } => out.numel().checked_div(out.shape()[axis]),
                _ => Some(out.numel()),
            };
            if let Primitive::DynamicUpdateSlice | Primitive::ScatterAdd { .. } = step.primitive
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
            };
            let inputs = [inputs, std::slice::from_ref(&partials)].concat();
            launch(&last, config, &functions, &inputs, out, n)
        }
        _ => Err(format!("{}: has no Triton kernel on CUDA yet", step.label)),
    }
}

pub(crate) fn fill(t: &Tensor, value: Scalar) -> Result<(), String> {
    let (name, source) = codegen::fill(t.dtype());
    let kernel = Kernel {
        name,
        source,
        by_value: vec![true],
        lanes: 1,
        splits: 1,
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

pub(crate) fn reads_strided(step: &Step) -> bool {
    matches!(
        step.primitive,
        Primitive::DotGeneral { .. } | Primitive::Fusion { .. } | Primitive::Cumsum { .. }
    )
}

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
    let programs = n.div_ceil(config.block_size / kernel.lanes) * kernel.splits;
    launch(kernel, config, functions, inputs, out, n, programs)
}
