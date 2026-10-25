mod fusion;
mod gemm;

use std::sync::OnceLock;

#[cfg(feature = "python")]
pub(crate) use crate::codegen::triton::Config;
use fusion::LoopFusion;
#[cfg(feature = "python")]
pub(crate) use gemm::set_gemm;

use super::common::autotune::{Autotuner, Kernels, Timer};
use super::common::simplify::Simplify;
use super::pass::{GraphModule, Pipeline};
use super::plan::{Buffer, Step, Tuned};
use super::{Options, Plan, cpu};
use crate::codegen::triton::{self as codegen, LOOP_CONFIGS, LoopConfig, UNTUNED_LOOP};
use crate::graph::{Graph, Primitive, TensorType};
use crate::stream::event::event_for;
use crate::{DType, Device, Scalar, Tensor, TensorOptions};

#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Kernel {
    pub name: String,
    pub source: String,
    pub by_value: Vec<bool>,
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
) -> Result<(), String>;

static LOOP: OnceLock<Loop> = OnceLock::new();

#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_loop(launch: Loop) {
    let _ = LOOP.set(launch);
}

pub(crate) fn compile(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    let mut module = GraphModule {
        graph: graph.clone(),
        device,
        options,
    };
    Pipeline::new("triton")
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
    let Primitive::Fusion { body, .. } = &step.primitive else {
        return None;
    };
    let by_value: Vec<bool> = step
        .inputs
        .iter()
        .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
        .collect();
    let (name, source) = codegen::kernel(body, &by_value);
    Some(Kernel {
        name,
        source,
        by_value,
    })
}

struct Loops;

impl Kernels for Loops {
    type Config = LoopConfig;

    fn configs(&self, step: &Step) -> Vec<LoopConfig> {
        match step.primitive {
            Primitive::Fusion { .. } => LOOP_CONFIGS.to_vec(),
            _ => Vec::new(),
        }
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

    fn time(
        &self,
        (kernel, inputs, out): &Self::Buffers,
        step: &Step,
        reps: usize,
    ) -> Result<f64, String> {
        let mut start = event_for(self.device, true)?;
        let mut end = event_for(self.device, true)?;
        start.record()?;
        for _ in 0..reps {
            run(kernel.as_ref(), step, inputs, out)?;
        }
        end.record()?;
        end.synchronize()?;
        Ok(start.elapsed_time(end.as_ref())? / 1e3 / reps as f64)
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
        (Primitive::Fusion { .. }, Some(kernel)) => {
            let mut functions: Vec<Function> = inputs
                .iter()
                .enumerate()
                .filter(|&(k, _)| !kernel.by_value[k])
                .map(|(k, t)| {
                    Function::new(format!("LOAD{k}"), codegen::load(t.shape(), t.strides()))
                })
                .collect();
            functions.push(Function::new(
                "STORE",
                codegen::store(out.shape(), out.strides()),
            ));
            let config = step
                .launch::<LoopConfig>()
                .unwrap_or(&LOOP_CONFIGS[UNTUNED_LOOP]);
            launch(kernel, config, &functions, inputs, out)
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
    };
    let store = Function::new("STORE", codegen::store(t.shape(), t.strides()));
    let bits = Tensor::from_slice(&[codegen::bits(t.dtype(), value)], DType::U64);
    launch(&kernel, &LOOP_CONFIGS[UNTUNED_LOOP], &[store], &[bits], t)
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
    )
}

pub(crate) fn reads_strided(step: &Step) -> bool {
    matches!(
        step.primitive,
        Primitive::DotGeneral { .. } | Primitive::Fusion { .. }
    )
}

fn launch(
    kernel: &Kernel,
    config: &LoopConfig,
    functions: &[Function],
    inputs: &[Tensor],
    out: &Tensor,
) -> Result<(), String> {
    let launch = LOOP.get().ok_or(
        "the Triton backend's loops launch from Python (lumen.codegen.triton): none is installed",
    )?;
    launch(kernel, config, functions, inputs, out)
}
