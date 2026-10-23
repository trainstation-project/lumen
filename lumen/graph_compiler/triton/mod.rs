mod fusion;
mod gemm;

use std::sync::OnceLock;

use fusion::LoopFusion;
#[cfg(feature = "python")]
pub(crate) use crate::codegen::triton::Config;
#[cfg(feature = "python")]
pub(crate) use gemm::set_gemm;

use super::common::autotune::Autotuner;
use super::common::simplify::Simplify;
use super::pass::{GraphModule, Pipeline};
use super::plan::{Buffer, Step};
use super::{Options, Plan, cpu};
use crate::codegen::triton as codegen;
use crate::graph::{Graph, Primitive, TensorType};
use crate::{DType, Device, Scalar, Tensor};

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
        options,
    };
    Pipeline::new("triton")
        .add(Simplify { matched: true })
        .add(LoopFusion)
        .run(&mut module)?;
    let mut plan = cpu::plan(&module.graph, options);
    let kernels = plan.steps().iter().map(|step| match &step.primitive {
        Primitive::Fusion { body, .. } => {
            let by_value: Vec<bool> = step
                .inputs
                .iter()
                .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
                .collect();
            let (name, source) = crate::codegen::triton::kernel(body, &by_value);
            Some(Kernel {
                name,
                source,
                by_value,
            })
        }
        _ => None,
    });
    plan.triton = Some(kernels.collect());
    if options.config.autotune {
        let tuner = Autotuner {
            timer: gemm::Events { device },
        };
        tuner.tune(&mut gemm::Gemms, &plan)?;
    }
    Ok(plan)
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
            launch(kernel, &functions, inputs, out)
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
    launch(&kernel, &[store], &[bits], t)
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
    launch(&kernel, &functions, std::slice::from_ref(src), dst)
}

pub(crate) fn reads_strided(step: &Step) -> bool {
    matches!(
        step.primitive,
        Primitive::DotGeneral { .. } | Primitive::Fusion { .. }
    )
}

fn launch(
    kernel: &Kernel,
    functions: &[Function],
    inputs: &[Tensor],
    out: &Tensor,
) -> Result<(), String> {
    let launch = LOOP.get().ok_or(
        "the Triton backend's loops launch from Python (lumen.codegen.triton): none is installed",
    )?;
    launch(kernel, functions, inputs, out)
}
