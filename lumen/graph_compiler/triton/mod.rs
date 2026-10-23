mod fusion;
mod gemm;

use std::sync::OnceLock;

use fusion::LoopFusion;
#[cfg(feature = "python")]
pub use gemm::Config;
#[cfg(feature = "python")]
pub(crate) use gemm::set_gemm;

use super::common::autotune::autotune;
use super::common::simplify::Simplify;
use super::pass::{GraphModule, Pipeline};
use super::plan::{Buffer, Step};
use super::{Options, Plan, cpu};
use crate::graph::{Graph, Primitive};
use crate::{DType, Device, Scalar, Tensor};

#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Kernel {
    pub name: String,
    pub source: String,
    pub by_value: Vec<bool>,
}

pub type Loop = fn(kernel: &Kernel, inputs: &[Tensor], out: &Tensor) -> Result<(), String>;

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
            let (name, source) = crate::codegen::triton::kernel(body, &by_value, &[]);
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
        autotune(&mut gemm::TritonAutotuner { device }, &mut plan)?;
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
        (Primitive::Fusion { body, .. }, Some(kernel)) => {
            let strides: Vec<Option<Vec<usize>>> = inputs
                .iter()
                .map(|t| (!t.is_contiguous()).then(|| t.strides().to_vec()))
                .collect();
            if strides.iter().all(Option::is_none) {
                return launch(kernel, inputs, out);
            }
            let (name, source) = crate::codegen::triton::kernel(body, &kernel.by_value, &strides);
            let strided = Kernel {
                name,
                source,
                by_value: kernel.by_value.clone(),
            };
            launch(&strided, inputs, out)
        }
        _ => Err(format!("{}: has no Triton kernel on CUDA yet", step.label)),
    }
}

pub(crate) fn fill(t: &Tensor, value: Scalar) -> Result<(), String> {
    let (name, source) = crate::codegen::triton::fill(t.dtype(), t.shape(), t.strides());
    let bits = crate::codegen::triton::bits(t.dtype(), value);
    let kernel = Kernel {
        name,
        source,
        by_value: vec![true],
    };
    launch(&kernel, &[Tensor::from_slice(&[bits], DType::U64)], t)
}

pub(crate) fn copy(dst: &Tensor, src: &Tensor) -> Result<(), String> {
    let (name, source) = crate::codegen::triton::copy(dst.shape(), src.strides(), dst.strides());
    let kernel = Kernel {
        name,
        source,
        by_value: vec![false],
    };
    launch(&kernel, std::slice::from_ref(src), dst)
}

pub(crate) fn reads_strided(step: &Step, k: usize, strides: &[usize]) -> bool {
    match &step.primitive {
        Primitive::DotGeneral { .. } => gemm::reads_strided(step, k, strides),
        Primitive::Fusion { .. } => true,
        _ => false,
    }
}

fn launch(kernel: &Kernel, inputs: &[Tensor], out: &Tensor) -> Result<(), String> {
    let launch = LOOP.get().ok_or(
        "the Triton backend's loops launch from Python (lumen.codegen.triton): none is installed",
    )?;
    launch(kernel, inputs, out)
}
