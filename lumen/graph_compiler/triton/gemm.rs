use std::sync::OnceLock;

use super::Function;
use crate::graph::Primitive;
use crate::graph_compiler::common::autotune::{Autotuner, picked_for};
use crate::graph_compiler::plan::Step;
use crate::stream::event::event_for;
use crate::{DType, Device, Tensor, TensorOptions};

#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Config {
    pub block_m: usize,
    pub block_n: usize,
    pub block_k: usize,
    pub group_m: usize,
    pub stages: usize,
    pub warps: usize,
}

const fn config(
    block_m: usize,
    block_n: usize,
    block_k: usize,
    stages: usize,
    warps: usize,
) -> Config {
    Config {
        block_m,
        block_n,
        block_k,
        group_m: 8,
        stages,
        warps,
    }
}

pub(crate) const CONFIGS: [Config; 7] = [
    config(128, 256, 64, 3, 8),
    config(64, 256, 32, 4, 4),
    config(128, 128, 32, 4, 4),
    config(128, 64, 32, 4, 4),
    config(64, 128, 32, 4, 4),
    config(128, 32, 32, 4, 4),
    config(64, 32, 32, 5, 2),
];

const UNTUNED: usize = 2;
const TUNER: &str = "triton";
const REPS: usize = 10;

pub type Gemm = fn(
    tensors: [&Tensor; 3],
    sizes: [usize; 4],
    accum_dtype: DType,
    config: &Config,
    functions: &[Function],
) -> Result<(), String>;

static GEMM: OnceLock<Gemm> = OnceLock::new();

#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_gemm(gemm: Gemm) {
    let _ = GEMM.set(gemm);
}

fn gemm(
    lhs: &Tensor,
    rhs: &Tensor,
    out: &Tensor,
    dot: &Primitive,
    config: &Config,
) -> Result<(), String> {
    let launch = GEMM
        .get()
        .ok_or("the Triton backend's GEMM is Python's (lumen.kernels.dot_general.triton): none is installed")?;
    let Primitive::DotGeneral { accum_dtype, .. } = dot else {
        unreachable!("a dot_general")
    };
    let code = crate::codegen::triton::gemm(
        dot,
        (lhs.shape(), lhs.strides()),
        (rhs.shape(), rhs.strides()),
        (out.shape(), out.strides()),
    );
    let functions = [
        Function::new("PROLOGUE_A", code.prologue_a),
        Function::new("PROLOGUE_B", code.prologue_b),
        Function::new("EPILOGUE", code.epilogue),
    ];
    launch(
        [lhs, rhs, out],
        code.sizes,
        *accum_dtype,
        config,
        &functions,
    )
}

fn key(step: &Step) -> Option<String> {
    let Primitive::DotGeneral { .. } = step.primitive else {
        return None;
    };
    let types: Vec<String> = step.inputs.iter().map(|(_, ty)| ty.to_string()).collect();
    Some(format!(
        "{} {} -> {}",
        types.join(" "),
        step.primitive,
        step.output.1
    ))
}

pub(crate) fn run(step: &Step, lhs: &Tensor, rhs: &Tensor, out: &Tensor) -> Result<(), String> {
    let picked = key(step).and_then(|key| picked_for(TUNER, &key));
    gemm(
        lhs,
        rhs,
        out,
        &step.primitive,
        &CONFIGS[picked.unwrap_or(UNTUNED)],
    )
}

pub(crate) struct TritonAutotuner {
    pub device: Device,
}

impl Autotuner for TritonAutotuner {
    type Config = usize;

    fn device(&self) -> &'static str {
        TUNER
    }

    fn key(&self, step: &Step) -> Option<String> {
        key(step)
    }

    fn candidates(&self, _: &Step) -> Vec<usize> {
        (0..CONFIGS.len()).collect()
    }

    fn generate(&mut self, _: &Step, config: usize) -> String {
        config.to_string()
    }

    fn compile(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn time(&self, step: &Step, kernels: &[String]) -> Result<Vec<f64>, String> {
        let zeros = |ty: &crate::graph::TensorType| {
            Tensor::zeros(
                &ty.shape,
                TensorOptions::new().dtype(ty.dtype).device(self.device),
            )
        };
        let (lhs, rhs, out) = (
            zeros(&step.inputs[0].1),
            zeros(&step.inputs[1].1),
            zeros(&step.output.1),
        );
        kernels
            .iter()
            .map(|kernel| {
                let config = &CONFIGS[kernel.parse::<usize>().expect("a config's index")];
                gemm(&lhs, &rhs, &out, &step.primitive, config)?;
                let mut start = event_for(self.device, true)?;
                let mut end = event_for(self.device, true)?;
                start.record()?;
                for _ in 0..REPS {
                    gemm(&lhs, &rhs, &out, &step.primitive, config)?;
                }
                end.record()?;
                end.synchronize()?;
                Ok(start.elapsed_time(end.as_ref())? / 1e3 / REPS as f64)
            })
            .collect()
    }

    fn apply(&self, _: &mut Step, _: &str, _: usize) {}
}
