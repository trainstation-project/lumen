use std::sync::OnceLock;

use super::Function;
use crate::codegen::triton::{CONFIGS, Config, UNTUNED};
use crate::graph::Primitive;
use crate::graph_compiler::common::autotune::{Kernels, Timer};
use crate::graph_compiler::plan::{Step, Tuned};
use crate::stream::event::event_for;
use crate::{DType, Device, Tensor, TensorOptions};


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

type Run = Box<dyn Fn(&Step, &Tensor, &Tensor, &Tensor) -> Result<(), String> + Send + Sync>;

pub(crate) fn run(step: &Step, lhs: &Tensor, rhs: &Tensor, out: &Tensor) -> Result<(), String> {
    match step.launch::<Run>() {
        Some(run) => run(step, lhs, rhs, out),
        None => gemm(lhs, rhs, out, &step.primitive, &CONFIGS[UNTUNED]),
    }
}

pub(crate) struct Gemms;

impl Kernels for Gemms {
    type Config = Config;

    fn configs(&self, step: &Step) -> Vec<Config> {
        match step.primitive {
            Primitive::DotGeneral { .. } => CONFIGS.to_vec(),
            _ => Vec::new(),
        }
    }

    fn build(&mut self, _: &Step, config: Config) -> Tuned {
        let run: Run = Box::new(move |step, lhs, rhs, out| gemm(lhs, rhs, out, &step.primitive, &config));
        Tuned::new(format!("{config:?}"), run)
    }
}

pub(crate) struct Events {
    pub device: Device,
}

impl Timer for Events {
    type Buffers = [Tensor; 3];

    fn buffers(&self, step: &Step) -> Result<Option<[Tensor; 3]>, String> {
        let zeros = |ty: &crate::graph::TensorType| {
            Tensor::zeros(
                &ty.shape,
                TensorOptions::new().dtype(ty.dtype).device(self.device),
            )
        };
        Ok(Some([
            zeros(&step.inputs[0].1),
            zeros(&step.inputs[1].1),
            zeros(&step.output.1),
        ]))
    }

    fn time(&self, [lhs, rhs, out]: &[Tensor; 3], step: &Step, reps: usize) -> Result<f64, String> {
        let mut start = event_for(self.device, true)?;
        let mut end = event_for(self.device, true)?;
        start.record()?;
        for _ in 0..reps {
            run(step, lhs, rhs, out)?;
        }
        end.record()?;
        end.synchronize()?;
        Ok(start.elapsed_time(end.as_ref())? / 1e3 / reps as f64)
    }
}
