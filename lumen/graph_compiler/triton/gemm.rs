use std::sync::OnceLock;

use super::Function;
use crate::codegen::triton::{self as codegen, CONFIGS, Config, UNTUNED};
use crate::graph::Primitive;
use crate::graph_compiler::common::autotune::Kernels;
use crate::graph_compiler::plan::Buffer;
use crate::graph_compiler::plan::{Step, Tuned};
use crate::tensor::contiguous_strides;
use crate::{DType, Tensor};

pub type Gemm = fn(
    tensors: [&Tensor; 3],
    sizes: [usize; 4],
    accum_dtype: DType,
    config: &Config,
    functions: &[Function],
    epilogue: &Epilogue,
) -> Result<(), String>;

#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Epilogue<'a> {
    pub tensors: &'a [Tensor],
    pub scalars: &'a [usize],
    pub flags: codegen::EpilogueFlags,
}

static GEMM: OnceLock<Gemm> = OnceLock::new();

#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_gemm(gemm: Gemm) {
    let _ = GEMM.set(gemm);
}

fn layout(t: &Tensor) -> (&[usize], &[usize]) {
    (t.shape(), t.strides())
}

fn gemm(
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
    config: &Config,
) -> Result<(), String> {
    let launch = GEMM
        .get()
        .ok_or("the Triton backend's GEMM is Python's (lumen.kernels.dot_general.triton): none is installed")?;
    let Primitive::Fusion { body, .. } = &step.primitive else {
        let code = codegen::gemm(
            &step.primitive,
            layout(&inputs[0]),
            layout(&inputs[1]),
            layout(out),
        );
        let functions = [
            Function::new("PROLOGUE_A", code.prologue_a),
            Function::new("PROLOGUE_B", code.prologue_b),
            Function::new("EPILOGUE", code.epilogue),
        ];
        let accum = accum_dtype(&step.primitive);
        let tensors = [&inputs[0], &inputs[1], out];
        let plain = Epilogue {
            tensors: &[],
            scalars: &[],
            flags: codegen::EpilogueFlags {
                group: 1,
                ..codegen::EpilogueFlags::default()
            },
        };
        return launch(tensors, code.sizes, accum, config, &functions, &plain);
    };
    let dot = codegen::gemm_dot(body).expect("a GEMM epilogue's dot");
    let operand = |k: usize| {
        let at = body.inputs().iter().position(|&v| v == dot.inputs[k]);
        &inputs[at.expect("the dot's operands are the fusion's inputs")]
    };
    let (lhs, rhs) = (operand(0), operand(1));
    let product = body.type_of(dot.output);
    let contiguous = contiguous_strides(&product.shape);
    let code = codegen::gemm(
        &dot.primitive,
        layout(lhs),
        layout(rhs),
        (&product.shape, &contiguous),
    );
    let by_value: Vec<bool> = step
        .inputs
        .iter()
        .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
        .collect();
    let layouts: Vec<_> = inputs.iter().map(layout).collect();
    let [_, rows, _, cols] = code.sizes;
    let outputs: Vec<_> = std::iter::once(out).chain(outs).map(layout).collect();
    let epilogue = codegen::epilogue(body, &by_value, dot, &layouts, &outputs, [rows, cols]);
    let functions = [
        Function::new("PROLOGUE_A", code.prologue_a),
        Function::new("PROLOGUE_B", code.prologue_b),
        Function::new("EPILOGUE", epilogue),
    ];
    let scalars: Vec<usize> = (0..inputs.len()).filter(|&k| by_value[k]).collect();
    let accum = accum_dtype(&dot.primitive);
    let tensors = [lhs, rhs, out];
    let extra: Vec<Tensor> = inputs.iter().chain(outs).cloned().collect();
    let fused = Epilogue {
        tensors: &extra,
        scalars: &scalars,
        flags: codegen::gemm_flags(body, dot),
    };
    launch(tensors, code.sizes, accum, config, &functions, &fused)
}

fn accum_dtype(dot: &Primitive) -> DType {
    let Primitive::DotGeneral { accum_dtype, .. } = dot else {
        unreachable!("a dot_general")
    };
    *accum_dtype
}

pub(crate) fn is_gemm(step: &Step) -> bool {
    match &step.primitive {
        Primitive::DotGeneral { .. } => true,
        Primitive::Fusion { body, .. } => codegen::gemm_dot(body).is_some(),
        _ => false,
    }
}

type Run = Box<dyn Fn(&Step, &[Tensor], &Tensor, &[Tensor]) -> Result<(), String> + Send + Sync>;

pub(crate) fn run(
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
) -> Result<(), String> {
    match step.launch::<Run>() {
        Some(run) => run(step, inputs, out, outs),
        None => gemm(step, inputs, out, outs, &CONFIGS[UNTUNED]),
    }
}

pub(crate) struct Gemms;

impl Kernels for Gemms {
    type Config = Config;

    fn configs(&self, step: &Step) -> Vec<Config> {
        match is_gemm(step) {
            true => CONFIGS.to_vec(),
            false => Vec::new(),
        }
    }

    fn build(&mut self, _: &Step, config: Config) -> Tuned {
        let run: Run =
            Box::new(move |step, inputs, out, outs| gemm(step, inputs, out, outs, &config));
        Tuned::new(format!("{config:?}"), run)
    }
}
