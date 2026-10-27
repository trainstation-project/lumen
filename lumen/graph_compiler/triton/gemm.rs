//! Running GEMM steps (plain dots and dots with fused epilogues) through the Python Triton GEMM kernel.

use std::sync::OnceLock;

use super::Function;
use crate::codegen::triton::{self as codegen, CONFIGS, Config, UNTUNED};
use crate::graph::Primitive;
use crate::graph_compiler::common::autotune::Kernels;
use crate::graph_compiler::plan::Buffer;
use crate::graph_compiler::plan::{Step, Tuned};
use crate::tensor::contiguous_strides;
use crate::{DType, Tensor};

/// Launches the Triton GEMM on `[lhs, rhs, out]` with `[batch, m, k, n]` sizes, the tile config, the
/// generated prologue/epilogue jit functions and the epilogue's extra operands.
pub type Gemm = fn(
    tensors: [&Tensor; 3],
    sizes: [usize; 4],
    accum_dtype: DType,
    config: &Config,
    functions: &[Function],
    epilogue: &Epilogue,
) -> Result<(), String>;

/// Extra operands of a fused epilogue.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Epilogue<'a> {
    /// The fusion's inputs then its secondary outputs, passed to the epilogue in `EXTRA`.
    pub tensors: &'a [Tensor],
    /// Indices of `tensors` passed by value as scalars.
    pub scalars: &'a [usize],
    /// How the kernel shapes its accumulator for the epilogue.
    pub flags: codegen::EpilogueFlags,
}

/// Name of a fusion of a matmul and the normalization row reading its product.
const DOT_ROW: &str = "dot_row";

/// Name of a matmul writing its product and partials of the normalization row reading it.
const DOT_PARTIALS: &str = "dot_partials";

/// The GEMM launcher installed from Python.
static GEMM: OnceLock<Gemm> = OnceLock::new();

/// Installs the GEMM launcher; later calls are ignored.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_gemm(gemm: Gemm) {
    let _ = GEMM.set(gemm);
}

/// A tensor's shape and strides.
fn layout(t: &Tensor) -> (&[usize], &[usize]) {
    (t.shape(), t.strides())
}

/// Runs a dot or GEMM-epilogue fusion `step` with tile `config` (the untuned one when `None`),
/// generating its strided prologues and epilogue; secondary fusion outputs go in `outs`.
fn gemm(
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
    tuned: Option<&Config>,
) -> Result<(), String> {
    let config = tuned.unwrap_or(&CONFIGS[UNTUNED]);
    let launch = GEMM
        .get()
        .ok_or("the Triton backend's GEMM is Python's (lumen.kernels.dot_general.triton): none is installed")?;
    let Primitive::Fusion { name, body, .. } = &step.primitive else {
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
    // A fusion: the dot computes into a contiguous tile and the epilogue writes the fusion's outputs.
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
    // A row epilogue needs whole rows in each program's tile; a partials epilogue tiles of its width.
    let rowed = name == DOT_ROW || name == DOT_PARTIALS;
    let whole;
    let (epilogue, config) = match rowed {
        true if name == DOT_PARTIALS => {
            let width = codegen::partial_tile(cols).expect("a partial tile width");
            whole = Config {
                block_n: width,
                ..*config
            };
            let code = codegen::partials_epilogue(
                body,
                &by_value,
                dot,
                &layouts,
                &outputs,
                [rows, cols],
                width,
            );
            (code, &whole)
        }
        true => {
            whole = match tuned {
                Some(c) => *c,
                None => codegen::untuned_row_gemm(cols),
            };
            let code =
                codegen::row_epilogue(body, &by_value, dot, &layouts, &outputs, [rows, cols]);
            (code, &whole)
        }
        false => {
            let code = codegen::epilogue(body, &by_value, dot, &layouts, &outputs, [rows, cols]);
            (code, config)
        }
    };
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
        flags: match rowed {
            true => codegen::EpilogueFlags {
                group: 1,
                ..codegen::EpilogueFlags::default()
            },
            false => codegen::gemm_flags(body, dot),
        },
    };
    launch(tensors, code.sizes, accum, config, &functions, &fused)
}

/// Accumulator dtype of a `dot_general`.
fn accum_dtype(dot: &Primitive) -> DType {
    let Primitive::DotGeneral { accum_dtype, .. } = dot else {
        unreachable!("a dot_general")
    };
    *accum_dtype
}

/// Whether `step` runs on the GEMM kernel: a dot, or a fusion around one.
pub(crate) fn is_gemm(step: &Step) -> bool {
    match &step.primitive {
        Primitive::DotGeneral { .. } => true,
        p @ Primitive::Fusion { body, .. } => {
            !super::attention::is_attention(p) && codegen::gemm_dot(body).is_some()
        }
        _ => false,
    }
}

/// A tuned GEMM launch with its config baked in.
type Run = Box<dyn Fn(&Step, &[Tensor], &Tensor, &[Tensor]) -> Result<(), String> + Send + Sync>;

/// Runs GEMM `step` with its tuned config, or the untuned default.
pub(crate) fn run(
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
) -> Result<(), String> {
    match step.launch::<Run>() {
        Some(run) => run(step, inputs, out, outs),
        None => gemm(step, inputs, out, outs, None),
    }
}

/// Autotuning candidates for GEMM steps.
pub(crate) struct Gemms;

impl Kernels for Gemms {
    type Config = Config;

    /// Every GEMM config for a GEMM step (whole-row tiles for a row epilogue), none otherwise.
    fn configs(&self, step: &Step) -> Vec<Config> {
        match &step.primitive {
            Primitive::Fusion { name, .. } if name == DOT_ROW => {
                codegen::row_gemm_configs(step.output.1.shape.last().copied().unwrap_or(1))
            }
            // Partials span one tile's columns, so every candidate takes that width.
            Primitive::Fusion { name, .. } if name == DOT_PARTIALS => {
                let n = step.output.1.shape.last().copied().unwrap_or(1);
                let width = codegen::partial_tile(n).expect("a partial tile width");
                let mut configs: Vec<Config> = Vec::new();
                for c in CONFIGS {
                    let c = Config {
                        block_n: width,
                        ..c
                    };
                    if !configs.iter().any(|d| {
                        (d.block_m, d.block_k, d.stages, d.warps)
                            == (c.block_m, c.block_k, c.stages, c.warps)
                    }) {
                        configs.push(c);
                    }
                }
                configs
            }
            _ if is_gemm(step) => CONFIGS.to_vec(),
            _ => Vec::new(),
        }
    }

    /// A launch of `step` with `config`.
    fn build(&mut self, _: &Step, config: Config) -> Tuned {
        let run: Run =
            Box::new(move |step, inputs, out, outs| gemm(step, inputs, out, outs, Some(&config)));
        Tuned::new(format!("{config:?}"), run)
    }
}
