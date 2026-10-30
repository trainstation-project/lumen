//! Running GEMM steps (plain dots and dots with fused epilogues) through the Python Triton GEMM kernel.

use std::sync::OnceLock;

use super::{Function, Launches};
use crate::codegen::gemm::GemmCodegen;
use crate::codegen::triton::{self as codegen, CONFIGS, Config, UNTUNED};
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::common::autotune::Kernels;
use crate::graph_compiler::plan::Buffer;
use crate::graph_compiler::plan::{Step, Tuned};
use crate::tensor::contiguous_strides;
use crate::{DType, Tensor};

/// Launches the Triton GEMM on `[lhs, rhs, out]` with `[batch, m, k, n]` sizes, the tile config, the
/// generated prologue/epilogue jit functions (their compiled kernels kept in the launches) and the
/// epilogue's extra operands.
pub type Gemm = fn(
    tensors: [&Tensor; 3],
    sizes: [usize; 4],
    accum_dtype: DType,
    config: &Config,
    functions: &[Function],
    launches: &Launches,
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

/// What a GEMM step's launch needs beyond its tensors and tile, generated from their layouts.
struct Prepared {
    /// The prologues and epilogue.
    functions: Vec<Function>,
    /// `[batch, m, k, n]`.
    sizes: [usize; 4],
    flags: codegen::EpilogueFlags,
    /// Fusion inputs passed by value.
    scalars: Vec<usize>,
    accum: DType,
    launches: Launches,
}

/// Generates a GEMM step's prologues and epilogue for the layouts of `inputs`, `out` and `outs`.
fn prepare(step: &Step, inputs: &[Tensor], out: &Tensor, outs: &[Tensor]) -> Prepared {
    let Primitive::Fusion { name, body, .. } = &step.primitive else {
        let code = codegen::gemm(
            &step.primitive,
            layout(&inputs[0]),
            layout(&inputs[1]),
            layout(out),
        );

        return Prepared {
            functions: vec![
                Function::new("PROLOGUE_A", code.prologue_a),
                Function::new("PROLOGUE_B", code.prologue_b),
                Function::new("EPILOGUE", code.epilogue),
            ],
            sizes: code.sizes,
            flags: codegen::EpilogueFlags {
                group: 1,
                ..codegen::EpilogueFlags::default()
            },
            scalars: Vec::new(),
            accum: accum_dtype(&step.primitive),
            launches: Launches::default(),
        };
    };

    // A fusion: the dot computes into a contiguous tile and the epilogue writes the fusion's outputs.
    let dot = codegen::gemm_dot(body).expect("a GEMM epilogue's dot");
    let (lhs, rhs) = operands(body, dot, inputs);

    let by_value: Vec<bool> = step
        .inputs
        .iter()
        .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
        .collect();

    let code = match name.as_str() {
        // A GEMM whose rows a normalization reads: the row kernel's epilogues.
        DOT_PARTIALS | DOT_ROW => {
            let product = body.type_of(dot.output);
            let contiguous = contiguous_strides(&product.shape);

            let mut code = codegen::gemm(
                &dot.primitive,
                layout(lhs),
                layout(rhs),
                (&product.shape, &contiguous),
            );

            let layouts: Vec<_> = inputs.iter().map(layout).collect();
            let [_, rows, _, cols] = code.sizes;
            let outputs: Vec<_> = std::iter::once(out).chain(outs).map(layout).collect();

            code.epilogue = match name.as_str() {
                DOT_PARTIALS => {
                    let width = codegen::partial_tile(cols).expect("a partial tile width");

                    codegen::partials_epilogue(
                        body,
                        &by_value,
                        dot,
                        &layouts,
                        &outputs,
                        [rows, cols],
                        width,
                    )
                }
                _ => codegen::row_epilogue(body, &by_value, dot, &layouts, &outputs, [rows, cols]),
            };

            code
        }
        _ => {
            let owned = |t: &Tensor| (t.shape().to_vec(), t.strides().to_vec());

            let layouts = codegen::Layouts {
                lhs: owned(lhs),
                rhs: owned(rhs),
                inputs: inputs.iter().map(owned).collect(),
                outputs: std::iter::once(out).chain(outs).map(owned).collect(),
            };

            codegen::Triton::gemm(body, dot, &by_value, &layouts)
                .expect("a Triton GEMM of every GEMM body")
        }
    };

    let flags = match name == DOT_ROW || name == DOT_PARTIALS {
        true => codegen::EpilogueFlags {
            group: 1,
            ..codegen::EpilogueFlags::default()
        },
        false => codegen::gemm_flags(body, dot),
    };

    Prepared {
        functions: vec![
            Function::new("PROLOGUE_A", code.prologue_a),
            Function::new("PROLOGUE_B", code.prologue_b),
            Function::new("EPILOGUE", code.epilogue),
        ],
        sizes: code.sizes,
        flags,
        scalars: (0..inputs.len()).filter(|&k| by_value[k]).collect(),
        accum: accum_dtype(&dot.primitive),
        launches: Launches::default(),
    }
}

/// The dot's lhs and rhs among a GEMM fusion's `inputs`.
fn operands<'a>(body: &Graph, dot: &Node, inputs: &'a [Tensor]) -> (&'a Tensor, &'a Tensor) {
    let operand = |k: usize| {
        let at = body.inputs().iter().position(|&v| v == dot.inputs[k]);
        &inputs[at.expect("the dot's operands are the fusion's inputs")]
    };

    (operand(0), operand(1))
}

/// Runs a dot, GEMM-epilogue fusion or linear cross entropy `step` with tile `config` (the untuned one
/// when `None`), its strided prologues and epilogue generated once per layout of its tensors; secondary
/// fusion outputs go in `outs`.
fn gemm(
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
    tuned: Option<&Config>,
) -> Result<(), String> {
    let launch = GEMM
        .get()
        .ok_or("the Triton backend's GEMM is Python's (lumen.kernels.dot_general.triton): none is installed")?;

    if let Primitive::LinearCrossEntropy = step.primitive {
        let config = tuned.copied().unwrap_or(CONFIGS[UNTUNED]);
        return super::cross_entropy::run(*launch, step, inputs, out, &config);
    }

    let key = super::layouts(inputs.iter().chain([out]).chain(outs));

    let prepared = step
        .memo
        .get(key, || Ok(prepare(step, inputs, out, outs)))?;

    let cols = prepared.sizes[3];

    // A row epilogue needs whole rows in each program's tile.
    let config = match (&step.primitive, tuned) {
        (_, Some(c)) => *c,
        (Primitive::Fusion { name, .. }, None) if name == DOT_ROW => {
            codegen::untuned_row_gemm(cols)
        }
        _ => CONFIGS[UNTUNED],
    };

    let Primitive::Fusion { body, .. } = &step.primitive else {
        let tensors = [&inputs[0], &inputs[1], out];

        let plain = Epilogue {
            tensors: &[],
            scalars: &[],
            flags: prepared.flags,
        };

        return launch(
            tensors,
            prepared.sizes,
            prepared.accum,
            &config,
            &prepared.functions,
            &prepared.launches,
            &plain,
        );
    };

    let dot = codegen::gemm_dot(body).expect("a GEMM epilogue's dot");
    let (lhs, rhs) = operands(body, dot, inputs);
    let extra: Vec<Tensor> = inputs.iter().chain(outs).cloned().collect();

    let fused = Epilogue {
        tensors: &extra,
        scalars: &prepared.scalars,
        flags: prepared.flags,
    };

    launch(
        [lhs, rhs, out],
        prepared.sizes,
        prepared.accum,
        &config,
        &prepared.functions,
        &prepared.launches,
        &fused,
    )
}

/// Accumulator dtype of a `dot_general`.
fn accum_dtype(dot: &Primitive) -> DType {
    let Primitive::DotGeneral { accum_dtype, .. } = dot else {
        unreachable!("a dot_general")
    };

    *accum_dtype
}

/// Whether `step` runs on the GEMM kernel: a dot, a fusion around one, or a linear cross entropy.
pub(crate) fn is_gemm(step: &Step) -> bool {
    match &step.primitive {
        Primitive::DotGeneral { .. } | Primitive::LinearCrossEntropy => true,
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
