//! Running GEMM steps (plain dots and dots with fused epilogues) through the Python Triton GEMM kernel.

use std::sync::OnceLock;

use crate::codegen::dot_general::GemmCodegen;
use crate::codegen::dot_general::triton::cross_entropy;
use crate::codegen::dot_general::triton::{self as codegen, CONFIGS, Config, UNTUNED};
use crate::codegen::utils::triton::Triton;
use crate::graph::{Graph, Node, Primitive, intern};
use crate::plan::Buffer;
use crate::plan::StepBase;
use crate::plan::step::triton::{layouts, traced_as};
use crate::runtime::kernel::triton::{Function, Launches};
use crate::tensor::contiguous_strides;
use crate::{DType, Tensor, TensorOptions};

/// Launches the Triton GEMM on `[lhs, rhs, out]` with `[batch, m, k, n]` sizes, the tile config, the
/// generated prologue/epilogue jit functions (their compiled kernels kept in the launches), the
/// epilogue's extra operands and, split-K, which of its two launches.
pub type Gemm = fn(
    tensors: [&Tensor; 3],
    sizes: [usize; 4],
    accum_dtype: DType,
    config: &Config,
    functions: &[Function],
    launches: &Launches,
    epilogue: &Epilogue,
    split: Option<&Split>,
) -> Result<(), String>;

/// One launch of a split-K GEMM (`split_k_kernel.py`): each of `chunks` K chunks' products written to
/// `partials` (`[chunks, batch, m, n]` in the accumulator dtype), or, with a `combine` block (`[rows,
/// columns]`), summed from them in chunk order and run through the epilogue.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Split<'a> {
    pub chunks: usize,
    pub partials: &'a Tensor,
    pub combine: Option<[usize; 2]>,
}

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
pub(crate) const DOT_ROW: &str = "dot_row";

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
    /// What a split's partials and combine are traced as: `dot_general (split-K)`, then the step's label with
    /// its dot as the chunks' sum, `reduce_sum (split-K)`.
    split_labels: [&'static str; 2],
}

/// Generates a GEMM step's prologues and epilogue for the layouts of `inputs`, `out` and `outs`.
fn prepare(step: &StepBase, inputs: &[Tensor], out: &Tensor, outs: &[Tensor]) -> Prepared {
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
            split_labels: split_labels(step.label),
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

                    codegen::row::partials_epilogue(
                        body,
                        &by_value,
                        dot,
                        &layouts,
                        &outputs,
                        [rows, cols],
                        width,
                    )
                }
                _ => codegen::row::row_epilogue(
                    body,
                    &by_value,
                    dot,
                    &layouts,
                    &outputs,
                    [rows, cols],
                ),
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

            Triton::gemm(body, dot, &by_value, &layouts).expect("a Triton GEMM of every GEMM body")
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
        split_labels: split_labels(step.label),
    }
}

/// A split-K GEMM step's launch labels from the step's `label`: `dot_general (split-K)` for the chunks'
/// products, and `label` with its leading `dot_general` read as `reduce_sum (split-K)` for their combine.
fn split_labels(label: &str) -> [&'static str; 2] {
    [
        "dot_general (split-K)",
        intern(label.replacen("dot_general", "reduce_sum (split-K)", 1)),
    ]
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
/// fusion outputs go in `outs`. With `split_k`, a GEMM of few tiles and a long K runs split
/// (`codegen::split_k`): its chunks' products, then their sum through the epilogue.
pub(crate) fn gemm(
    step: &StepBase,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
    tuned: Option<&Config>,
    split_k: bool,
) -> Result<(), String> {
    let launch = GEMM
        .get()
        .ok_or("the Triton backend's GEMM is Python's (lumen.codegen.dot_general.triton.gemm_kernel): none is installed")?;

    if let Primitive::LinearCrossEntropy = step.primitive {
        let config = tuned.copied().unwrap_or(CONFIGS[UNTUNED]);
        return cross_entropy::run(*launch, step, inputs, out, &config);
    }

    let key = layouts(inputs.iter().chain([out]).chain(outs));

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

    let (tensors, extra) = match &step.primitive {
        Primitive::Fusion { body, .. } => {
            let dot = codegen::gemm_dot(body).expect("a GEMM epilogue's dot");
            let (lhs, rhs) = operands(body, dot, inputs);
            ([lhs, rhs, out], inputs.iter().chain(outs).cloned().collect())
        }
        _ => ([&inputs[0], &inputs[1], out], Vec::new()),
    };

    let epilogue = Epilogue {
        tensors: &extra,
        scalars: &prepared.scalars,
        flags: prepared.flags,
    };

    // A row epilogue reads whole rows of the GEMM's tile, so it runs unsplit.
    let row = matches!(&step.primitive, Primitive::Fusion { name, .. } if name == DOT_ROW || name == DOT_PARTIALS);
    let chunks = codegen::split_k(prepared.sizes).filter(|_| split_k && !row);

    let run = |split: Option<&Split>| {
        launch(
            tensors,
            prepared.sizes,
            prepared.accum,
            &config,
            &prepared.functions,
            &prepared.launches,
            &epilogue,
            split,
        )
    };

    let Some(chunks) = chunks else {
        return run(None);
    };

    let [l, m, _, n] = prepared.sizes;
    let options = TensorOptions::new().dtype(prepared.accum).device(out.device());
    let partials = unsafe { Tensor::empty(&[chunks * l * m * n], options) };

    // With row halves, the combine's blocks cover the first half's rows.
    let flags = prepared.flags;
    let rows = match flags.group > 1 && !flags.interleaved && flags.rows {
        true => m / flags.group,
        false => m,
    };

    let block = codegen::combine_block(rows, flags.interleaved && flags.rows);

    for (combine, label) in [None, Some(block)].into_iter().zip(prepared.split_labels) {
        let split = Split {
            chunks,
            partials: &partials,
            combine,
        };

        traced_as(step, label, out, || run(Some(&split)))?;
    }

    Ok(())
}

/// Accumulator dtype of a `dot_general`.
fn accum_dtype(dot: &Primitive) -> DType {
    let Primitive::DotGeneral { accum_dtype, .. } = dot else {
        unreachable!("a dot_general")
    };

    *accum_dtype
}

/// Whether `step` runs on the GEMM kernel: a dot, a fusion around one, or a linear cross entropy.
pub(crate) fn is_gemm(step: &StepBase) -> bool {
    match &step.primitive {
        Primitive::DotGeneral { .. } | Primitive::LinearCrossEntropy => true,
        p @ Primitive::Fusion { body, .. } => {
            !crate::codegen::attention::triton::launch::is_attention(p)
                && codegen::gemm_dot(body).is_some()
        }
        _ => false,
    }
}
