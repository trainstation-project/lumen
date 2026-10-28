//! Running `linear_cross_entropy` steps: the Triton GEMM of the hidden states and the weight with a
//! cross-entropy epilogue writing each output tile's per-row max and sum of exponentials (and the label's
//! logit), then a kernel combining a row's tiles into its loss and `logsumexp`.

use super::gemm::{Epilogue, Gemm};
use super::{Function, Kernel};
use crate::codegen::triton::{self as codegen, Config, LOOP_CONFIGS, UNTUNED_LOOP};
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::tensor::contiguous_strides;
use crate::{DType, Tensor, TensorOptions};

/// A linear cross entropy's prologues and epilogue, generated for its tensors' layouts.
struct Prepared {
    functions: Vec<Function>,
    /// `[batch, m, k, n]`: the rows, the hidden size and the classes.
    sizes: [usize; 4],
}

/// The prologues reading `h` (`[b, d]`) and `w` (`[v, d]`) and the epilogue reducing their product's
/// tiles, its labels `target`.
fn prepare(h: &Tensor, w: &Tensor, target: &Tensor) -> Prepared {
    let dot = Primitive::DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![1],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };
    let (rows, classes) = (h.shape()[0], w.shape()[0]);
    let logits = [rows, classes];
    let code = codegen::gemm(
        &dot,
        (h.shape(), h.strides()),
        (w.shape(), w.strides()),
        (&logits, &contiguous_strides(&logits)),
    );
    let epilogue =
        codegen::cross_entropy_epilogue(rows, classes, (target.shape(), target.strides()));
    Prepared {
        functions: vec![
            Function::new("PROLOGUE_A", code.prologue_a),
            Function::new("PROLOGUE_B", code.prologue_b),
            Function::new("EPILOGUE", epilogue),
        ],
        sizes: code.sizes,
    }
}

/// Runs linear cross entropy `step` on `[h, w, target]` into `out` (`[2, b]`: the losses, then the
/// `logsumexp`s) with GEMM tile `config`: the GEMM writes a partial max and sum a row per tile of
/// `BLOCK_SIZE_N` classes, then one program a row combines them.
pub(super) fn run(
    gemm: Gemm,
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
    config: &Config,
) -> Result<(), String> {
    let [h, w, target] = inputs else {
        unreachable!("linear_cross_entropy reads h, w and the labels")
    };
    let key = super::layouts(inputs.iter().chain([out]));
    let prepared = step.memo.get(key, || Ok(prepare(h, w, target)))?;
    let [_, rows, _, classes] = prepared.sizes;
    if rows == 0 {
        return Ok(());
    }
    let tiles = classes.div_ceil(config.block_n);
    let options = TensorOptions::new().dtype(DType::F32).device(out.device());
    // Each tile's max and sum of exps a row, and each row's label logit.
    let (peaks, sums, picked) = unsafe {
        (
            Tensor::empty(&[rows, tiles], options),
            Tensor::empty(&[rows, tiles], options),
            Tensor::empty(&[rows], options),
        )
    };
    let extra = [sums.clone(), picked.clone(), target.clone()];
    let epilogue = Epilogue {
        tensors: &extra,
        scalars: &[],
        flags: codegen::EpilogueFlags {
            group: 1,
            ..codegen::EpilogueFlags::default()
        },
    };
    gemm(
        [h, w, &peaks],
        prepared.sizes,
        DType::F32,
        config,
        &prepared.functions,
        &epilogue,
    )?;
    let (name, source) = codegen::cross_entropy_combine(tiles, (out.shape(), out.strides()));
    let partials = [peaks, sums, picked];
    let functions: Vec<Function> = partials
        .iter()
        .enumerate()
        .map(|(k, t)| Function::new(format!("LOAD{k}"), codegen::load(t.shape(), t.strides())))
        .collect();
    let loop_config = LOOP_CONFIGS[UNTUNED_LOOP];
    // One program a row: its lanes are the whole block.
    let kernel = Kernel {
        name,
        source,
        by_value: vec![false; 3],
        lanes: loop_config.block_size,
        splits: 1,
        segments: Vec::new(),
    };
    super::launch(&kernel, &loop_config, &functions, &partials, out, rows)
}
