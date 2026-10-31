//! Running `linear_cross_entropy` steps: the Triton GEMM of the hidden states and the weight with a
//! cross-entropy epilogue writing each output tile's per-row max and sum of exponentials (and the label's
//! logit), then a kernel combining a row's tiles into its loss and `logsumexp`.

use super::gemm::{Epilogue, Gemm};
use super::{Function, Kernel, Launches, Template};
use crate::codegen::triton::{self as codegen, Config, LOOP_CONFIGS, UNTUNED_LOOP};
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::tensor::contiguous_strides;
use crate::{DType, Device, Tensor, TensorOptions};

/// The fixed GEMM epilogue (lumen.codegen.dot_general.triton.linear_cross_entropy), bound by the source
/// importing it.
const EPILOGUE: &str = "cross_entropy_epilogue";

/// A linear cross entropy's prologues for its tensors' layouts and its fixed epilogue.
struct Prepared {
    functions: Vec<Function>,
    /// The kernels compiled for the GEMM and the combine.
    launches: Launches,
    /// `[batch, m, k, n]`: the rows, the hidden size and the classes.
    sizes: [usize; 4],
    /// The epilogue's by-value scalars: the rows, the classes and the labels' stride.
    scalars: [Tensor; 3],
}

/// The prologues reading `h` (`[b, d]`) and `w` (`[v, d]`), the epilogue reducing their product's tiles,
/// and the scalars it reads its labels `target` with.
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

    let import =
        format!("from lumen.codegen.dot_general.triton.linear_cross_entropy import {EPILOGUE}\n");
    let scalar =
        |v: usize| Tensor::full(&[], v as i64, TensorOptions::new().device(Device::Cpu(0)));

    Prepared {
        functions: vec![
            Function::new("PROLOGUE_A", code.prologue_a),
            Function::new("PROLOGUE_B", code.prologue_b),
            Function::new("EPILOGUE", (EPILOGUE.into(), import)),
        ],
        sizes: code.sizes,
        launches: Launches::default(),
        scalars: [scalar(rows), scalar(classes), scalar(target.strides()[0])],
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

    let [rows_at, classes_at, stride_at] = &prepared.scalars;

    let extra = [
        sums.clone(),
        picked.clone(),
        target.clone(),
        rows_at.clone(),
        classes_at.clone(),
        stride_at.clone(),
    ];

    let epilogue = Epilogue {
        tensors: &extra,
        scalars: &[3, 4, 5],
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
        &prepared.launches,
        &epilogue,
    )?;

    if !out.is_contiguous() {
        return Err(format!(
            "{}: the combine writes a contiguous [2, b] result, got strides {:?}",
            step.label,
            out.strides()
        ));
    }

    let partials = [peaks, sums, picked];
    let loop_config = LOOP_CONFIGS[UNTUNED_LOOP];

    // One program a row, its lanes the whole block: the fixed combine over contiguous partials and `out`.
    let kernel = Kernel {
        name: "cross_entropy_combine".into(),
        source: String::new(),
        by_value: vec![false; 3],
        lanes: loop_config.block_size,
        splits: 1,
        segments: Vec::new(),
        template: Some(Template::CrossEntropyCombine { tiles }),
    };

    super::launch(
        &kernel,
        &loop_config,
        &[],
        &prepared.launches,
        &partials,
        out,
        rows,
    )
}
