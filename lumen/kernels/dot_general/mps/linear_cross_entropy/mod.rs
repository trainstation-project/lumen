//! Fused `h @ w^T` plus cross entropy on Metal, without materializing the logits.

use super::float_tiles;
use crate::Tensor;
use crate::graph::TensorType;
use crate::graph_compiler::plan::Step;
use crate::kernels::mps::{Grid, dims_arg, launch};
use crate::tensor::contiguous_strides;

/// Bytes of scratch for the per-row, per-column-tile maxima and exp-sums.
pub(crate) fn linear_cross_entropy_scratch(h: &TensorType, w: &TensorType) -> usize {
    let (b, v) = (h.shape[0], w.shape[0]);
    2 * b * v.div_ceil(float_tiles(1, b, v).2.1) * 4
}

/// Launches the fused matmul writing per-tile softmax partials and target logits, then the
/// combine that turns them into each row's loss (`out[..b]`) and log-sum-exp (`out[b..]`).
pub(crate) fn encode_linear_cross_entropy(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (h, w, target) = (&step.inputs[0].1, &step.inputs[1].1, &step.inputs[2].1);
    let (b, d, v) = (h.shape[0], h.shape[1], w.shape[0]);
    if b == 0 || v == 0 {
        return Ok(());
    }
    if scratch.is_null() {
        return Err(format!(
            "{}: needs scratch for its partials: compile the graph for MPS (lumen.compile, Plan(graph, \"mps\"))",
            step.label
        ));
    }
    let (small, mid, (tm, tn)) = float_tiles(1, b, v);
    let kind = match (small, mid) {
        (true, _) => "small_",
        (_, true) => "mid_",
        _ => "",
    };
    let tiles = v.div_ceil(tn);
    // Scratch holds `[b, tiles]` maxima followed by `[b, tiles]` sums.
    let (maxima, sums) = (
        scratch.cast_const(),
        unsafe { scratch.add(b * tiles * 4) }.cast_const(),
    );
    // h is [b, d] row-major; w is [v, d], read as a [d, v] rhs with K stride 1 and N stride d.
    let p = dims_arg([b, v, d, 0, d, 1, 0, 1, d]);
    let partials = TensorType::new(crate::DType::F32, &[2, b, tiles]);
    crate::profiler::next_launch("linear_cross_entropy (partials)", || {
        crate::profiler::IoTypes {
            inputs: step.inputs.iter().map(|(_, ty)| ty.clone()).collect(),
            outputs: vec![partials.clone()],
            input_strides: step.input_strides(),
            output_strides: vec![contiguous_strides(&partials.shape)],
        }
    });
    launch(
        &format!("lce_partials_{kind}{}_{}", h.dtype, target.dtype),
        &[
            inputs[0],
            inputs[1],
            inputs[2],
            output.cast_const(),
            maxima,
            sums,
        ],
        &[p],
        Grid::Groups([tiles, b.div_ceil(tm), 1]),
        keep.clone(),
        step.label,
    )?;
    crate::profiler::next_launch("linear_cross_entropy (combine)", || {
        crate::profiler::IoTypes {
            inputs: vec![partials.clone()],
            outputs: vec![step.output.1.clone()],
            input_strides: vec![contiguous_strides(&partials.shape)],
            output_strides: step.output_strides(),
        }
    });
    launch(
        "lce_combine",
        &[maxima, sums, output.cast_const()],
        &[dims_arg([b, tiles])],
        Grid::Groups([b.div_ceil(8), 1, 1]),
        keep,
        step.label,
    )
}
