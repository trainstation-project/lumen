//! softmax on MPS (`mps.metal`): a threadgroup a row of the last
//! dimension, online softmax.

use crate::Tensor;
use crate::graph::Primitive::Softmax;
use crate::graph::TensorType;
use crate::graph::plan::Step;
use crate::ops::mps::{Grid, launch, u64_arg};

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let x = &step.inputs[0].1;
    if step.primitive
        != (Softmax {
            axis: x.shape.len() - 1,
        })
    {
        return Err(format!(
            "softmax on MPS takes the last dimension, got {}",
            step.primitive
        ));
    }
    encode_softmax(x, None, step.label, inputs, output, keep)
}

/// Softmax of `x` over its last dimension, a threadgroup a row: the kernel
/// `fused` if given (a softmax fusion's, computing its input from
/// `inputs`), else `softmax_rows`, reading `inputs[0]`; recorded in the
/// profiler as `name`.
pub(crate) fn encode_softmax(
    x: &TensorType,
    fused: Option<&str>,
    name: &'static str,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let count = x.shape.last().copied().unwrap_or(1);
    let rows = x.numel().checked_div(count).unwrap_or(0);
    let kernel = fused.map_or_else(|| format!("softmax_rows_{}", x.dtype), str::to_owned);
    let mut buffers = inputs.to_vec();
    buffers.push(output.cast_const());
    launch(
        &kernel,
        &buffers,
        &[u64_arg(count)],
        Grid::Groups([rows, 1, 1]),
        keep,
        name,
    )
}
