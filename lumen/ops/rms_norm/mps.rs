//! rms_norm on MPS (`mps.metal`): a threadgroup a row of the last
//! dimension.

use crate::Tensor;
use crate::graph::plan::Step;
use crate::graph::{Primitive, TensorType};
use crate::ops::mps::{Grid, launch, u64_arg};

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let x = &step.inputs[0].1;
    let weighted = step.inputs.len() == 2;
    encode_rms_norm(
        &step.primitive,
        x,
        weighted,
        None,
        step.label,
        inputs,
        output,
        keep,
    )
}

/// RMS norm `p` of `x` over its last dimension, a threadgroup a row, with
/// a weight if `weighted`: the kernel `fused` if given (an rms_norm
/// fusion's, computing `x` from `inputs`, the weight among them), else
/// `rms_norm_rows[_weighted]`, reading `inputs` (`x`, then the weight);
/// recorded in the profiler as `name`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rms_norm(
    p: &Primitive,
    x: &TensorType,
    weighted: bool,
    fused: Option<&str>,
    name: &'static str,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let Primitive::RmsNorm { epsilon } = p else {
        unreachable!("rms_norm")
    };
    let count = x.shape.last().copied().unwrap_or(1);
    let rows = x.numel().checked_div(count).unwrap_or(0);
    let own = match weighted {
        true => format!("rms_norm_rows_weighted_{}", x.dtype),
        false => format!("rms_norm_rows_{}", x.dtype),
    };
    let kernel = fused.map_or(own, str::to_owned);
    let mut buffers = inputs.to_vec();
    buffers.push(output.cast_const());
    let args = [u64_arg(count), (*epsilon as f32).to_ne_bytes().to_vec()];
    launch(
        &kernel,
        &buffers,
        &args,
        Grid::Groups([rows, 1, 1]),
        keep,
        name,
    )
}
