//! Metal layout: the generator of transposes fused with elementwise ops, staged through tiles, and the
//! reshape steps the plan copies outputs with. Other lone layout steps run as one-node loops.

pub(crate) mod transpose;

use crate::Tensor;
use crate::graph_compiler::plan::Step;

/// Encodes a reshape step, the copy the plan writes an output aliasing another buffer with: the input's
/// bytes copied whole, a reshape keeping their order.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    crate::codegen::dynamic_slice::mps::copy(inputs[0], output, &step.output.1, keep, step.label)
}
