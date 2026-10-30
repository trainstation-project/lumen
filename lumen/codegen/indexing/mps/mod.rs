//! Metal indexing: strided gathers and scatter-add kernels.

pub(crate) mod gather;
pub(crate) mod scatter_add;

use super::IndexingCodegen;
use crate::codegen::mps::Mps;
use crate::graph::TensorType;

/// Metal indexing: scatter-adds run Metal's fixed kernel.
impl IndexingCodegen for Mps {
    /// The kernel's name and source.
    type Code = (String, String);

    /// None: a scatter-add runs the fixed `scatter_add` kernel.
    fn scatter_add(operand: &TensorType, _: usize, _: usize) -> Result<(String, String), String> {
        Err(format!(
            "a scatter_add into {operand} runs Metal's fixed kernel"
        ))
    }
}
