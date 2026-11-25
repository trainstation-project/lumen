//! MPS's answer to the `place_views` pass: the operand layouts its GEMMs read in place.

use crate::codegen::devices::mps::Mps;
use crate::graph::Primitive;

/// Metal GEMMs read the operand layouts `dot_general::mps::reads_strided` accepts in place.
impl super::StridedOperands for Mps {
    /// The MPS GEMM's rule.
    fn reads_strided(
        dot: &Primitive,
        operands: [&crate::graph::TensorType; 2],
        k: usize,
        strides: &[usize],
    ) -> bool {
        crate::codegen::dot_general::mps::reads_strided(dot, operands, k, strides)
    }
}
