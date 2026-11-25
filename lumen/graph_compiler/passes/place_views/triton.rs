//! Triton's answer to the `place_views` pass: the operand layouts its GEMMs read in place.

use crate::codegen::devices::triton::Triton;
use crate::graph::Primitive;

/// Triton's GEMM reads its operands at any strides (its prologues are generated for their layouts).
impl super::StridedOperands for Triton {
    /// Any layout.
    fn reads_strided(
        _: &Primitive,
        _: [&crate::graph::TensorType; 2],
        _: usize,
        _: &[usize],
    ) -> bool {
        true
    }
}
