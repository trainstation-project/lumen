//! TPU's answer to the `place_views` pass: the operand layouts its GEMMs read in place.

use crate::codegen::devices::tpu::Tpu;
use crate::codegen::dot_general::tpu::slice_view;

/// The TPU GEMM reads slices `slice_view` takes, in a step of its own or a `dot_general` fusion.
impl super::StridedOperands for Tpu {
    /// None: it reads a slice in place only knowing where the slice starts (`reads_slice`).
    fn reads_strided(
        _: &crate::graph::Primitive,
        _: [&crate::graph::TensorType; 2],
        _: usize,
        _: &[usize],
    ) -> bool {
        false
    }

    /// Whether `kernel` is the GEMM's (`dot_general`) and `slice_view` takes the slice.
    fn reads_slice(
        kernel: &str,
        _: &crate::graph::Primitive,
        operands: [&crate::graph::TensorType; 2],
        k: usize,
        strides: &[usize],
        starts: &[usize],
    ) -> bool {
        kernel == "dot_general" && slice_view(operands[k], strides, starts).is_some()
    }
}
