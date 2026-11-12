//! What MPS runs for this pass.

use crate::codegen::dot_general::mps as gemm;
use crate::graph::{Primitive, TensorType};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::dot::canonicalize::CanonicalizeDots;
use crate::graph_compiler::pipeline::Pass;
use crate::tensor::contiguous_strides;

/// Whether the GEMM kernels read operand `k` of `dot` as laid out (contiguously): each of its batch, free and
/// contracting dimension groups merges into one stride.
pub(crate) fn reads_contiguous(dot: &Primitive, operands: [&TensorType; 2], k: usize) -> bool {
    gemm::reads_strided(dot, operands, k, &contiguous_strides(&operands[k].shape))
}

/// Always: matmul operands the GEMM kernels can't read as three strides are transposed first.
impl DevicePass<Mps> for CanonicalizeDots {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Mps>> {
        Box::new(CanonicalizeDots {
            readable: reads_contiguous,
        })
    }
}
