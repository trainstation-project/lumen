//! What Triton runs for this pass.

use super::LinearReorderingDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::linear_reordering::LinearReordering;
use crate::graph_compiler::pipeline::{Pass, when};

/// Triton's answers to the pass.
impl LinearReorderingDevice for Triton {
    /// With `linear_reordering`, as on MPS: a matmul operand's factor along the kept axes moves to its output.
    fn linear_reordering<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        let reordering = LinearReordering {
            tile: Some(crate::codegen::dot_general::triton::MOST_GEMM_ROW),
        };

        when(config.linear_reordering, reordering)
    }
}
