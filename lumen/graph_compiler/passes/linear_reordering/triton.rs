//! What Triton runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::linear_reordering::LinearReordering;
use crate::graph_compiler::pipeline::{Pass, when};

/// With `linear_reordering`, as on MPS: a matmul operand's factor along the kept axes moves to its output.
impl DevicePass<Triton> for LinearReordering {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
        let reordering = LinearReordering {
            tile: Some(crate::codegen::dot_general::triton::MOST_GEMM_ROW),
        };

        when(config.linear_reordering, reordering)
    }
}
