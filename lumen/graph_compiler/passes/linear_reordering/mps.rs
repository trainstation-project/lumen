//! What MPS runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::linear_reordering::LinearReordering;
use crate::graph_compiler::pipeline::{Pass, when};

/// With `linear_reordering`: a matmul operand's factor along the kept axes (a norm's 1 / rms, a weight's
/// per-channel scale) moves to its output, for its epilogue.
impl DevicePass<Mps> for LinearReordering {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Mps>> {
        let reordering = LinearReordering {
            tile: Some(crate::codegen::dot_general::mps::ROW_TILE),
        };

        when(config.linear_reordering, reordering)
    }
}
