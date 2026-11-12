//! What MPS runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::linear_cross_entropy::LinearCrossEntropy;
use crate::graph_compiler::pipeline::{Pass, when};

/// With `fuse` and `fused_linear_cross_entropy`.
impl DevicePass<Mps> for LinearCrossEntropy {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Mps>> {
        when(
            config.fuse && config.fused_linear_cross_entropy,
            LinearCrossEntropy,
        )
    }
}
