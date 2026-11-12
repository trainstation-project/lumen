//! What Triton runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::chunked_cross_entropy::ChunkedCrossEntropy;
use crate::graph_compiler::pipeline::{Pass, when};

/// With `fuse` and `fused_linear_cross_entropy`.
impl DevicePass<Triton>
    for crate::graph_compiler::passes::chunked_cross_entropy::ChunkedCrossEntropy
{
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
        when(
            config.fuse && config.fused_linear_cross_entropy,
            ChunkedCrossEntropy,
        )
    }
}
