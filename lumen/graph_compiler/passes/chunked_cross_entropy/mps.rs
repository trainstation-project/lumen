//! What MPS runs for this pass.

use super::ChunkedCrossEntropyDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::chunked_cross_entropy::ChunkedCrossEntropy;
use crate::graph_compiler::pipeline::{Pass, when};

/// MPS's answers to the pass.
impl ChunkedCrossEntropyDevice for Mps {
    /// With `fuse` and `fused_linear_cross_entropy`.
    fn chunked_cross_entropy<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        when(
            config.fuse && config.fused_linear_cross_entropy,
            ChunkedCrossEntropy,
        )
    }
}
