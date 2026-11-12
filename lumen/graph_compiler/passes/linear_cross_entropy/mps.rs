//! What MPS runs for this pass.

use super::LinearCrossEntropyDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::linear_cross_entropy::LinearCrossEntropy;
use crate::graph_compiler::pipeline::{Pass, when};

/// MPS's answers to the pass.
impl LinearCrossEntropyDevice for Mps {
    /// With `fuse` and `fused_linear_cross_entropy`.
    fn linear_cross_entropy<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        when(
            config.fuse && config.fused_linear_cross_entropy,
            LinearCrossEntropy,
        )
    }
}
