//! What Triton runs for this pass.

use super::AttentionDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::attention::{FlashAttention, FlashAttentionBackward};
use crate::graph_compiler::pipeline::{Pass, when};

/// Triton's answers to the pass.
impl AttentionDevice for Triton {
    /// With `fuse` and `flash_attention`.
    fn flash_attention<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        when(config.fuse && config.flash_attention, FlashAttention)
    }

    /// With `fuse` and `flash_attention`.
    fn flash_attention_backward<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        when(
            config.fuse && config.flash_attention,
            FlashAttentionBackward,
        )
    }
}
