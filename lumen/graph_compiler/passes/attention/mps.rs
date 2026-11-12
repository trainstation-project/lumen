//! What MPS runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::attention::{FlashAttention, FlashAttentionBackward};
use crate::graph_compiler::pipeline::{Pass, when};

/// With `fuse` and `flash_attention`.
impl DevicePass<Mps> for FlashAttention {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Mps>> {
        when(config.fuse && config.flash_attention, FlashAttention)
    }
}

/// With `fuse` and `flash_attention`.
impl DevicePass<Mps> for FlashAttentionBackward {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Mps>> {
        when(
            config.fuse && config.flash_attention,
            FlashAttentionBackward,
        )
    }
}
