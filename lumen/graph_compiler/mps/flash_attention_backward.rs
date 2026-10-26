//! Pass that fuses attention backward subgraphs into the MPS flash attention backward kernel.

use super::{Kernel, MpsModule, codegen, request};
use crate::graph_compiler::common::attention;
use crate::graph_compiler::pass::CompilerPass;

/// Compiler pass replacing attention backward patterns with `AttentionBackward` kernels.
pub(super) struct FlashAttentionBackward;

impl<'a> CompilerPass<MpsModule<'a>> for FlashAttentionBackward {
    fn name(&self) -> &'static str {
        "flash_attention_backward"
    }

    /// Fuses backward blocks whose dQ fits the kernel, unless deterministic mode forbids its atomic dQ accumulation.
    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let deterministic = m.options.config.mps.deterministic;
        let kernels = &mut m.kernels;
        m.graph = attention::fuse_backward(
            &m.graph,
            |b| !deterministic && codegen::backward_block_with_dq(b).is_some(),
            |_| request(kernels, Kernel::AttentionBackward),
        );
        Ok(())
    }
}
