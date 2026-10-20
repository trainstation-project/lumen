use super::{Kernel, MpsModule, codegen, request};
use crate::graph_compiler::attention;
use crate::graph_compiler::pass::CompilerPass;

pub(super) struct FlashAttentionBackward;

impl<'a> CompilerPass<MpsModule<'a>> for FlashAttentionBackward {
    fn name(&self) -> &'static str {
        "flash_attention_backward"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let deterministic = m.options.config.deterministic;
        let kernels = &mut m.kernels;
        m.graph = attention::fuse_backward(
            &m.graph,
            |b| !deterministic && codegen::backward_block_with_dq(b).is_some(),
            |_| true,
            |_| request(kernels, Kernel::AttentionBackward),
        );
        Ok(())
    }
}
