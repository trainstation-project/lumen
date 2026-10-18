//! Attention's backward as flash-attention kernels
//! ([`attention::fuse_backward`]).

use super::{Kernel, MpsModule, codegen, request};
use crate::graph_compiler::attention;
use crate::graph_compiler::pass::Pass;

/// Each attention's backward one kernel adding dQ atomically (unless
/// `deterministic`, or its blocks do not fit), or two (dV and dK, dQ), the
/// probabilities recomputed from the forward's log-sum-exp.
pub(super) struct FlashAttentionBackward;

impl<'a> Pass<MpsModule<'a>> for FlashAttentionBackward {
    fn name(&self) -> &'static str {
        "flash_attention_backward"
    }

    fn run(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let deterministic = m.options.config.deterministic;
        let kernels = &mut m.kernels;
        m.graph = attention::fuse_backward(
            &m.graph,
            |b| !deterministic && codegen::backward_block_with_dq(b).is_some(),
            |_| request(kernels, Kernel::AttentionBackward),
        );
        Ok(())
    }
}
