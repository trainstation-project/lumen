//! How MPS fuses attention's backward: its answers to the `FlashAttentionBackward` pass.

use crate::codegen::attention::mps::backward_block_with_dq;
use crate::graph_compiler::common::attention::{AttentionBackwards, Backward};
use crate::graph_compiler::mps::{Kernel, MpsModule, request};

impl AttentionBackwards for MpsModule<'_> {
    /// Backwards whose dQ fits the dK/dV kernel, unless deterministic mode forbids its atomic dQ accumulation.
    fn together(&self) -> impl Fn(&Backward) -> bool + 'static {
        let deterministic = self.options.config.deterministic;
        move |b| !deterministic && backward_block_with_dq(b).is_some()
    }

    /// Requests an attention backward kernel, returning its pending name.
    fn backward_kernel(&mut self) -> String {
        request(&mut self.kernels, Kernel::AttentionBackward)
    }
}
