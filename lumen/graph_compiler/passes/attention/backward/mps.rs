//! How MPS fuses attention's backward: its answers to the `FlashAttentionBackward` pass.

use crate::codegen::attention::mps::backward_block_with_dq;
use crate::codegen::mps::{Kernel, request};
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::attention::{AttentionBackwards, Backward};

impl AttentionBackwards for DeviceModule<'_, Mps> {
    /// Backwards whose dQ fits the dK/dV kernel, unless deterministic mode forbids its atomic dQ accumulation.
    fn together(&self) -> impl Fn(&Backward) -> bool + 'static {
        let deterministic = self.options.config.deterministic;
        move |b| !deterministic && backward_block_with_dq(b).is_some()
    }

    /// Requests an attention backward kernel, returning its pending name.
    fn backward_kernel(&mut self) -> String {
        request(Kernel::AttentionBackward)
    }
}
