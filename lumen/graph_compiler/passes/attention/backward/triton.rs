//! How Triton fuses attention's backward: its answers to the `FlashAttentionBackward` pass.

use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::attention::{AttentionBackwards, BACKWARD, Backward};

/// Each attention backward runs as one kernel adding dQ atomically as it computes dK and dV, or, when
/// `deterministic`, a dK/dV kernel and a dQ kernel.
impl AttentionBackwards for DeviceModule<'_, Triton> {
    /// Every backward runs as one kernel unless deterministic, so dQ is summed over key blocks in a fixed order.
    fn together(&self) -> impl Fn(&Backward) -> bool + 'static {
        let deterministic = self.options.config.deterministic;
        move |_| !deterministic
    }

    /// Every backward fusion runs on the backward kernels.
    fn backward_kernel(&mut self) -> String {
        BACKWARD.into()
    }
}
