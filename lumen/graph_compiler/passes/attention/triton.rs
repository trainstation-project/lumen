//! Triton's answers to the attention passes: which attentions its kernels run, and how they are named.

use crate::codegen::devices::triton::Triton;
use crate::graph_compiler::device::DeviceModule;

/// Flash-attention matching for Triton, as on MPS: elementwise epilogues reading any leaf, runtime causal
/// offsets (decode).
impl super::Attentions for DeviceModule<'_, Triton> {
    /// Elementwise epilogues are fused into the forward kernel's output store.
    fn contraction_epilogues(&self) -> bool {
        true
    }

    /// Graph values of the by-value scalar inputs.
    fn scalars(&self) -> Vec<crate::graph::Var> {
        crate::graph_compiler::positions(&self.options.scalars)
            .into_iter()
            .map(|i| self.graph.inputs()[i])
            .collect()
    }

    /// Every forward attention runs on the one forward kernel.
    fn kernel(&mut self) -> String {
        super::FORWARD.into()
    }
}

/// Each attention backward runs as one kernel adding dQ atomically as it computes dK and dV, or, when
/// `deterministic`, a dK/dV kernel and a dQ kernel.
impl super::AttentionBackwards for DeviceModule<'_, Triton> {
    /// Every backward runs as one kernel unless deterministic, so dQ is summed over key blocks in a fixed order.
    fn together(&self) -> impl Fn(&super::Backward) -> bool + 'static {
        let deterministic = self.options.config.deterministic;
        move |_| !deterministic
    }

    /// Every backward fusion runs on the backward kernels.
    fn backward_kernel(&mut self) -> String {
        super::BACKWARD.into()
    }
}
