//! How Triton fuses forward attention: its answers to the forward `FlashAttention` pass.

use crate::graph::Var;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::attention::{Attentions, FORWARD};
use crate::graph_compiler::positions;

/// Flash-attention matching for Triton, as on MPS: elementwise epilogues reading any leaf, runtime causal
/// offsets (decode).
impl Attentions for DeviceModule<'_, Triton> {
    /// Elementwise epilogues are fused into the forward kernel's output store.
    fn contraction_epilogues(&self) -> bool {
        true
    }

    /// Graph values of the by-value scalar inputs.
    fn scalars(&self) -> Vec<Var> {
        positions(&self.options.scalars)
            .into_iter()
            .map(|i| self.graph.inputs()[i])
            .collect()
    }

    /// Every forward attention runs on the one forward kernel.
    fn kernel(&mut self) -> String {
        FORWARD.into()
    }
}
