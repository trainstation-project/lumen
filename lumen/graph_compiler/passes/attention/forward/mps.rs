//! How MPS fuses forward attention: its answers to the forward `FlashAttention` pass.

use crate::codegen::mps::{Kernel, request};
use crate::graph::Var;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::attention::Attentions;

impl Attentions for DeviceModule<'_, Mps> {
    /// Whether attention may absorb elementwise epilogues.
    fn contraction_epilogues(&self) -> bool {
        self.options.config.contraction_epilogues
    }

    /// Graph values of the by-value scalar inputs.
    fn scalars(&self) -> Vec<Var> {
        self.scalars
            .iter()
            .map(|&i| self.graph.inputs()[i])
            .collect()
    }

    /// Requests an attention kernel, returning its pending name.
    fn kernel(&mut self) -> String {
        request(Kernel::Attention)
    }
}
