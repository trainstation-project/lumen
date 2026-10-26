//! Pass fusing forward attention into flash-attention kernels.

use crate::graph::Var;
use crate::graph_compiler::common::attention::{self, Epilogues};
use crate::graph_compiler::pass::{CompilerPass, Module};

/// A module whose backend can generate flash-attention kernels.
pub(crate) trait Attentions: Module {
    /// Whether elementwise epilogues may be fused into the attention.
    fn contraction_epilogues(&self) -> bool;
    /// Which epilogues the backend accepts.
    fn epilogues(&self) -> Epilogues;
    /// Values passed to kernels by value rather than as buffers.
    fn scalars(&self) -> Vec<Var>;
    /// Requests a kernel for the next fusion and returns its name.
    fn kernel(&mut self) -> String;
    /// Whether the causal diagonal may include a runtime offset.
    fn runtime_offsets(&self) -> bool;
}

/// Compiler pass running [`attention::fuse`].
pub(crate) struct FlashAttention;

impl<M: Attentions> CompilerPass<M> for FlashAttention {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "flash_attention"
    }

    /// Replaces each forward attention in the module's graph with a flash-attention fusion.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let epilogues = m.contraction_epilogues().then(|| m.epilogues());
        let (graph, scalars) = (m.graph().clone(), m.scalars());
        let runtime_offsets = m.runtime_offsets();
        *m.graph_mut() =
            attention::fuse(&graph, epilogues, &scalars, runtime_offsets, |_| m.kernel());
        Ok(())
    }
}
