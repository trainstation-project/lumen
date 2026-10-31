//! Passes fusing forward and backward attention into flash-attention kernels.

use crate::graph::Var;
use crate::graph_compiler::common::attention::{self, Backward, Epilogues};
use crate::graph_compiler::pass::{CompilerPass, Module};

/// A module whose backend can generate flash-attention kernels.
pub(crate) trait Attentions: Module {
    /// Whether elementwise epilogues may be fused into the attention.
    fn contraction_epilogues(&self) -> bool;
    /// Which epilogues the backend accepts: by default any leaf, without reshapes.
    fn epilogues(&self) -> Epilogues {
        Epilogues {
            leaves: |_, _| true,
            reshaped: false,
            transposed: true,
        }
    }
    /// Values passed to kernels by value rather than as buffers.
    fn scalars(&self) -> Vec<Var>;
    /// Requests a kernel for the next fusion and returns its name.
    fn kernel(&mut self) -> String;
    /// Whether the causal diagonal may include a runtime offset: by default the kernel reads it at run time.
    fn runtime_offsets(&self) -> bool {
        true
    }
}

/// A module whose backend can generate flash-attention backward kernels.
pub(crate) trait AttentionBackwards: Module {
    /// Which backwards run as one kernel computing dK, dV and dQ; the others get a dK/dV kernel and
    /// a dQ kernel.
    fn together(&self) -> impl Fn(&Backward) -> bool + 'static;
    /// Requests a kernel for the next backward fusion and returns its name.
    fn backward_kernel(&mut self) -> String;
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

/// Compiler pass running [`attention::fuse_backward`].
pub(crate) struct FlashAttentionBackward;

impl<M: AttentionBackwards> CompilerPass<M> for FlashAttentionBackward {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "flash_attention_backward"
    }

    /// Replaces each attention backward in the module's graph with its flash-attention fusions.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let graph = m.graph().clone();
        let together = m.together();
        *m.graph_mut() = attention::fuse_backward(&graph, together, |_| m.backward_kernel());
        Ok(())
    }
}
