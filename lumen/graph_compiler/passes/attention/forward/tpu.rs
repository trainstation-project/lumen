//! How TPU fuses forward attention: its answers to the forward `FlashAttention` pass.

use crate::graph::{Graph, Var};
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::attention::{self, Attentions};

/// Flash-attention matching for TPU: scalar epilogue leaves, no extra scalars, static offsets.
impl Attentions for DeviceModule<'_, Tpu> {
    /// Whether attention may absorb elementwise epilogues.
    fn contraction_epilogues(&self) -> bool {
        self.options.config.contraction_epilogues
    }

    /// Epilogues reading only scalar leaves, through reshapes but not transposes.
    fn epilogues(&self) -> attention::Epilogues {
        attention::Epilogues {
            leaves: |graph, v| graph.type_of(v).shape.is_empty(),
            reshaped: true,
            transposed: false,
        }
    }

    /// None: TPU kernels take no by-value scalars.
    fn scalars(&self) -> Vec<Var> {
        Vec::new()
    }

    /// Names the forward fusion `flash_attention`.
    fn kernel(&mut self) -> String {
        "flash_attention".into()
    }

    /// No: the TPU kernel's causal diagonal is fixed at compile time.
    fn runtime_offsets(&self) -> bool {
        false
    }
}

/// How a forward fusion `body` writes its output.
pub(in crate::graph_compiler::passes::attention) fn outputs(
    body: &Graph,
) -> Vec<attention::Access> {
    attention::of_body(body)
        .map(|a| a.out)
        .into_iter()
        .collect()
}
