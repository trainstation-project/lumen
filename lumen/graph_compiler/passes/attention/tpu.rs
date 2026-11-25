//! TPU's answers to the attention passes: which attentions its kernels run, and how they are named.

use crate::codegen::devices::tpu::Tpu;
use crate::graph_compiler::device::DeviceModule;

/// Flash-attention matching for TPU: scalar epilogue leaves, no extra scalars, runtime offsets for decode.
impl super::Attentions for DeviceModule<'_, Tpu> {
    /// Whether attention may absorb elementwise epilogues.
    fn contraction_epilogues(&self) -> bool {
        self.options.config.contraction_epilogues
    }

    /// Epilogues reading only scalar leaves, through reshapes but not transposes.
    fn epilogues(&self) -> super::Epilogues {
        super::Epilogues {
            leaves: |graph, v| graph.type_of(v).shape.is_empty(),
            reshaped: true,
            transposed: false,
        }
    }

    /// None: TPU kernels take no by-value scalars.
    fn scalars(&self) -> Vec<crate::graph::Var> {
        Vec::new()
    }

    /// Names the forward fusion `flash_attention`.
    fn kernel(&mut self) -> String {
        "flash_attention".into()
    }

    /// Yes: the decode kernel (one query a head) reads its causal offset at run time.
    fn runtime_offsets(&self) -> bool {
        true
    }
}

/// Each attention backward runs as a dK/dV kernel and a dQ kernel.
impl super::AttentionBackwards for DeviceModule<'_, Tpu> {
    /// None: the TPU backward runs dK/dV and dQ as separate kernels.
    fn together(&self) -> impl Fn(&super::Backward) -> bool + 'static {
        |_| false
    }

    /// Names the backward fusion `flash_attention_backward`.
    fn backward_kernel(&mut self) -> String {
        "flash_attention_backward".into()
    }
}

/// TPU's attention kernels read and write whole tiles: `attention_layouts` adapts its attention fusions, checking
/// each against `codegen::attention::tpu::kernel`.
impl super::layouts::TiledAttentions for Tpu {
    /// What `codegen::attention::tpu::kernel` makes of the fusion.
    fn check(name: &str, body: &crate::graph::Graph) -> Result<(), String> {
        crate::codegen::attention::tpu::kernel(name, body).map(|_| ())
    }
}
