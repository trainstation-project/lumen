//! Triton flash-attention autotuning candidates and launch default.
//!
//! Autotuning Triton flash attention: its configs for attention steps.

use crate::autotune::Kernels;
use crate::codegen::attention::AttentionTuning;
use crate::codegen::attention::triton::launch;
use crate::codegen::dot_general::triton::launch::Run;
use crate::codegen::utils::triton::Triton;
use crate::graph_compiler::CompilerConfig;
use crate::plan::{TritonStep, Tuned};

/// Tile shape and launch parameters for one Triton flash-attention candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct AttentionConfig {
    /// Query rows per tile.
    pub block_m: usize,
    /// Key rows per tile.
    pub block_n: usize,
    /// Software pipelining depth (`num_stages`).
    pub stages: usize,
    /// Warps per program (`num_warps`).
    pub warps: usize,
}

/// Builds a flash-attention config.
const fn attention(block_m: usize, block_n: usize, stages: usize, warps: usize) -> AttentionConfig {
    AttentionConfig {
        block_m,
        block_n,
        stages,
        warps,
    }
}

/// Flash-attention candidates the autotuner benchmarks, largest tiles first.
pub(crate) const ATTENTION_CONFIGS: [AttentionConfig; 9] = [
    attention(128, 128, 2, 8),
    attention(128, 64, 3, 8),
    attention(128, 64, 3, 4),
    attention(64, 128, 2, 4),
    attention(64, 64, 3, 4),
    attention(64, 64, 2, 4),
    attention(64, 32, 2, 4),
    attention(32, 32, 1, 4),
    attention(16, 32, 1, 4),
];

/// The config used when autotuning is off: 64x64 tiles, or 32x32 when a head row is over 256 bytes, so
/// its tiles fit in shared memory.
pub(crate) fn untuned_attention(head_bytes: usize) -> AttentionConfig {
    match head_bytes <= 256 {
        true => ATTENTION_CONFIGS[5],
        false => ATTENTION_CONFIGS[7],
    }
}

#[cfg(feature = "python")]
crate::autotune::python::config_class! {
    /// `lumen.autotune.TritonAttentionConfig`: a tile of the Triton backend's flash attention.
    PyTritonAttentionConfig(AttentionConfig) = "TritonAttentionConfig" {
        block_m: usize,
        block_n: usize,
        stages: usize,
        warps: usize,
    }

    /// Every config the autotuner times, in its order.
    #[staticmethod]
    fn candidates() -> Vec<Self> {
        ATTENTION_CONFIGS.iter().copied().map(PyTritonAttentionConfig).collect()
    }
}

/// Autotuning candidates for flash-attention steps.
pub(crate) struct Attentions;

impl Kernels for Attentions {
    type Step = TritonStep;
    type Config = AttentionConfig;

    /// Every attention config for an attention step, none otherwise.
    fn configs(&self, step: &TritonStep) -> Vec<AttentionConfig> {
        match launch::is_attention(&step.primitive) {
            true => ATTENTION_CONFIGS.to_vec(),
            false => Vec::new(),
        }
    }

    /// A launch of `step` with `config`.
    fn build(&mut self, _: &TritonStep, config: AttentionConfig) -> Tuned {
        let run: Run = Box::new(move |step, inputs, out, outs| {
            launch::attention(step, inputs, out, outs, Some(&config))
        });

        Tuned::new(format!("{config:?}"), run)
    }
}

/// Triton flash attention's configs.
impl AttentionTuning for Triton {
    type Tuner = Attentions;

    /// The attention tuner, the same under every config.
    fn attention_tuner(_: &CompilerConfig) -> Attentions {
        Attentions
    }
}
