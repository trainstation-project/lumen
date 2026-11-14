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

/// Flash-attention candidates the autotuner benchmarks, largest tiles first.
pub(crate) const ATTENTION_CONFIGS: [AttentionConfig; 9] = [
    AttentionConfig {
        block_m: 128,
        block_n: 128,
        stages: 2,
        warps: 8,
    },
    AttentionConfig {
        block_m: 128,
        block_n: 64,
        stages: 3,
        warps: 8,
    },
    AttentionConfig {
        block_m: 128,
        block_n: 64,
        stages: 3,
        warps: 4,
    },
    AttentionConfig {
        block_m: 64,
        block_n: 128,
        stages: 2,
        warps: 4,
    },
    AttentionConfig {
        block_m: 64,
        block_n: 64,
        stages: 3,
        warps: 4,
    },
    AttentionConfig {
        block_m: 64,
        block_n: 64,
        stages: 2,
        warps: 4,
    },
    AttentionConfig {
        block_m: 64,
        block_n: 32,
        stages: 2,
        warps: 4,
    },
    AttentionConfig {
        block_m: 32,
        block_n: 32,
        stages: 1,
        warps: 4,
    },
    AttentionConfig {
        block_m: 16,
        block_n: 32,
        stages: 1,
        warps: 4,
    },
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
