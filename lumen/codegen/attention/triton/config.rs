//! Triton flash-attention autotuning candidates and launch default.

/// Tile shape and launch parameters for one Triton flash-attention candidate.
#[derive(Debug, Clone, Copy)]
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
