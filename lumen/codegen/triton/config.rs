//! Autotuning candidates and launch defaults for Triton kernels: GEMM, looped (row/elementwise),
//! reduction and flash-attention.

/// Tile shape and launch parameters for one Triton GEMM candidate.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Config {
    /// Output rows per program.
    pub block_m: usize,
    /// Output columns per program.
    pub block_n: usize,
    /// Reduction-axis slice loaded per loop iteration.
    pub block_k: usize,
    /// Rows of program tiles grouped together so neighbouring programs share operand tiles in L2.
    pub group_m: usize,
    /// Software pipelining depth (`num_stages`).
    pub stages: usize,
    /// Warps per program (`num_warps`).
    pub warps: usize,
}

/// Builds a GEMM config with the default `group_m` of 8.
const fn config(
    block_m: usize,
    block_n: usize,
    block_k: usize,
    stages: usize,
    warps: usize,
) -> Config {
    Config {
        block_m,
        block_n,
        block_k,
        group_m: 8,
        stages,
        warps,
    }
}

/// GEMM candidates the autotuner benchmarks, largest tiles first.
pub(crate) const CONFIGS: [Config; 7] = [
    config(128, 256, 64, 3, 8),
    config(64, 256, 32, 4, 4),
    config(128, 128, 32, 4, 4),
    config(128, 64, 32, 4, 4),
    config(64, 128, 32, 4, 4),
    config(128, 32, 32, 4, 4),
    config(64, 32, 32, 5, 2),
];

/// Index into `CONFIGS` used when autotuning is off.
pub(crate) const UNTUNED: usize = 2;

/// Block size and warp count for a kernel that loops over a flat range or a row.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct LoopConfig {
    /// Elements handled per program per iteration.
    pub block_size: usize,
    pub warps: usize,
}

/// Block sizes swept for looped kernels.
const BLOCK_SIZES: [usize; 9] = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
/// Warp counts swept for looped kernels.
const WARPS: [usize; 4] = [4, 8, 16, 32];

/// Cartesian product of `BLOCK_SIZES` and `WARPS`, block-size major.
pub(crate) const LOOP_CONFIGS: [LoopConfig; BLOCK_SIZES.len() * WARPS.len()] = {
    let mut configs = [LoopConfig {
        block_size: 0,
        warps: 0,
    }; BLOCK_SIZES.len() * WARPS.len()];
    let mut i = 0;
    while i < configs.len() {
        configs[i] = LoopConfig {
            block_size: BLOCK_SIZES[i / WARPS.len()],
            warps: WARPS[i % WARPS.len()],
        };
        i += 1;
    }
    configs
};

/// Index into `LOOP_CONFIGS` used when autotuning is off: block size 1024 with 4 warps.
pub(crate) const UNTUNED_LOOP: usize = 5 * WARPS.len();

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

/// Most lanes per output when the reduced axis is innermost.
pub(super) const ROW_LANES: usize = 256;
/// Most lanes per output for a split reduction over the innermost axis.
pub(super) const SPLIT_ROW_LANES: usize = 512;
/// Most lanes per output when the reduced axis is strided (outputs are contiguous instead).
pub(super) const COLUMN_LANES: usize = 16;
/// Reductions with fewer outputs than this are split across programs to fill the GPU.
pub(super) const SPLIT_BELOW: usize = 1024;
/// Reduced elements each split aims to cover.
pub(super) const SPLIT_SPAN: usize = 8192;
/// Upper bound on splits per output.
pub(super) const MOST_SPLITS: usize = 512;
/// Widest row a normalization row kernel holds whole in registers (XMA's `MAX_TRITON_BLOCK_SIZE`);
/// wider rows are walked in chunks of `BLOCK_SIZE`, a pass a reduction.
pub(super) const MOST_WHOLE_ROW: usize = 65536;

/// Loop configs for a kernel of `lanes`: those whose block covers them, or, when it is wider than
/// every block (a whole row past 8192), its lanes as the block with each warp count.
pub(crate) fn loop_configs(lanes: usize) -> Vec<LoopConfig> {
    let fitting: Vec<LoopConfig> = LOOP_CONFIGS
        .into_iter()
        .filter(|c| c.block_size >= lanes)
        .collect();
    match fitting.is_empty() {
        true => WARPS
            .iter()
            .map(|&warps| LoopConfig {
                block_size: lanes,
                warps,
            })
            .collect(),
        false => fitting,
    }
}

/// The loop config used when autotuning is off: `UNTUNED_LOOP`, its block widened to `lanes` when
/// they don't fit.
pub(crate) fn untuned_loop(lanes: usize) -> LoopConfig {
    let config = LOOP_CONFIGS[UNTUNED_LOOP];
    LoopConfig {
        block_size: config.block_size.max(lanes),
        ..config
    }
}
