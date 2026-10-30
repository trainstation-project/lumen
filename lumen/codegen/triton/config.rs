//! Autotuning candidates and launch defaults for Triton kernels: GEMM, looped (row/elementwise),
//! reduction and flash-attention.

use crate::graph_compiler::common::diamonds::Partials;

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
pub(crate) const SPLIT_BELOW: usize = 1024;
/// Reduced elements each split aims to cover.
pub(crate) const SPLIT_SPAN: usize = 8192;
/// Upper bound on splits per output.
pub(crate) const MOST_SPLITS: usize = 512;
/// Widest row a normalization row kernel holds whole in registers (XMA's `MAX_TRITON_BLOCK_SIZE`);
/// wider rows are walked in chunks of `BLOCK_SIZE`, a pass a reduction.
pub(super) const MOST_WHOLE_ROW: usize = 65536;

/// How column sums (a norm weight's gradient) ride along row kernels: rows up to 8192 wide, each
/// program holding a partial per column, split into 32 to 256 row groups.
pub(crate) const ROW_PARTIALS: Partials = Partials {
    columns: 8192,
    groups: 256,
    fewest_groups: 32,
};

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

/// Widest row a GEMM's tile holds whole for a row epilogue (`BLOCK_SIZE_N`); wider products are written
/// and normalized by a row kernel. On an H100 (bf16, autotuned) the fused kernel matches a GEMM and a
/// row kernel at 256 but is slower at 512 (121 vs 112 µs) and 1024 (252 vs 132 µs): a tile of whole
/// wide rows leaves too few programs.
pub(super) const MOST_GEMM_ROW: usize = 256;

/// Most accumulator elements a row-epilogue GEMM tile holds (`BLOCK_SIZE_M x BLOCK_SIZE_N`).
const ROW_TILE_ELEMENTS: usize = 32 << 10;

/// GEMM candidates for a row epilogue over rows `n` wide: the tile's `BLOCK_SIZE_N` covers the row, each
/// program owning `block_m` whole rows, as many as the accumulator budget allows.
pub(crate) fn row_gemm_configs(n: usize) -> Vec<Config> {
    let block_n = n.next_power_of_two().max(16);
    let mut configs = Vec::new();

    for block_m in [16, 32, 64] {
        if block_m > 16 && block_m * block_n > ROW_TILE_ELEMENTS {
            continue;
        }

        for block_k in [32, 64] {
            for warps in [4, 8] {
                configs.push(Config {
                    block_m,
                    block_n,
                    block_k,
                    group_m: 1,
                    stages: 2,
                    warps,
                });
            }
        }
    }

    configs
}

/// The row-epilogue GEMM config used when autotuning is off: the most rows the accumulator budget
/// allows (at least 16), `BLOCK_SIZE_K` 32, two stages, 8 warps.
pub(crate) fn untuned_row_gemm(n: usize) -> Config {
    let block_n = n.next_power_of_two().max(16);

    Config {
        block_m: (ROW_TILE_ELEMENTS / block_n).clamp(16, 64),
        block_n,
        block_k: 32,
        group_m: 1,
        stages: 2,
        warps: 8,
    }
}

/// Columns of a wide row each partial covers: the narrowest GEMM tile's `BLOCK_SIZE_N`, so every
/// candidate tile (each a multiple of it) writes whole partials and the autotuner picks any.
const PARTIAL_WIDTH: usize = 32;

/// Narrowest partial: a row width no power of two this wide divides (an odd one) stays whole.
const LEAST_PARTIAL_WIDTH: usize = 8;

/// The partial width for a GEMM whose product rows are `n` wide and normalized after it, rows too wide
/// for one tile (past `MOST_GEMM_ROW`): the widest power of two up to `PARTIAL_WIDTH` dividing `n`, so
/// rows of any width MPS splits are split too (1000 in partials of 8).
pub(crate) fn partial_tile(n: usize) -> Option<usize> {
    let tile = (1 << n.trailing_zeros()).min(PARTIAL_WIDTH);
    (n > MOST_GEMM_ROW && tile >= LEAST_PARTIAL_WIDTH).then_some(tile)
}
