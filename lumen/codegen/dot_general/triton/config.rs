//! Triton GEMM autotuning candidates and launch defaults, plain and with a row epilogue.

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

/// Widest row a GEMM's tile holds whole for a row epilogue (`BLOCK_SIZE_N`); wider products are written
/// and normalized by a row kernel. On an H100 (bf16, autotuned) the fused kernel matches a GEMM and a
/// row kernel at 256 but is slower at 512 (121 vs 112 µs) and 1024 (252 vs 132 µs): a tile of whole
/// wide rows leaves too few programs.
pub(crate) const MOST_GEMM_ROW: usize = 256;

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
