//! CUDA (CuTe DSL) GEMM autotuning candidates and the tile it launches with untuned.

/// CTA tile and warps of one Ampere GEMM (`gemm_sm80.AmpereGemm`) candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Output rows per CTA.
    pub tile_m: usize,
    /// Output columns per CTA.
    pub tile_n: usize,
    /// Reduction-axis slice each pipeline stage holds.
    pub tile_k: usize,
    /// Warps per CTA (4 or 8), laid out 2 along M by `warps / 2` along N.
    pub warps: usize,
}

/// Builds a GEMM config.
const fn config(tile_m: usize, tile_n: usize, tile_k: usize, warps: usize) -> Config {
    Config {
        tile_m,
        tile_n,
        tile_k,
        warps,
    }
}

/// GEMM candidates the autotuner benchmarks, largest tiles first. On an H100 (bf16) 128 x 256 of 64 x 64 warp tiles
/// leads from 4096³ up (457 vs 405 TFLOPS for 128 x 128), and the 64-row tiles at 128 x 4096 x 4096 (142 vs 70),
/// where bigger tiles leave SMs idle.
pub(crate) const CONFIGS: [Config; 7] = [
    config(128, 256, 32, 8),
    config(128, 256, 64, 8),
    config(128, 128, 32, 4),
    config(128, 64, 32, 4),
    config(64, 128, 32, 4),
    config(64, 64, 32, 4),
    config(64, 64, 64, 4),
];

/// Fewest multiply-adds a matmul is autotuned at (1024³): below, the GEMM takes tens of microseconds whatever its tile,
/// while timing every candidate costs a CuTe compile each.
pub(crate) const TUNED_MACS: usize = 1 << 30;

/// Index into `CONFIGS` of the untuned tile for outputs it fills the GPU with: 128 x 256, 8 warps.
const WIDE: usize = 0;

/// Index into `CONFIGS` of the untuned tile otherwise: 128 x 128, 4 warps, two CTAs an SM.
const NARROW: usize = 2;

/// The config used untuned for a `batch` x (`m`, `n`) output on a GPU of `sms` SMs: the wide tile once its CTAs
/// alone make a wave, else the narrow one so smaller outputs still spread over every SM.
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub(crate) fn untuned(batch: usize, m: usize, n: usize, sms: usize) -> Config {
    let wide = CONFIGS[WIDE];
    let tiles = batch * m.div_ceil(wide.tile_m) * n.div_ceil(wide.tile_n);

    match tiles >= sms {
        true => wide,
        false => CONFIGS[NARROW],
    }
}
