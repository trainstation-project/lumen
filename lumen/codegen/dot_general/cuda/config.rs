//! CUDA (CuTe DSL) GEMM autotuning candidates and the tile it launches with untuned.

/// CTA tile, warps and pipeline stages of one Ampere GEMM (`ampere.py`'s `AmpereGemm`) candidate.
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
    /// A/B pipeline stages in shared memory, 0 for as many as fit at the CTAs per SM `warps` allows (two with 4
    /// warps, one with 8); a count only one CTA per SM fits runs one.
    pub stages: usize,
}

/// Builds a GEMM config.
const fn config(
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
    warps: usize,
    stages: usize,
) -> Config {
    Config {
        tile_m,
        tile_n,
        tile_k,
        warps,
        stages,
    }
}

/// GEMM candidates the autotuner benchmarks, largest tiles first, each with 3 to 5 stages where they fit (a 64-deep
/// k-tile's A/B stage is twice a 32-deep one's), and the 32-deep ones also with as many as fit (6 to 9, the stages
/// they ran with before stages were tuned). On an H100 (bf16) 128 x 256 of 64 x 64 warp tiles leads from 4096³
/// up (457 vs 405 TFLOPS for 128 x 128), and the 64-row tiles at 128 x 4096 x 4096 (142 vs 70), where bigger tiles
/// leave SMs idle; on an A100 128 x 128 x 64 at 1000 x 4096 x 1000, where 64 CTAs leave SMs idle (80 vs 86 us for
/// 128 x 128 x 32).
pub(crate) const CONFIGS: [Config; 24] = [
    config(128, 256, 32, 8, 3),
    config(128, 256, 32, 8, 4),
    config(128, 256, 32, 8, 5),
    config(128, 256, 32, 8, 0),
    config(128, 256, 64, 8, 3),
    config(128, 128, 32, 4, 3),
    config(128, 128, 32, 4, 4),
    config(128, 128, 32, 4, 5),
    config(128, 128, 64, 4, 3),
    config(128, 128, 64, 4, 4),
    config(128, 64, 32, 4, 3),
    config(128, 64, 32, 4, 4),
    config(128, 64, 32, 4, 5),
    config(128, 64, 32, 4, 0),
    config(64, 128, 32, 4, 3),
    config(64, 128, 32, 4, 4),
    config(64, 128, 32, 4, 5),
    config(64, 128, 32, 4, 0),
    config(64, 64, 32, 4, 3),
    config(64, 64, 32, 4, 4),
    config(64, 64, 32, 4, 5),
    config(64, 64, 32, 4, 0),
    config(64, 64, 64, 4, 3),
    config(64, 64, 64, 4, 4),
];

/// Fewest multiply-adds a matmul is autotuned at (1024³): below, the GEMM takes tens of microseconds whatever its
/// tile, while timing every candidate costs a CuTe compile each.
pub(crate) const TUNED_MACS: usize = 1 << 30;

/// The untuned tile for outputs it fills the GPU with: 128 x 256, 8 warps, as many stages as fit.
const WIDE: Config = config(128, 256, 32, 8, 0);

/// The untuned tile otherwise: 128 x 128, 4 warps, two CTAs an SM.
const NARROW: Config = config(128, 128, 32, 4, 0);

/// The config used untuned for a `batch` x (`m`, `n`) output on a GPU of `sms` SMs: the wide tile once its CTAs
/// alone make a wave, else the narrow one so smaller outputs still spread over every SM.
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub(crate) fn untuned(batch: usize, m: usize, n: usize, sms: usize) -> Config {
    let tiles = batch * m.div_ceil(WIDE.tile_m) * n.div_ceil(WIDE.tile_n);

    match tiles >= sms {
        true => WIDE,
        false => NARROW,
    }
}

#[cfg(feature = "python")]
crate::autotune::python::config_class! {
    /// `lumen.autotune.CudaGemmConfig`: a tile of the CUDA backend's SM80 GEMM, one of its autotuning candidates or
    /// any other (`stages` 0 for as many as fit).
    PyCudaGemmConfig(Config) = "CudaGemmConfig" {
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
        warps: usize,
        stages: usize = 0,
    }

    /// Every config the autotuner times, in its order.
    #[staticmethod]
    fn candidates() -> Vec<Self> {
        CONFIGS.iter().copied().map(PyCudaGemmConfig).collect()
    }
}
