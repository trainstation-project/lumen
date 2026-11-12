//! CUDA (CuTe DSL) flash-attention autotuning candidates and the config it launches with untuned.

/// CTA tile, pipeline stages, warps and Q placement of one Ampere attention forward (`ampere.py`'s
/// `AmpereAttention`) candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Query rows per CTA.
    pub tile_m: usize,
    /// Keys per K/V block.
    pub tile_n: usize,
    /// K/V blocks in flight in shared memory.
    pub stages: usize,
    /// Warps per CTA, stacked along the query rows (16 each).
    pub warps: usize,
    /// Whether Q is held in registers, V reusing its shared memory.
    pub q_in_regs: bool,
}

/// Builds an attention config.
const fn config(
    tile_m: usize,
    tile_n: usize,
    stages: usize,
    warps: usize,
    q_in_regs: bool,
) -> Config {
    Config {
        tile_m,
        tile_n,
        stages,
        warps,
        q_in_regs,
    }
}

/// Candidates the autotuner benchmarks: on an A100 (bf16, batch 4 x 16 heads x 4096), the fastest few at each head
/// dim, causal or not, of a sweep over 64- and 128-row tiles, 32- to 128-key blocks, 1 or 2 stages, 4 or 8 warps
/// and Q in registers or not: 128 x 128 x 2 stages at head dim 64 (1.23x torch's FlashAttention-2), 64 x 64 with Q
/// in registers at 128 (1.11x; 1.05x causal).
pub(crate) const CONFIGS: [Config; 8] = [
    config(128, 128, 2, 4, false),
    config(128, 128, 1, 4, false),
    config(128, 64, 1, 4, false),
    config(128, 64, 2, 4, true),
    config(64, 128, 1, 4, false),
    config(64, 64, 1, 4, false),
    config(64, 64, 1, 4, true),
    config(64, 64, 2, 4, true),
];

/// Fewest multiply-adds (both products) an attention is autotuned at: below, it takes tens of microseconds whatever
/// its config, while timing every candidate costs a CuTe compile each.
pub(crate) const TUNED_MACS: usize = 1 << 30;

/// Shared memory an SM80 CTA may use (an A100's 164 KiB, less the 1 KiB the driver reserves).
const SHARED_BYTES: usize = 163 * 1024;

/// Whether `config` runs head dims `h` (Q and K) and `hv` (V) of 16-bit elements: its warps' rows fit the tile and
/// its Q, K and V tiles fit in shared memory (padded to multiples of 16, each 1 KiB aligned).
pub(crate) fn fits(config: &Config, h: usize, hv: usize) -> bool {
    let (h, hv) = (h.next_multiple_of(16), hv.next_multiple_of(16));
    let kib = |elements: usize| (2 * elements).next_multiple_of(1024);
    let q = kib(config.tile_m * h);
    let k = kib(config.tile_n * h * config.stages);
    let v = kib(config.tile_n * hv * config.stages);
    let qv = if config.q_in_regs { q.max(v) } else { q + v };

    config.warps * 16 <= config.tile_m
        && (2 * config.tile_m).is_multiple_of(32 * config.warps)
        && qv + k <= SHARED_BYTES
}

/// The candidates that run head dims `h` and `hv`.
pub(crate) fn configs(h: usize, hv: usize) -> Vec<Config> {
    CONFIGS.iter().copied().filter(|c| fits(c, h, hv)).collect()
}

/// The config used untuned for head dims `h` and `hv`: the sweep's best for head dim 64 up to it, 64 x 64 with Q in
/// registers above (which every head dim up to 256 fits).
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub(crate) fn untuned(h: usize, hv: usize) -> Config {
    match h.max(hv) <= 64 {
        true => CONFIGS[0],
        false => CONFIGS[6],
    }
}

#[cfg(feature = "python")]
crate::autotune::python::config_class! {
    /// `lumen.autotune.CudaAttentionConfig`: a config of the CUDA backend's SM80 flash-attention forward, one of its
    /// autotuning candidates or any other.
    PyCudaAttentionConfig(Config) = "CudaAttentionConfig" {
        tile_m: usize,
        tile_n: usize,
        stages: usize = 1,
        warps: usize = 4,
        q_in_regs: bool = false,
    }

    /// Every config the autotuner times, in its order.
    #[staticmethod]
    fn candidates() -> Vec<Self> {
        CONFIGS.iter().copied().map(PyCudaAttentionConfig).collect()
    }

    /// Whether the config runs head dims `h` (Q and K) and `hv` (V): forced on an attention it doesn't fit, the
    /// attention keeps its untuned config.
    fn fits(&self, h: usize, hv: usize) -> bool {
        fits(&self.0, h, hv)
    }
}
