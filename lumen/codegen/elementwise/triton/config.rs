//! Triton loop launch configs: the autotuning candidates and default of the elementwise loops, which the
//! other kernels looping over a flat range or a row (row, reduction, scan and the in-place updates) launch
//! with too.

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
