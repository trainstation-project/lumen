#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Config {
    pub block_m: usize,
    pub block_n: usize,
    pub block_k: usize,
    pub group_m: usize,
    pub stages: usize,
    pub warps: usize,
}

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

pub(crate) const CONFIGS: [Config; 7] = [
    config(128, 256, 64, 3, 8),
    config(64, 256, 32, 4, 4),
    config(128, 128, 32, 4, 4),
    config(128, 64, 32, 4, 4),
    config(64, 128, 32, 4, 4),
    config(128, 32, 32, 4, 4),
    config(64, 32, 32, 5, 2),
];

pub(crate) const UNTUNED: usize = 2;
