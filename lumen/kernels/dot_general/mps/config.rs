#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Algorithm {
    Tiled,
    Mpp,
    Skinny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmConfig {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub threads: usize,
    pub cols: usize,
    pub prefetch: bool,
    pub algorithm: Algorithm,
}

pub(super) const MAX_THREADGROUP_BYTES: usize = 32 << 10;
