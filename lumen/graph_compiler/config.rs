use std::sync::{PoisonError, RwLock};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompilerConfig {
    pub fuse: bool,
    pub merge_dots: bool,
    pub normalization_diamonds: bool,
    pub reduction_epilogues: bool,
    pub contraction_epilogues: bool,
    pub multi_output_fusion: bool,
    pub horizontal_fusion: bool,
    pub online_softmax: bool,
    pub flash_attention: bool,
    pub fused_linear_cross_entropy: bool,
    pub split_k: bool,
    pub deterministic: bool,
    pub neural_engine: bool,
    pub row_cache: usize,
    pub memory_limit: usize,
    pub fused_linear_cross_entropy_chunk_size: Option<usize>,
    pub safe_kernels: bool,
    pub use_triton_backend: bool,
    pub autotune: bool,
    pub dump_to: Option<String>,
}

impl CompilerConfig {
    pub(crate) fn memory_limit(&self) -> Option<usize> {
        (self.memory_limit > 0).then_some(self.memory_limit)
    }
}

impl Default for CompilerConfig {
    fn default() -> Self {
        CompilerConfig {
            fuse: true,
            merge_dots: true,
            normalization_diamonds: true,
            reduction_epilogues: true,
            contraction_epilogues: true,
            multi_output_fusion: true,
            horizontal_fusion: true,
            online_softmax: true,
            flash_attention: true,
            fused_linear_cross_entropy: true,
            split_k: true,
            deterministic: false,
            neural_engine: false,
            row_cache: 8,
            memory_limit: 0,
            fused_linear_cross_entropy_chunk_size: None,
            safe_kernels: false,
            use_triton_backend: false,
            autotune: true,
            dump_to: None,
        }
    }
}

static CONFIG: RwLock<Option<CompilerConfig>> = RwLock::new(None);

static VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn version() -> u64 {
    VERSION.load(std::sync::atomic::Ordering::Acquire)
}

pub fn config() -> CompilerConfig {
    let config = CONFIG.read().unwrap_or_else(PoisonError::into_inner);
    config.clone().unwrap_or_default()
}

pub fn update(change: impl FnOnce(&mut CompilerConfig)) {
    let mut config = CONFIG.write().unwrap_or_else(PoisonError::into_inner);
    let mut next = config.clone().unwrap_or_default();
    change(&mut next);
    *config = Some(next);
    VERSION.fetch_add(1, std::sync::atomic::Ordering::Release);
}
