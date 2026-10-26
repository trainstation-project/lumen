//! Compiler configuration for MPS devices.

use super::{CompilerConfig, device_config};

device_config! {
    MpsConfig, PyMpsConfig, "MpsConfig", mps, crate::Device::Mps(0), {
        fuse, set_fuse: bool = true;
        normalization_diamonds, set_normalization_diamonds: bool = true;
        reduction_epilogues, set_reduction_epilogues: bool = true;
        contraction_epilogues, set_contraction_epilogues: bool = true;
        multi_output_fusion, set_multi_output_fusion: bool = true;
        horizontal_fusion, set_horizontal_fusion: bool = true;
        online_softmax, set_online_softmax: bool = true;
        fused_linear_cross_entropy, set_fused_linear_cross_entropy: bool = true;
        split_k, set_split_k: bool = true;
        deterministic, set_deterministic: bool = false;
        neural_engine, set_neural_engine: bool = false;
        row_cache, set_row_cache: usize = 8;
        safe_kernels, set_safe_kernels: bool = false;
        autotune, set_autotune: bool = true;
    }
}

/// Exposes the shared settings to the compiler.
impl CompilerConfig for MpsConfig {
    /// Whether to merge dots that share an operand.
    fn merge_dots(&self) -> bool {
        self.merge_dots
    }

    /// Whether to fuse attention into flash-attention kernels.
    fn flash_attention(&self) -> bool {
        self.flash_attention
    }

    /// Rows per chunk of the fused linear cross entropy; `None` disables it.
    fn fused_linear_cross_entropy_chunk_size(&self) -> Option<usize> {
        self.fused_linear_cross_entropy_chunk_size
    }

    /// Byte budget for the compiled plan's memory; `None` when unlimited.
    fn memory_limit(&self) -> Option<usize> {
        (self.memory_limit > 0).then_some(self.memory_limit)
    }

    /// Directory to dump the graph to after each pass.
    fn dump_to(&self) -> Option<&str> {
        self.dump_to.as_deref()
    }
}
