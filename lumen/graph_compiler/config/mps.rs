use super::{CompilerConfig, device_config};

device_config! {
    MpsConfig, PyMpsConfig, "MpsConfig", mps {
        fuse, set_fuse: bool = true;
        merge_dots, set_merge_dots: bool = true;
        normalization_diamonds, set_normalization_diamonds: bool = true;
        reduction_epilogues, set_reduction_epilogues: bool = true;
        contraction_epilogues, set_contraction_epilogues: bool = true;
        multi_output_fusion, set_multi_output_fusion: bool = true;
        horizontal_fusion, set_horizontal_fusion: bool = true;
        online_softmax, set_online_softmax: bool = true;
        flash_attention, set_flash_attention: bool = true;
        fused_linear_cross_entropy, set_fused_linear_cross_entropy: bool = true;
        fused_linear_cross_entropy_chunk_size, set_fused_linear_cross_entropy_chunk_size: Option<usize> = None;
        split_k, set_split_k: bool = true;
        deterministic, set_deterministic: bool = false;
        neural_engine, set_neural_engine: bool = false;
        row_cache, set_row_cache: usize = 8;
        safe_kernels, set_safe_kernels: bool = false;
        autotune, set_autotune: bool = true;
        memory_limit, set_memory_limit: usize = 0;
        dump_to, set_dump_to: Option<String> = None;
    }
}

impl CompilerConfig for MpsConfig {
    fn merge_dots(&self) -> bool {
        self.merge_dots
    }

    fn flash_attention(&self) -> bool {
        self.flash_attention
    }

    fn fused_linear_cross_entropy_chunk_size(&self) -> Option<usize> {
        self.fused_linear_cross_entropy_chunk_size
    }

    fn memory_limit(&self) -> Option<usize> {
        (self.memory_limit > 0).then_some(self.memory_limit)
    }

    fn dump_to(&self) -> Option<&str> {
        self.dump_to.as_deref()
    }
}
