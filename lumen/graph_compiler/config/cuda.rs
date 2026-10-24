use super::{CompilerConfig, device_config};

device_config! {
    CudaConfig, PyCudaConfig, "CudaConfig", cuda {
        merge_dots, set_merge_dots: bool = true;
        flash_attention, set_flash_attention: bool = true;
        fused_linear_cross_entropy_chunk_size, set_fused_linear_cross_entropy_chunk_size: Option<usize> = None;
        use_triton_backend, set_use_triton_backend: bool = false;
        autotune, set_autotune: bool = true;
        memory_limit, set_memory_limit: usize = 0;
        dump_to, set_dump_to: Option<String> = None;
    }
}

impl CompilerConfig for CudaConfig {
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
