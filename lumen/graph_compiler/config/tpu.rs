use super::{CompilerConfig, device_config};

device_config! {
    TpuConfig, PyTpuConfig, "TpuConfig", tpu, crate::Device::Tpu(0), {
        normalization_diamonds, set_normalization_diamonds: bool = true;
        contraction_epilogues, set_contraction_epilogues: bool = true;
        horizontal_fusion, set_horizontal_fusion: bool = true;
        safe_kernels, set_safe_kernels: bool = false;
    }
}

impl CompilerConfig for TpuConfig {
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
