use super::{CompilerConfig, device_config};

device_config! {
    CpuConfig, PyCpuConfig, "CpuConfig", cpu, crate::Device::Cpu(0), {
    }
}

impl CompilerConfig for CpuConfig {
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
