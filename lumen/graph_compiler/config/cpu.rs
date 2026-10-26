//! Compiler configuration for CPU (and meta) devices.

use super::{CompilerConfig, device_config};

device_config! {
    CpuConfig, PyCpuConfig, "CpuConfig", cpu, crate::Device::Cpu(0), {
    }
}

/// Exposes the shared settings to the compiler.
impl CompilerConfig for CpuConfig {
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
