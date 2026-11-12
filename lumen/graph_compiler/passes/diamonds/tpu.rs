//! What TPU runs for this pass.

use super::DiamondsDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::diamonds::FusedDiamonds;
use crate::graph_compiler::pipeline::{Pass, when};

/// TPU's answers to the pass.
impl DiamondsDevice for Tpu {
    /// With `normalization_diamonds`.
    fn fused_diamonds<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        let diamonds = FusedDiamonds {
            fusible: crate::codegen::row::tpu::fusible,
            fits: crate::codegen::row::tpu::fits,
        };

        when(config.normalization_diamonds, diamonds)
    }
}
