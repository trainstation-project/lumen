//! What TPU runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::diamonds::FusedDiamonds;
use crate::graph_compiler::pipeline::{Pass, when};

/// With `normalization_diamonds`.
impl DevicePass<Tpu> for FusedDiamonds {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Tpu>> {
        let diamonds = FusedDiamonds {
            fusible: crate::codegen::row::tpu::fusible,
            fits: crate::codegen::row::tpu::fits,
        };

        when(config.normalization_diamonds, diamonds)
    }
}
