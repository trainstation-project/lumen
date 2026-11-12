//! What Triton runs for this pass.

use super::DiamondsDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::diamonds::Diamonds;
use crate::graph_compiler::passes::fusion::triton as fusion;
use crate::graph_compiler::pipeline::{Pass, when};

/// Triton's answers to the pass.
impl DiamondsDevice for Triton {
    /// With `fuse` and `normalization_diamonds`: normalization rows for the fusion, as MPS finds them.
    fn diamonds<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        let diamonds = Diamonds {
            fusible: fusion::fusible,
            partials: crate::codegen::row::triton::ROW_PARTIALS,
        };

        when(config.fuse && config.normalization_diamonds, diamonds)
    }
}
