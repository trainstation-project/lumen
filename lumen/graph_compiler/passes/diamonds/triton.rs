//! What Triton runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::diamonds::Diamonds;
use crate::graph_compiler::passes::fusion::triton as fusion;
use crate::graph_compiler::pipeline::{Pass, when};

/// With `fuse` and `normalization_diamonds`: normalization rows for the fusion, as MPS finds them.
impl DevicePass<Triton> for Diamonds {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
        let diamonds = Diamonds {
            fusible: fusion::fusible,
            partials: crate::codegen::row::triton::ROW_PARTIALS,
        };

        when(config.fuse && config.normalization_diamonds, diamonds)
    }
}
