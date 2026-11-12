//! What MPS runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::diamonds::Diamonds;
use crate::graph_compiler::passes::fusion::mps as fusion;
use crate::graph_compiler::pipeline::{Pass, when};

/// With `fuse` and `normalization_diamonds`.
impl DevicePass<Mps> for Diamonds {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Mps>> {
        let diamonds = Diamonds {
            fusible: fusion::fusible,
            partials: crate::codegen::row::mps::ROW_PARTIALS,
        };

        when(config.fuse && config.normalization_diamonds, diamonds)
    }
}
