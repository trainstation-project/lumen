//! What Triton runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::updates::Updates;
use crate::graph_compiler::pipeline::{Pass, when};

/// With `fuse`.
impl DevicePass<Triton> for Updates {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
        let updates = Updates {
            least: 2,
            most: usize::MAX,
        };

        when(config.fuse, updates)
    }
}
