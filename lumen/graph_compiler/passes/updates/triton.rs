//! What Triton runs for this pass.

use super::UpdatesDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::updates::Updates;
use crate::graph_compiler::pipeline::{Pass, when};

/// Triton's answers to the pass.
impl UpdatesDevice for Triton {
    /// With `fuse`.
    fn updates<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        let updates = Updates {
            least: 2,
            most: usize::MAX,
        };

        when(config.fuse, updates)
    }
}
