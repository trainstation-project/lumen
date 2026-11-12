//! What MPS runs for this pass.

use super::UpdatesDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::updates::Updates;
use crate::graph_compiler::pipeline::{Pass, when};

/// MPS's answers to the pass.
impl UpdatesDevice for Mps {
    /// With `fuse`.
    fn updates<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        let updates = Updates {
            least: 2,
            most: MOST_UPDATES,
        };

        when(config.fuse, updates)
    }
}

/// Most updates one `updates` kernel takes: Metal binds 31 buffers, two of them the operand and output.
const MOST_UPDATES: usize = 29;
