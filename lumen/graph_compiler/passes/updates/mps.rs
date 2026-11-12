//! What MPS runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::updates::Updates;
use crate::graph_compiler::pipeline::{Pass, when};

/// Most updates one `updates` kernel takes: Metal binds 31 buffers, two of them the operand and output.
const MOST_UPDATES: usize = 29;
use crate::graph_compiler::passes::DevicePass;

/// With `fuse`.
impl DevicePass<Mps> for Updates {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Mps>> {
        let updates = Updates {
            least: 2,
            most: MOST_UPDATES,
        };

        when(config.fuse, updates)
    }
}
