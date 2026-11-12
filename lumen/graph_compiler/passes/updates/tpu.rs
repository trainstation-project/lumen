//! What TPU runs for this pass.

use super::UpdatesDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::updates::Updates;
use crate::graph_compiler::pipeline::Pass;

/// TPU's answers to the pass.
impl UpdatesDevice for Tpu {
    /// Always.
    fn updates<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Updates {
            least: 1,
            most: usize::MAX,
        })
    }
}
