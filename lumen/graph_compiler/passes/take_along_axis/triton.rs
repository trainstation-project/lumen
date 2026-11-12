//! What Triton runs for this pass.

use super::TakeAlongAxisDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::take_along_axis::TakeAlongAxis;
use crate::graph_compiler::pipeline::Pass;

/// Triton's answers to the pass.
impl TakeAlongAxisDevice for Triton {
    /// Always.
    fn take_along_axis<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(TakeAlongAxis)
    }
}
