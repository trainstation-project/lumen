//! What MPS runs for this pass.

use super::TakeAlongAxisDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::take_along_axis::TakeAlongAxis;
use crate::graph_compiler::pipeline::Pass;

/// MPS's answers to the pass.
impl TakeAlongAxisDevice for Mps {
    /// Always.
    fn take_along_axis<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(TakeAlongAxis)
    }
}
