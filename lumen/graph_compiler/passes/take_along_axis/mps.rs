//! What MPS runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::take_along_axis::TakeAlongAxis;
use crate::graph_compiler::pipeline::Pass;

/// Always.
impl DevicePass<Mps> for TakeAlongAxis {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Mps>> {
        Box::new(TakeAlongAxis)
    }
}
