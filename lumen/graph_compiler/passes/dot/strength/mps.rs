//! What MPS runs for this pass.

use super::DotStrengthDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::dot::strength::DotStrength;
use crate::graph_compiler::pipeline::Pass;

/// MPS's answers to the pass.
impl DotStrengthDevice for Mps {
    /// Always: batched inner products (attention backward's delta, read through transposes) run as elementwise
    /// fusions.
    fn dot_strength<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(DotStrength)
    }
}
