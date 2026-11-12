//! What Triton runs for this pass.

use super::DotStrengthDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::dot::strength::DotStrength;
use crate::graph_compiler::pipeline::Pass;

/// Triton's answers to the pass.
impl DotStrengthDevice for Triton {
    /// Always: batched inner products (attention backward's delta, read through transposes) run as elementwise
    /// fusions.
    fn dot_strength<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(DotStrength)
    }
}
