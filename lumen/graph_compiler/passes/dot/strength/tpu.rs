//! What TPU runs for this pass.

use super::DotStrengthDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::dot::strength::DotStrength;
use crate::graph_compiler::pipeline::Pass;

/// TPU's answers to the pass.
impl DotStrengthDevice for Tpu {
    /// Always.
    fn dot_strength<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(DotStrength)
    }
}
