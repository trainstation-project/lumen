//! What TPU runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::dot::strength::DotStrength;
use crate::graph_compiler::pipeline::Pass;

/// Always.
impl DevicePass<Tpu> for DotStrength {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Tpu>> {
        Box::new(DotStrength)
    }
}
