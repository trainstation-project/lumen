//! What Triton runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::dot::strength::DotStrength;
use crate::graph_compiler::pipeline::Pass;

/// Always: batched inner products (attention backward's delta, read through transposes) run as elementwise
/// fusions.
impl DevicePass<Triton> for DotStrength {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
        Box::new(DotStrength)
    }
}
