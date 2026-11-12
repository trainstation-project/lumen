//! What MPS runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::dot::strength::DotStrength;
use crate::graph_compiler::pipeline::Pass;

/// Always: batched inner products (attention backward's delta, read through transposes) run as elementwise
/// fusions.
impl DevicePass<Mps> for DotStrength {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Mps>> {
        Box::new(DotStrength)
    }
}
