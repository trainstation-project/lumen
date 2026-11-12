//! What Triton runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::place_views::PlaceViews;
use crate::graph_compiler::pipeline::Pass;

/// Always: Triton's GEMM reads its operands at any strides.
impl DevicePass<Triton> for PlaceViews {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
        Box::new(PlaceViews {
            reads: |_, _, _, _| true,
        })
    }
}
