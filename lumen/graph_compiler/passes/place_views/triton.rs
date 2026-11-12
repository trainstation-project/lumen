//! What Triton runs for this pass.

use super::PlaceViewsDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::place_views::PlaceViews;
use crate::graph_compiler::pipeline::Pass;

/// Triton's answers to the pass.
impl PlaceViewsDevice for Triton {
    /// Always: Triton's GEMM reads its operands at any strides.
    fn place_views<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(PlaceViews {
            reads: |_, _, _, _| true,
        })
    }
}
