//! What MPS runs for this pass.

use super::PlaceViewsDevice;
use crate::codegen::dot_general::mps as gemm;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::place_views::PlaceViews;
use crate::graph_compiler::pipeline::Pass;

/// MPS's answers to the pass.
impl PlaceViewsDevice for Mps {
    /// Always: slices a GEMM reads in place become views before the lone layout ops left become elementwise
    /// fusions, as on Triton.
    fn place_views<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(PlaceViews {
            reads: gemm::reads_strided,
        })
    }
}
