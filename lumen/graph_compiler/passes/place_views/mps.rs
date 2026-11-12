//! What MPS runs for this pass.

use crate::codegen::dot_general::mps as gemm;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::place_views::PlaceViews;
use crate::graph_compiler::pipeline::Pass;

/// Always: slices a GEMM reads in place become views before the lone layout ops left become elementwise
/// fusions, as on Triton.
impl DevicePass<Mps> for PlaceViews {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Mps>> {
        Box::new(PlaceViews {
            reads: gemm::reads_strided,
        })
    }
}
