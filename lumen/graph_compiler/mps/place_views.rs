//! Pass that turns slices read only by GEMMs into strided views instead of copies.

use super::{MpsModule, codegen};
use crate::codegen::dot_general::mps::reads_strided;
use crate::graph::{Graph, Primitive};
use crate::graph_compiler::common::attention;
use crate::graph_compiler::common::place_views::{self, place_views};
use crate::graph_compiler::pass::CompilerPass;

/// Compiler pass marking slices as views where their readers allow it.
pub(super) struct PlaceViews;

impl<'a> CompilerPass<MpsModule<'a>> for PlaceViews {
    fn name(&self) -> &'static str {
        "place_views"
    }

    /// Marks each slice read only by GEMMs that stride through it a view (`common::place_views`).
    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        place_views(&mut m.graph, reads_strided, codegen::gemm_dot);
        Ok(())
    }
}

/// Whether fusion input `k` may be bound as a strided view: it is not attention and no dot reads it.
pub(crate) fn loads_strided(body: &Graph, k: usize) -> bool {
    let Some(&v) = body.inputs().get(k) else {
        return false;
    };

    let template = |p: &Primitive| matches!(p, Primitive::DotGeneral { .. });

    attention::of_body(body).is_none()
        && attention::backward_of_body(body).is_none()
        && !body
            .nodes()
            .iter()
            .any(|n| template(&n.primitive) && n.inputs.contains(&v))
}

/// Whether an attention fusion reads its inputs at `strides`: any, with a contiguous last dim (its
/// kernel is generated for the layout when it runs).
pub(crate) fn attention_reads_strided(body: &Graph, strides: &[usize]) -> bool {
    strides.last().is_none_or(|&s| s == 1)
        && (attention::of_body(body).is_some() || attention::backward_of_body(body).is_some())
}

/// Whether GEMM fusion input `k` is read only by the dot, as one operand, in a layout MPS's matmul
/// kernels stride through: an attention's dots are its kernel's, not a GEMM's.
pub(crate) fn fusion_reads_strided(body: &Graph, k: usize, strides: &[usize]) -> bool {
    attention::of_body(body).is_none()
        && attention::backward_of_body(body).is_none()
        && place_views::fusion_reads_strided(body, k, strides, reads_strided, codegen::gemm_dot)
}
