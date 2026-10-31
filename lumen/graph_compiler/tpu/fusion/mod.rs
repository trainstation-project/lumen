//! TPU fusion passes: dot epilogues, row-wise dot fusion, elementwise/reduction fusion and horizontal fusion.
mod dot_epilogues;
mod dot_rows;
mod emit;
mod groups;
mod horizontal;

pub(super) use dot_epilogues::FuseDots;
pub(super) use dot_rows::DotRows;
pub(super) use horizontal::HorizontalFusion;

use super::TpuModule;
use crate::graph_compiler::pass::CompilerPass;

/// Groups elementwise and reduction nodes into `elementwise` and `reduction` fusions.
pub(super) struct Fusion;

impl CompilerPass<TpuModule<'_>> for Fusion {
    fn name(&self) -> &'static str {
        "fusion"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = emit::elementwise_fusion(&m.graph);
        Ok(())
    }
}
