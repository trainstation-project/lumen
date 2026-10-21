mod dot_epilogues;
mod emit;
mod groups;
mod horizontal;

pub(super) use dot_epilogues::FuseDots;
pub(super) use horizontal::HorizontalFusion;

use super::TpuModule;
use crate::graph_compiler::pass::CompilerPass;

pub(super) struct LoopFusion;

impl CompilerPass<TpuModule<'_>> for LoopFusion {
    fn name(&self) -> &'static str {
        "loop_fusion"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = emit::loop_fusion(&m.graph);
        Ok(())
    }
}
