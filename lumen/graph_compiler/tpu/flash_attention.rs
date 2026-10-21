use super::TpuModule;
use crate::graph_compiler::attention;
use crate::graph_compiler::pass::CompilerPass;

pub(super) struct FlashAttention;

impl CompilerPass<TpuModule<'_>> for FlashAttention {
    fn name(&self) -> &'static str {
        "flash_attention"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = attention::fuse(&m.graph, false, &[], |_| "flash_attention".into());
        Ok(())
    }
}
