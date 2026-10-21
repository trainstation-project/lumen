use super::TpuModule;
use crate::graph_compiler::attention;
use crate::graph_compiler::pass::CompilerPass;

pub(super) struct FlashAttention;

impl CompilerPass<TpuModule<'_>> for FlashAttention {
    fn name(&self) -> &'static str {
        "flash_attention"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        let scalars = attention::Epilogues {
            leaves: |graph, v| graph.type_of(v).shape.is_empty(),
            reshaped: true,
        };
        let epilogues = m.options.config.contraction_epilogues.then_some(scalars);
        m.graph = attention::fuse(&m.graph, epilogues, &[], |_| "flash_attention".into());
        Ok(())
    }
}
