use super::{Kernel, MpsModule, request};
use crate::graph::Var;
use crate::graph_compiler::attention;
use crate::graph_compiler::pass::CompilerPass;

pub(super) struct FlashAttention;

impl<'a> CompilerPass<MpsModule<'a>> for FlashAttention {
    fn name(&self) -> &'static str {
        "flash_attention"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let scalars: Vec<Var> = m.scalars.iter().map(|&i| m.graph.inputs()[i]).collect();
        let epilogues = m.options.config.contraction_epilogues;
        let kernels = &mut m.kernels;
        m.graph = attention::fuse(
            &m.graph,
            epilogues,
            &scalars,
            |_| true,
            |_| request(kernels, Kernel::Attention),
        );
        Ok(())
    }
}
