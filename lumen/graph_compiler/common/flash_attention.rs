use crate::graph::Var;
use crate::graph_compiler::common::attention::{self, Epilogues};
use crate::graph_compiler::pass::{CompilerPass, Module};

pub(crate) trait Attentions: Module {
    fn contraction_epilogues(&self) -> bool;
    fn epilogues(&self) -> Epilogues;
    fn scalars(&self) -> Vec<Var>;
    fn kernel(&mut self) -> String;
    fn runtime_offsets(&self) -> bool;
}

pub(crate) struct FlashAttention;

impl<M: Attentions> CompilerPass<M> for FlashAttention {
    fn name(&self) -> &'static str {
        "flash_attention"
    }

    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let epilogues = m.contraction_epilogues().then(|| m.epilogues());
        let (graph, scalars) = (m.graph().clone(), m.scalars());
        let runtime_offsets = m.runtime_offsets();
        *m.graph_mut() =
            attention::fuse(&graph, epilogues, &scalars, runtime_offsets, |_| m.kernel());
        Ok(())
    }
}
