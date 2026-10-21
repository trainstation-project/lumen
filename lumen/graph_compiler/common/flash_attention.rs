use crate::graph::Var;
use crate::graph_compiler::common::attention::{self, Epilogues};
use crate::graph_compiler::pass::{CompilerPass, Module};

pub(crate) trait Attentions: Module {
    fn epilogues(&self) -> Epilogues;

    fn scalars(&self) -> Vec<Var>;

    fn kernel(&mut self) -> String;
}

pub(crate) struct FlashAttention;

impl<M: Attentions> CompilerPass<M> for FlashAttention {
    fn name(&self) -> &'static str {
        "flash_attention"
    }

    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let epilogues = m
            .options()
            .config
            .contraction_epilogues
            .then(|| m.epilogues());
        let (graph, scalars) = (m.graph().clone(), m.scalars());
        *m.graph_mut() = attention::fuse(&graph, epilogues, &scalars, |_| m.kernel());
        Ok(())
    }
}
