//! Fuses a matmul into the normalization `row` fusion that consumes its whole output rows.
use super::super::TpuModule;
use crate::codegen::tpu::row::{Kind, kinds};
use crate::graph::Graph;
use crate::graph_compiler::common::dot_rows::dot_rows;
use crate::graph_compiler::pass::CompilerPass;

/// Whether the TPU row kernel reads input `at` of row body `body` as full rows.
fn full(body: &Graph, at: usize) -> bool {
    kinds(body).is_some_and(|kinds| kinds[body.inputs()[at]] == Kind::Full)
}

/// Fuses dots into the row fusions that read them when contraction epilogues are enabled.
pub(in crate::graph_compiler::tpu) struct DotRows;

impl CompilerPass<TpuModule<'_>> for DotRows {
    fn name(&self) -> &'static str {
        "dot_rows"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        if m.options.config.tpu.contraction_epilogues {
            m.graph = dot_rows(&m.graph, full, |_| true, false);
        }
        Ok(())
    }
}
