//! TPU's dot-row fusion: a matmul fused into the normalization `row` fusion that consumes its whole output
//! rows, where the TPU row kernel reads them whole.

use super::dot_rows;
use crate::codegen::row::tpu::{Kind, kinds};
use crate::graph::Graph;
use crate::graph_compiler::pass::CompilerPass;
use crate::graph_compiler::tpu::TpuModule;

/// Whether the TPU row kernel reads input `at` of row body `body` as full rows.
fn full(body: &Graph, at: usize) -> bool {
    kinds(body).is_some_and(|kinds| kinds[body.inputs()[at]] == Kind::Full)
}

/// Fuses dots into the row fusions that read them when contraction epilogues are enabled.
pub(crate) struct DotRows;

impl CompilerPass<TpuModule<'_>> for DotRows {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "dot_rows"
    }

    /// Fuses the module's dots into the row fusions reading their whole rows.
    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        if m.options.config.contraction_epilogues {
            m.graph = dot_rows(&m.graph, full, |_| true, false);
        }

        Ok(())
    }
}
