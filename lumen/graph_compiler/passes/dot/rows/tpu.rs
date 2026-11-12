//! TPU's dot-row fusion: a matmul fused into the normalization `row` fusion that consumes its whole output
//! rows, where the TPU row kernel reads them whole.

use super::dot_rows;
use crate::codegen::row::tpu::{Kind, kinds};
use crate::graph::Graph;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::CompilerPass;
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::pipeline::{Pass, when};

/// Whether the TPU row kernel reads input `at` of row body `body` as full rows.
fn full(body: &Graph, at: usize) -> bool {
    kinds(body).is_some_and(|kinds| kinds[body.inputs()[at]] == Kind::Full)
}

/// Fuses dots into the row fusions that read them when contraction epilogues are enabled.
pub(crate) struct DotRows;

impl CompilerPass<DeviceModule<'_, Tpu>> for DotRows {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "dot_rows"
    }

    /// Fuses the module's dots into the row fusions reading their whole rows.
    fn apply(&mut self, m: &mut DeviceModule<'_, Tpu>) -> Result<(), String> {
        if m.options.config.contraction_epilogues {
            m.graph = dot_rows(&m.graph, full, |_| true, false);
        }

        Ok(())
    }
}

/// With `normalization_diamonds`.
impl DevicePass<Tpu> for DotRows {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Tpu>> {
        when(config.normalization_diamonds, DotRows)
    }
}
