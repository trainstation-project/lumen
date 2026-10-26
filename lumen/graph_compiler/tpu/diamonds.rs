//! Fuses normalization diamonds (row reductions broadcast back over their rows) into `row` kernels.
use super::TpuModule;
use crate::codegen::tpu::row::{fits, fusible};
use crate::graph_compiler::common::diamonds::{diamonds, fused};
use crate::graph_compiler::pass::CompilerPass;

/// Replaces normalization diamonds the TPU row kernel supports with `row` fusions.
pub(super) struct Diamonds;

impl CompilerPass<TpuModule<'_>> for Diamonds {
    fn name(&self) -> &'static str {
        "normalization_diamonds"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = fused(&m.graph, &diamonds(&m.graph, fusible), fusible, fits);
        Ok(())
    }
}
