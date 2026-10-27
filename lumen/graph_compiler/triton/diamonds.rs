//! Fuses normalization diamonds (row reductions broadcast back over their rows) into Triton row kernels.

use super::TritonModule;
use crate::codegen::triton::{fusible, row_fits};
use crate::graph_compiler::common::diamonds::{diamonds, fused};
use crate::graph_compiler::pass::CompilerPass;

/// Replaces normalization diamonds the Triton row kernel supports with `row` fusions.
pub(super) struct Diamonds;

impl CompilerPass<TritonModule<'_>> for Diamonds {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "normalization_diamonds"
    }

    /// Rewrites each fitting row into a `row` fusion.
    fn apply(&mut self, m: &mut TritonModule<'_>) -> Result<(), String> {
        m.graph = fused(&m.graph, &diamonds(&m.graph, fusible), fusible, row_fits);
        Ok(())
    }
}
