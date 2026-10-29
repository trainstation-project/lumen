//! Finds normalization diamonds (row reductions broadcast back over their rows) for the shared loop
//! fusion to fuse into Triton row kernels, as MPS does.

use super::TritonModule;
use crate::codegen::triton::{ROW_PARTIALS, fusible};
use crate::graph_compiler::common::diamonds::{diamonds, partials};
use crate::graph_compiler::pass::CompilerPass;

/// Finds each normalization row, with column sums over the same rows riding along as per-group
/// partials, for loop fusion.
pub(super) struct Diamonds;

impl CompilerPass<TritonModule<'_>> for Diamonds {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "normalization_diamonds"
    }

    /// Stores the rows found and the graph rewritten for their partial sums.
    fn apply(&mut self, m: &mut TritonModule<'_>) -> Result<(), String> {
        (m.graph, m.rows) = partials(&m.graph, diamonds(&m.graph, fusible), fusible, ROW_PARTIALS);
        Ok(())
    }
}
