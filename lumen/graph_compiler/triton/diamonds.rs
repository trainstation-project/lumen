//! Fuses normalization diamonds (row reductions broadcast back over their rows) into Triton row kernels,
//! and a matmul whose product a row normalizes into its GEMM's epilogue.

use super::TritonModule;
use crate::codegen::triton::{dot_row_fits, fusible, joins_rows, partial_tile, row_fits};
use crate::graph_compiler::common::diamonds::{diamonds, fused};
use crate::graph_compiler::common::dot_rows::{dot_partials, dot_rows};
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

/// Folds each matmul whose whole output rows a `row` fusion reads into a `dot_row` fusion; when the rows
/// are too wide for one tile, its GEMM writes the row's first reductions' partials instead
/// (`dot_partials`).
pub(super) struct DotRows;

impl CompilerPass<TritonModule<'_>> for DotRows {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "dot_rows"
    }

    /// Rewrites each fitting dot and row pair into a `dot_row` fusion.
    fn apply(&mut self, m: &mut TritonModule<'_>) -> Result<(), String> {
        m.graph = dot_rows(&m.graph, joins_rows, dot_row_fits, true);
        m.graph = dot_partials(&m.graph, joins_rows, partial_tile, true);
        Ok(())
    }
}
