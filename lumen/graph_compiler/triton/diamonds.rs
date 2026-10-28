//! Fuses normalization diamonds (row reductions broadcast back over their rows) into Triton row kernels,
//! and a matmul whose product a row normalizes into its GEMM's epilogue.

use super::TritonModule;
use crate::codegen::triton::{
    ROW_PARTIALS, dot_row_fits, fusible, joins_rows, partial_tile, row_fits,
};
use crate::graph::Primitive;
use crate::graph_compiler::common::diamonds::{absorbed, diamonds, fused, partials, widened};
use crate::graph_compiler::common::dot_rows::{dot_partials, dot_rows};
use crate::graph_compiler::pass::CompilerPass;

/// Replaces normalization diamonds the Triton row kernel supports with `row` fusions, column sums over
/// the same rows riding along as per-group partials.
pub(super) struct Diamonds;

impl CompilerPass<TritonModule<'_>> for Diamonds {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "normalization_diamonds"
    }

    /// Rewrites each fitting row, with its column partials and the elementwise work around it whose
    /// values other kernels read too (`widened`), into a `row` fusion.
    fn apply(&mut self, m: &mut TritonModule<'_>) -> Result<(), String> {
        let (graph, rows) = partials(&m.graph, diamonds(&m.graph, fusible), fusible, ROW_PARTIALS);
        let rows = absorbed(&graph, rows, fusible);
        // A widened row is kept only where the row kernel takes it; otherwise the row as found.
        let wide = widened(&graph, rows.clone(), fusible);
        let rows: Vec<_> = rows
            .into_iter()
            .zip(wide)
            .map(|(row, wide)| {
                let alone = fused(&graph, std::slice::from_ref(&wide), fusible, row_fits);
                let takes = alone.nodes().iter().any(
                    |n| matches!(&n.primitive, Primitive::Fusion { name, .. } if name == "row"),
                );
                if takes { wide } else { row }
            })
            .collect();
        m.graph = fused(&graph, &rows, fusible, row_fits);
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
