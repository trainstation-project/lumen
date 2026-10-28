//! Pass that finds normalization diamonds (row reductions feeding back into their rows) and attaches
//! column sums over the same rows to them as per-group partial sums.

use super::MpsModule;
use super::fusion::fusible;
use crate::graph::Graph;
pub(super) use crate::graph_compiler::common::diamonds::Row;
use crate::graph_compiler::common::diamonds::{Partials, partials};
use crate::graph_compiler::pass::CompilerPass;
use crate::kernels::reduction::mps::REDUCE_THREADS;

/// How column sums ride along MPS row kernels: at most 16 `REDUCE_THREADS`-wide column chunks a row,
/// split into 16 to 64 row groups.
const PARTIALS: Partials = Partials {
    columns: 16 * REDUCE_THREADS,
    groups: 64,
    fewest_groups: 16,
};

/// Normalization rows of `graph` whose nodes the MPS fusion can merge.
pub(super) fn diamonds(graph: &Graph) -> Vec<Row> {
    crate::graph_compiler::common::diamonds::diamonds(graph, fusible)
}

/// Compiler pass finding normalization rows and attaching column partial sums to them.
pub(super) struct Diamonds;

impl<'a> CompilerPass<MpsModule<'a>> for Diamonds {
    fn name(&self) -> &'static str {
        "normalization_diamonds"
    }

    /// Stores the rows found and the graph rewritten for their partial sums.
    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        (m.graph, m.rows) = partials(&m.graph, diamonds(&m.graph), fusible, PARTIALS);
        Ok(())
    }
}
