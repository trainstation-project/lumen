//! How MPS sizes the column sums riding along its normalization rows as per-group partial sums, for the
//! shared `Diamonds` pass.

use crate::codegen::reduction::mps::REDUCE_THREADS;
use crate::graph_compiler::common::diamonds::Partials;
pub(super) use crate::graph_compiler::common::diamonds::Row;

/// How column sums ride along MPS row kernels: at most 16 `REDUCE_THREADS`-wide column chunks a row,
/// split into 16 to 64 row groups.
pub(super) const PARTIALS: Partials = Partials {
    columns: 16 * REDUCE_THREADS,
    groups: 64,
    fewest_groups: 16,
};
