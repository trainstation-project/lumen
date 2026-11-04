//! Emitters: each backend's statements computing a fused body's values at an index, the core of every
//! generated kernel.

pub(crate) mod cuda;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod triton;

use crate::graph::{Graph, Var};

/// A backend's emitter: builds the statements computing a fused body's values, memoizing each
/// `(value, index)` so shared subexpressions are emitted once.
pub(crate) trait Emit<'a> {
    /// An emitter over `body` with nothing emitted, the inputs flagged in `by_value` read as scalars.
    fn new(body: &'a Graph, by_value: &'a [bool]) -> Self;
    /// The local holding `v` at flat index `idx`, emitting the statements that compute it.
    fn value(&mut self, v: Var, idx: String) -> String;
}
