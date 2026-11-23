//! The graph itself: typed values (`types`), nodes applying primitives with their scopes and source lines (`node`), the
//! graph built from them (`graph`), pruning (`prune`), in-place placement views (`placements`) and precision warnings
//! (`precision`).

#[cfg(feature = "python")]
pub(crate) mod convert;
mod graph;
mod node;
mod placements;
mod precision;
pub mod primitive;
mod prune;
#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
pub(crate) mod tests;
mod types;

pub use graph::Graph;
#[cfg_attr(not(feature = "python"), allow(unused_imports))]
pub(crate) use node::intern_scope;
pub use node::{Lines, Node, Range, Scope};
pub(crate) use node::{intern, intern_lines};
pub use placements::View;
pub use primitive::{
    BinaryOp, FUSION_SEPARATOR, NEURAL_ENGINE, Primitive, Reduction, UPDATES, UnaryOp,
};
pub use types::{TensorType, Var};
