//! How tensor metadata flows through each primitive, as `Primitive::infer` gives its type. `Primitive::names`
//! infers a result's axis names from its operands' (PyTorch's named-tensor name inference); the tracer calls it
//! for every node with a named operand. `Primitive::strategies` lists how a primitive runs on local shards of
//! operands placed over a device mesh (PyTorch DTensor's single-dim strategies), which `sharding.py`
//! chooses from. The graph itself never holds names or placements.
//!
//! Files: `names.rs` (names), `placement.rs` (`Placement`, `Strategy`), `rules.rs` (strategy families shared by many
//! primitives) and `strategies.rs` (strategies).

mod names;
mod placement;
mod rules;
mod strategies;

pub use names::Named;
pub use placement::{Op, Placement, Strategy};
