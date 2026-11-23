//! The traced dataflow graph: the graph and its primitives (`ir`), how each primitive's result names and sharding follow
//! from its operands' (`propagation`), and custom ops' bridge back to their Python functions (`custom`). Each binds
//! itself to Python (its `lumen.graph.<folder>._C`); the tracer (`tracing/`) is Python.

pub(crate) mod custom;
pub mod ir;
pub mod propagation;

pub use ir::*;
