//! The traced dataflow graph: the graph and its primitives (`ir`), how each primitive's result names and sharding follow
//! from its operands' (`propagation`), custom ops' bridge back to their Python functions (`custom`), and the tracer's
//! native walk over a compiled call's arguments (`tracing`). Each binds itself to Python (its `lumen.graph.<folder>._C`);
//! the tracer itself (`tracing/tracer.py`) is Python.

pub(crate) mod custom;
pub mod ir;
pub mod propagation;
pub(crate) mod tracing;

pub use ir::*;
