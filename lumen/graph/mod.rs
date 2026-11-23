//! The traced dataflow graph: the graph and its primitives (`ir`), how each primitive's result names and sharding follow
//! from its operands' (`propagation`), and the Python bindings the tracer (`tracing/`, Python) builds it through.

pub(crate) mod custom;
pub mod ir;
pub mod propagation;
#[cfg(feature = "python")]
pub(crate) mod python;

pub use ir::*;
