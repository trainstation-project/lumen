//! Which nodes each backend's kernels fuse, and how its fused bodies are named and read: what codegen and the graph
//! compiler's passes (`passes/fusion`, `place_concats`, `horizontal`) both ask.

pub(crate) mod cuda;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod triton;
