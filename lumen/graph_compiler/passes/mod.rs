//! Device-independent compiler passes: rewrites of the graph alone, each a
//! [`CompilerPass`](super::pass::CompilerPass) of any module, which any
//! device's pipeline may run (the frontend's, the CPU's, the MPS's). A
//! device's own passes (its kernels' layouts, tiles, fusion) are in its
//! folder (`graph_compiler/mps`).

pub(crate) mod chunked_cross_entropy;
pub(crate) mod clamp_indices;
pub(crate) mod cse;
pub(crate) mod dot_strength;
pub(crate) mod gated_backward;
pub(crate) mod linear_cross_entropy;
pub(crate) mod simplify;
