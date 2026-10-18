//! Device-independent compiler passes: rewrites of the graph alone, each a
//! [`CompilerPass`](super::pass::CompilerPass) of any module, which any
//! device's pipeline may run (the frontend's, the CPU's, the MPS's). A
//! device's own passes (its kernels' layouts, tiles, fusion) are in its
//! folder (`graph_compiler/mps`).

pub(crate) mod cse;
pub(crate) mod simplify;

// Passes only the MPS pipeline runs so far: unused where it is not built.
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod chunked_cross_entropy;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod clamp_indices;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod dot_strength;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod gated_backward;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod linear_cross_entropy;
