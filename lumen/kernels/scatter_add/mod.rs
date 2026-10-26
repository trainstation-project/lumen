//! `scatter_add` kernels: adds update rows into an operand at indexed positions along an axis.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
