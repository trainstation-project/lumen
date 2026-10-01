//! Layout primitives (a plan's reshape copies, broadcast_in_dim,
//! transpose): their kernels per backend, `mps.metal` and `mps.rs` for
//! graph plans on MPS (see [`crate::ops::mps`]).

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
