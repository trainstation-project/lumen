//! cumsum: its kernels per backend, `mps/` (`kernels.metal`, `mod.rs`) for
//! graph plans on MPS (see [`crate::ops::mps`]).

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
