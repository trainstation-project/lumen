//! dot_general: their kernels per backend, `mps/` (`kernels.metal`, `mod.rs`) for
//! graph plans on MPS (see [`crate::ops::mps`]) and `tpu.rs` (the spec of a
//! Pallas kernel, `tpu.py`) for plans on TPU (see [`crate::ops::tpu`]).

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
