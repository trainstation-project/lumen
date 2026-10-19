//! reduce_sum and reduce_max: their kernels per backend, `mps/` (`kernels.metal`, `mod.rs`) for
//! graph plans on MPS (see [`crate::ops::mps`]), `tpu/` (a sum over the last dimension, in
//! lumen's Pallas) for TPU plans (see [`crate::ops::tpu`]).

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
