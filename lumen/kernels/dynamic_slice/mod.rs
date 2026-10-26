//! `dynamic_slice` and `dynamic_update_slice` kernels for the MPS and TPU backends.
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
