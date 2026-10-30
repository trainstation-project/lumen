//! Gather kernels: the TPU gather of rows at indices and the Metal gather of a strided view into a
//! contiguous buffer, each backend's beside the others.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
