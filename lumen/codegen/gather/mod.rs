//! Gather kernels: the TPU gather of rows at indices. MPS and Triton gather in their loops.

#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
