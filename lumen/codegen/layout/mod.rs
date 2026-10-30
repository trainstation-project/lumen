//! Layout code generation and kernels: copies between strided and contiguous forms, reshapes of tiled
//! tensors, and transposes fused with elementwise ops.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
