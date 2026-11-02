//! Scatter-add kernels: indexed rows added into an operand, each backend's fixed kernels beside the
//! others (Triton's in `triton/kernel.py`). Nothing fuses into a scatter-add, so there is no code to
//! generate.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod triton;
