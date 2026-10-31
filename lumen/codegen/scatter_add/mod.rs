//! Scatter-add kernels: indexed rows added into an operand, each backend's a fixed kernel beside the
//! others (Triton's in `triton.py`). Nothing fuses into a scatter-add, so there is no code to
//! generate.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
