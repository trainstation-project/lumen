//! Row-fusion kernels, which hold whole rows in fast memory.
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
