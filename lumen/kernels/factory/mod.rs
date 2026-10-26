//! Kernels for factory primitives (`Full`, `Iota`, `RandomBits`) that create tensors from nothing.
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
