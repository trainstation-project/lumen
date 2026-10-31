//! Kernels for factory primitives (`Full`, `Iota`) that create tensors from nothing; `RandomBits` is
//! codegen/random's.
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
