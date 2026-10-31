//! Store code generation: the addressing of each backend's loads and stores through a tensor's strides,
//! shared by the kernels of every op.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod triton;
