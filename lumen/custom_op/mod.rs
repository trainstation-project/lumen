//! Custom ops: user-supplied kernels launched on a device stream, the C ABI calling registered launchers
//! without Python (`ffi`), plus their Python bindings.

#[cfg(feature = "python")]
pub(crate) mod ffi;
pub mod mps;
#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
mod tests;
