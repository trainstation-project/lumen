//! Custom ops: user-supplied kernels launched on a device stream, plus their Python bindings.

pub mod mps;
#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
mod tests;
