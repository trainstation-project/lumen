#[cfg(feature = "python")]
pub(crate) mod dlpack;
pub mod dtype;
pub(crate) mod parameter;
#[cfg(feature = "python")]
pub(crate) mod python;
pub mod scalar;
pub mod storage;
// `Tensor` lives in `tensor/tensor.rs`, re-exported below so its path is
// `lumen::tensor::Tensor`; the module itself stays private.
#[allow(clippy::module_inception)]
mod tensor;
pub mod tensor_options;
#[cfg(test)]
mod tests;

pub use tensor::Tensor;
pub(crate) use tensor::{contiguous_strides, flat_offset, for_each_index};
