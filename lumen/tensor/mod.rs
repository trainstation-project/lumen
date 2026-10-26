//! Tensors: dtypes, scalars, storage, parameters, and the Python and DLPack bindings.
#[cfg(feature = "python")]
pub(crate) mod dlpack;
pub mod dtype;
pub(crate) mod parameter;
#[cfg(feature = "python")]
pub(crate) mod python;
pub mod scalar;
pub mod storage;
#[allow(clippy::module_inception)]
mod tensor;
pub mod tensor_options;
#[cfg(test)]
mod tests;

pub use tensor::Tensor;
pub(crate) use tensor::{contiguous_strides, for_each_index};
