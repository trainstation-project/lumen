macro_rules! library_name {
    () => {
        "lumen"
    };
}

macro_rules! op_name {
    ($name:literal) => {
        concat!(library_name!(), "::", $name)
    };
}

pub const LIBRARY_NAME: &str = library_name!();

pub mod allocator;
pub(crate) mod codegen;
pub mod custom_op;
pub mod device;
pub mod graph;
pub mod graph_compiler;
pub mod ops;
pub mod pallas;
pub mod profiler;
#[cfg(feature = "python")]
mod python;
pub mod safetensors;
pub mod stream;
pub mod tensor;

pub use allocator::{Allocator, CpuAllocator, DataPtr};
pub use device::Device;
pub use tensor::Tensor;
pub use tensor::dtype::{DType, Element};
pub use tensor::scalar::Scalar;
pub use tensor::storage::Storage;
pub use tensor::tensor_options::TensorOptions;

#[cfg(test)]
mod library_name_tests {
    #[test]
    fn op_names_use_the_library_name() {
        assert_eq!(super::LIBRARY_NAME, "lumen");
        assert_eq!(op_name!("fill_"), op_name!("fill_"));
    }
}
