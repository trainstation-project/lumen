/// The library's name as a literal, for building other compile-time names
/// with `concat!` (which takes literals, not constants); use
/// [`LIBRARY_NAME`] otherwise.
macro_rules! library_name {
    () => {
        "lumen"
    };
}

/// An op's full name, `"lumen::<name>"`, as a compile-time literal.
macro_rules! op_name {
    ($name:literal) => {
        concat!(library_name!(), "::", $name)
    };
}

/// The library's name (exposed to Python as `lumen._C.LIBRARY_NAME`): the
/// namespace of op names, `lumen::fill_`.
pub const LIBRARY_NAME: &str = library_name!();

pub mod allocator;
pub mod device;
pub mod graph;
pub mod ops;
pub mod profiler;
#[cfg(feature = "python")]
mod python;
pub mod stream;
pub mod tensor;

pub use allocator::static_allocator::StaticAllocator;
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
