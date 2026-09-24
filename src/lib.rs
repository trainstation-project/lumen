pub mod allocator;
pub mod device;

pub use allocator::caching::{CachePolicy, CachingAllocator, DeviceBackend};
pub use allocator::cuda::CudaPolicy;
pub use allocator::mps::MpsPolicy;
pub use allocator::{Allocator, CpuAllocator, DataPtr};
pub use device::Device;
