pub mod allocator;
pub mod device;

pub use allocator::caching::CachingAllocator;
pub use allocator::cuda::CudaPolicy;
pub use allocator::mps::MpsPolicy;
pub use allocator::traits::{CachePolicy, DeviceBackend};
pub use allocator::{Allocator, CpuAllocator, DataPtr};
pub use device::Device;
