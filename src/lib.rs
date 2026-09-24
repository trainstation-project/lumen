pub mod allocator;
pub mod device;

pub use allocator::caching::{CacheConfig, CachingAllocator, DeviceBackend};
pub use allocator::{Allocator, CpuAllocator, DataPtr};
pub use device::Device;
