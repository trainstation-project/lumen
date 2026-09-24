pub mod allocator;
pub mod device;

pub use allocator::cuda::{CachingAllocator, DeviceBackend};
pub use allocator::{Allocator, CpuAllocator, DataPtr};
pub use device::Device;
