//! Statistics for the caching allocator.

/// A snapshot of the allocator's counters (subset of c10's `DeviceStats`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Bytes currently handed out to users (block sizes, not request sizes).
    pub allocated_bytes: usize,
    /// Peak of `allocated_bytes` since the last `reset_peak_stats`.
    pub allocated_bytes_peak: usize,
    /// Bytes currently obtained from the device (cached + in use).
    pub reserved_bytes: usize,
    /// Peak of `reserved_bytes`.
    pub reserved_bytes_peak: usize,
    /// Number of user allocations served.
    pub num_alloc: usize,
    /// Number of user blocks freed back to the cache.
    pub num_free: usize,
    /// Number of backend allocations (segments).
    pub num_device_alloc: usize,
    /// Number of backend frees.
    pub num_device_free: usize,
}
