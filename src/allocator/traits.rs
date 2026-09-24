//! The two extension points ("seams") of
//! [`CachingAllocator`](super::caching::CachingAllocator), plus the size
//! constants they share.
//!
//! - [`DeviceBackend`] is the raw-memory seam: allocate/free on the device
//!   (`cudaMalloc`/`cudaFree`, shared-mode `MTLBuffer`s, or a test mock).
//!   The pooling logic never calls a device API directly.
//! - [`CachePolicy`] is the size-math seam: alignment, rounding, segment
//!   sizes, split thresholds — the parts where PyTorch's CUDA and MPS
//!   allocators differ.
//!
//! Both are implemented next to each backend (`cuda.rs`, `mps.rs`).

// ---------------- shared size constants ----------------

/// Largest "small" allocation; anything bigger goes to the large pool
/// (c10: kSmallSize, MPS: kMaxSmallAlloc).
pub const K_SMALL_SIZE: usize = 1 << 20; // 1 MiB
/// Allocations at least this big get a dedicated segment (c10/MPS: kMinLargeAlloc).
pub const K_MIN_LARGE_ALLOC: usize = 10 << 20; // 10 MiB
/// Round-up granularity for dedicated segments (c10/MPS: kRoundLarge).
pub const K_ROUND_LARGE: usize = 2 << 20; // 2 MiB

/// A dedicated segment for a `size`-byte block: `size` rounded up to
/// [`K_ROUND_LARGE`].
pub(crate) fn dedicated_segment_size(size: usize) -> usize {
    size.div_ceil(K_ROUND_LARGE) * K_ROUND_LARGE
}

// ---------------- size policy ----------------

/// The size math that differs between PyTorch's CUDA and MPS allocators
/// ([`crate::CudaPolicy`], [`crate::MpsPolicy`]). Everything else — pools,
/// best-fit search, splitting, coalescing, stats, releasing segments — is
/// shared by [`CachingAllocator`](super::caching::CachingAllocator).
pub trait CachePolicy: Send + Sync + 'static {
    /// Alignment of every block, and so of every pointer handed out.
    fn alignment(&self) -> usize;
    /// Block size for a request of `nbytes` (> 0) bytes.
    fn round_size(&self, nbytes: usize) -> usize;
    /// Whether to split `remaining` bytes off a block in the small (or
    /// large) pool into a new free block.
    fn should_split(&self, small: bool, remaining: usize) -> bool;
    /// How big a segment to grab from the device for a `size`-byte block
    /// when `reserved` bytes are already held.
    fn segment_size(&self, size: usize, reserved: usize) -> usize;
    /// Panic on requests the device can never serve. Default: none.
    fn check_request(&self, _nbytes: usize) {}
}

// ---------------- backend abstraction ----------------

/// Raw device memory operations. Implemented by the CUDA backend (`cuda.rs`),
/// the Metal backend (`mps.rs`), and by mocks in tests.
///
/// # Safety
/// Implementations must return pointers to at least `nbytes` of device
/// memory that remain valid until `device_free`.
pub trait DeviceBackend: Send + Sync + 'static {
    /// Allocate `nbytes` on the device; null on failure.
    ///
    /// # Safety
    /// The returned pointer must denote at least `nbytes` of device memory
    /// that stays valid until passed to `device_free`.
    unsafe fn device_alloc(&self, nbytes: usize) -> *mut u8;
    /// Free a pointer from `device_alloc`.
    ///
    /// # Safety
    /// `ptr` must come from `device_alloc` and not have been freed already.
    unsafe fn device_free(&self, ptr: *mut u8);
}
