pub const K_SMALL_SIZE: usize = 1 << 20; // 1 MiB
pub const K_MIN_LARGE_ALLOC: usize = 10 << 20; // 10 MiB
pub const K_ROUND_LARGE: usize = 2 << 20; // 2 MiB

pub(crate) fn dedicated_segment_size(size: usize) -> usize {
    size.div_ceil(K_ROUND_LARGE) * K_ROUND_LARGE
}

pub trait CachePolicy: Send + Sync + 'static {
    fn alignment(&self) -> usize;
    fn round_size(&self, nbytes: usize) -> usize;
    fn should_split(&self, small: bool, remaining: usize) -> bool;
    fn segment_size(&self, size: usize, reserved: usize) -> usize;
    fn check_request(&self, _nbytes: usize) {}
}
