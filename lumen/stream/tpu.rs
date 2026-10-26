//! The TPU stream, synchronized through the TPU allocator.

use super::Stream;

/// lumen's stream on TPU device `index`.
pub struct TpuStream {
    pub index: usize,
}

impl Stream for TpuStream {
    /// Blocks until the TPU's queued work finishes, panicking if the wait fails.
    fn synchronize(&self) {
        crate::allocator::tpu::wait(self.index).unwrap_or_else(|e| panic!("{e}"))
    }
}
