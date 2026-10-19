use super::Stream;

/// A TPU's stream: its programs run in order on its memory, asynchronously
/// (each reads the one before's results, `lumen/ops/tpu`); every read of it
/// (`copy_to_host`) waits for them itself.
pub struct TpuStream {
    pub index: usize,
}

impl Stream for TpuStream {
    /// Waits for the programs run on the device so far.
    fn synchronize(&self) {
        crate::allocator::tpu::wait(self.index).unwrap_or_else(|e| panic!("{e}"))
    }
}
