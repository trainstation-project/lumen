use super::Stream;

/// A TPU's stream. Everything the TPU backend runs so far (the allocator's
/// reads of device memory) waits for its own work before returning: nothing
/// is in flight to wait for.
pub struct TpuStream;

impl Stream for TpuStream {
    fn synchronize(&self) {}
}
