//! The CPU stream: CPU work runs synchronously, so there is nothing to wait for.

use super::Stream;

/// lumen's stream on the CPU.
pub struct CpuStream;

impl Stream for CpuStream {
    /// A no-op: CPU work has finished by the time it returns.
    fn synchronize(&self) {}
}
