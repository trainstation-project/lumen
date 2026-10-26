//! The meta-device stream: meta tensors have no data, so no work is ever queued.

use super::Stream;

/// lumen's stream on the meta device.
pub struct MetaStream;

impl Stream for MetaStream {
    /// A no-op: the meta device runs no kernels.
    fn synchronize(&self) {}
}
