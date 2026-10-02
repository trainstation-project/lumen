//! Graph compilers, one per device: each rewrites a [`Graph`] for its
//! device's kernels (fusing primitives into generated kernels, XLA-style),
//! then plans its memory into a [`Plan`]. Devices without one run the
//! graph's primitives as they are.
//!
//! - [`mps`]: loop fusion into generated Metal kernels.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;

use crate::Device;
use crate::graph::{Graph, Plan};

/// `graph` compiled for `device`: a plan that runs on any device, its
/// fusions in `device`'s kernels where the device has a compiler.
pub fn compile(graph: &Graph, device: Device) -> Result<Plan, String> {
    match device {
        #[cfg(lumen_mps_linked)]
        Device::Mps => mps::compile(graph),
        _ => Ok(Plan::compile(graph)),
    }
}
