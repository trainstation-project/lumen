//! Graph compilers, one per device: each rewrites a [`Graph`] for its
//! device's kernels (fusing primitives into generated kernels, XLA-style),
//! then plans its memory into a [`Plan`]. Devices without one run the
//! graph's primitives as they are.
//!
//! Every device first runs the passes that hold for any: [`cse`] (common
//! subexpression elimination).
//!
//! - [`mps`]: dot canonicalization, then loop fusion into generated Metal
//!   kernels, planned with the kernels' scratch.

mod cse;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(test)]
mod tests;

use crate::Device;
use crate::graph::{Graph, Plan, PlanOptions};

/// How to compile a graph.
#[derive(Debug, Clone)]
pub struct Options {
    /// Fuse primitives into generated kernels (where the device does).
    pub fuse: bool,
    /// Inputs the caller donates (JAX: `donate_argnums`; see
    /// [`PlanOptions::donate`]).
    pub donate: Vec<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            fuse: true,
            donate: Vec::new(),
        }
    }
}

/// `graph` compiled for `device`: a plan that runs on any device, its
/// fusions in `device`'s kernels where the device has a compiler.
pub fn compile(graph: &Graph, device: Device) -> Result<Plan, String> {
    compile_with(graph, device, &Options::default())
}

/// [`compile`] with `options`.
pub fn compile_with(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    let graph = &cse::cse(graph);
    match device {
        #[cfg(lumen_mps_linked)]
        Device::Mps => mps::compile(graph, options),
        _ => {
            let plan = PlanOptions {
                scratch: None,
                donate: options.donate.clone(),
            };
            Ok(Plan::compile_with(graph, &plan))
        }
    }
}
