//! Rejecting graphs with an op a device has no kernel for, before any other pass: each device's check in its own
//! file here.

#[cfg(lumen_mps_linked)]
mod mps;
mod triton;

use crate::graph::Graph;
use crate::graph_compiler::passes::{CompilerPass, Module};

/// A pass failing on a graph `check` rejects, leaving the graph as it is otherwise.
pub(crate) struct Unsupported {
    pub(crate) check: fn(&Graph) -> Result<(), String>,
}

impl<M: Module> CompilerPass<M> for Unsupported {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "unsupported"
    }

    /// Fails when the module's graph has an op the device has no kernel for.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        (self.check)(m.graph())
    }
}
