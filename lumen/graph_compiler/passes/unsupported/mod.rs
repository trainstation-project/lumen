//! Rejecting graphs with an op a device has no kernel for, before any other pass: each device's check in its own
//! file here.

#[cfg(lumen_mps_linked)]
mod mps;
mod triton;

use crate::graph::Graph;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::identity::Identity;
use crate::graph_compiler::passes::{CompilerPass, Module};
use crate::graph_compiler::pipeline::Pass;

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

/// What each device runs for this pass (each device's in its own file here); a device it doesn't run on runs the
/// identity pass.
pub(crate) trait UnsupportedDevice: Sized + 'static {
    /// Rejects graphs with an op the device has no kernel for.
    fn unsupported<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Identity)
    }
}
