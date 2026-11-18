//! Rejecting graphs with an op a device has no kernel for, before any other pass: `Unsupported` fails compilation
//! with a readable error and changes nothing otherwise. It is the first pass in the list, so it sees the graph
//! exactly as traced.
//!
//! Devices: it runs on each device whose codegen implements `codegen::Supports` (its check of the graph): MPS
//! (`codegen/mps.rs`: float64 and float8 values, bfloat16 transcendentals, dots of two dtypes) and Triton
//! (`codegen/triton.rs`: float16 and bfloat16 transcendentals, which libdevice lacks). CPU, CUDA and TPU get the
//! identity pass. "Transcendental" is `Primitive::is_transcendental`; the operand checked is a node's first
//! input (or its output when it has none).

use crate::codegen::Supports;
use crate::graph::Graph;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::{CompilerPass, DevicePass, Module};
use crate::graph_compiler::pipeline::Pass;

/// A pass failing on a graph `check` rejects, leaving the graph as it is otherwise.
pub(crate) struct Unsupported {
    pub(crate) check: fn(&Graph) -> Result<(), String>,
}

/// Always, on every device whose codegen implements `Supports`: rejects graphs with an op it has no kernel for.
impl<D: Supports> DevicePass<D> for Unsupported {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, D>> {
        Box::new(Unsupported { check: D::check })
    }
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
