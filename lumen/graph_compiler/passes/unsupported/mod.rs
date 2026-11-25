//! Rejecting graphs with an op a device has no kernel for, before any other pass: `Unsupported` fails compilation
//! with a readable error and changes nothing otherwise. It is the first pass in the list, so it sees the graph
//! exactly as traced.
//!
//! Devices: every device implements `Supports` (its check of the graph, in `<device>.rs`): the CPU (no Pallas
//! kernels), CUDA (ops and dtypes the CuTe DSL backend has kernels for), MPS (float64 and float8 values, bfloat16
//! transcendentals, dots of two dtypes), Triton (float16 and bfloat16 transcendentals, which libdevice lacks) and TPU
//! (float16, cumsum, scatter-add and random bits). "Transcendental" is
//! `Primitive::is_transcendental`; the operand checked is a node's first input (or its output when it has none).

mod cpu;
mod cuda;
#[cfg(lumen_mps_linked)]
mod mps;
mod tpu;
mod triton;

use crate::graph::Graph;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::{CompilerPass, Module};

/// A device telling the `unsupported` pass which graphs it has no kernels for.
pub(crate) trait Supports {
    /// The error for the first op of `graph` this backend has no kernel for, if any.
    fn check(graph: &crate::graph::Graph) -> Result<(), String>;
}

/// A pass failing on a graph `check` rejects, leaving the graph as it is otherwise.
pub(crate) struct Unsupported {
    pub(crate) check: fn(&Graph) -> Result<(), String>,
}

impl Unsupported {
    /// The pass, with device `D`'s check.
    pub(crate) fn of<D: Supports>() -> Self {
        Unsupported { check: D::check }
    }
}

impl<D: Supports> CompilerPass<DeviceModule<'_, D>> for Unsupported {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "unsupported"
    }

    /// Fails when the module's graph has an op the device has no kernel for.
    fn apply(&mut self, m: &mut DeviceModule<'_, D>) -> Result<(), String> {
        (self.check)(m.graph())
    }
}
