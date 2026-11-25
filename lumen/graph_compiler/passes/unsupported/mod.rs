//! Rejecting graphs with an op a device has no kernel for, before any other pass: `Unsupported` fails compilation
//! with a readable error and changes nothing otherwise. It is the first pass in the list, so it sees the graph
//! exactly as traced.
//!
//! Devices: every device's codegen implements `codegen::devices::Supports` (its check of the graph): the CPU (`reference`:
//! no Pallas kernels), CUDA (`codegen/devices/cuda`: ops and dtypes the CuTe DSL backend has kernels for), MPS
//! (`codegen/devices/mps`: float64 and float8 values, bfloat16 transcendentals, dots of two dtypes), Triton
//! (`codegen/devices/triton`: float16 and bfloat16 transcendentals, which libdevice lacks) and TPU
//! (`codegen/devices/tpu`: float16, cumsum, scatter-add and random bits). "Transcendental" is
//! `Primitive::is_transcendental`; the operand checked is a node's first input (or its output when it has none).

use crate::codegen::devices::Supports;
use crate::graph::Graph;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::{CompilerPass, Module};

/// A pass failing on a graph `check` rejects, leaving the graph as it is otherwise.
pub(crate) struct Unsupported {
    pub(crate) check: fn(&Graph) -> Result<(), String>,
}

impl<D: Supports> CompilerPass<DeviceModule<'_, D>> for Unsupported {
    /// Always, with the device's check.
    fn configured(_: &CompilerConfig) -> Option<Self> {
        Some(Unsupported { check: D::check })
    }

    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "unsupported"
    }

    /// Fails when the module's graph has an op the device has no kernel for.
    fn apply(&mut self, m: &mut DeviceModule<'_, D>) -> Result<(), String> {
        (self.check)(m.graph())
    }
}
