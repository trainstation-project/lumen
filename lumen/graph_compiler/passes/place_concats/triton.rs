//! What Triton runs for this pass.

use super::PlaceConcatsDevice;
use crate::graph::Node;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::fusion::triton as fusion;
use crate::graph_compiler::passes::place_concats::{PlaceConcats, Slots};
use crate::graph_compiler::pipeline::{Pass, when};

/// Triton's answers to the pass.
impl PlaceConcatsDevice for Triton {
    /// With `fuse`.
    fn place_concats<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        when(config.fuse, PlaceConcats)
    }
}

impl Slots for DeviceModule<'_, Triton> {
    /// Kernels of their own writing fresh memory (`fusion::slot_writes`).
    fn writes(&self, _: &Node, _: usize, producer: &Node, _: usize) -> bool {
        fusion::slot_writes(&self.graph, producer)
    }
}
