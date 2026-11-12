//! What Triton runs for this pass.

use crate::graph::Node;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::fusion::triton as fusion;
use crate::graph_compiler::passes::place_concats::{PlaceConcats, Slots};
use crate::graph_compiler::pipeline::{Pass, when};

impl Slots for DeviceModule<'_, Triton> {
    /// Kernels of their own writing fresh memory (`fusion::slot_writes`).
    fn writes(&self, _: &Node, _: usize, producer: &Node, _: usize) -> bool {
        fusion::slot_writes(&self.graph, producer)
    }
}

/// With `fuse`.
impl DevicePass<Triton> for PlaceConcats {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
        when(config.fuse, PlaceConcats)
    }
}
