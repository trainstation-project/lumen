//! What TPU runs for this pass.

use crate::graph::Node;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::fusion::tpu as fusion;
use crate::graph_compiler::passes::place_concats::{PlaceConcats, Slots};
use crate::graph_compiler::pipeline::Pass;

/// Exposes the graph and the TPU compiler config to the shared pass framework.
impl Slots for DeviceModule<'_, Tpu> {
    /// What `fusion::slot_writes` accepts.
    fn writes(&self, concat: &Node, k: usize, producer: &Node, _: usize) -> bool {
        fusion::slot_writes(&self.graph, concat, k, producer)
    }

    /// Any operand: on TPU's tiled layout a concatenation fusion would build is a copy of its own.
    fn worth(&self, _: &Node, _: usize, _: &Node, _: usize) -> bool {
        true
    }
}

/// Always.
impl DevicePass<Tpu> for PlaceConcats {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Tpu>> {
        Box::new(PlaceConcats)
    }
}
