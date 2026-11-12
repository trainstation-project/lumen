//! What TPU runs for this pass.

use super::PlaceConcatsDevice;
use crate::graph::Node;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::fusion::tpu as fusion;
use crate::graph_compiler::passes::place_concats::{PlaceConcats, Slots};
use crate::graph_compiler::pipeline::Pass;

/// TPU's answers to the pass.
impl PlaceConcatsDevice for Tpu {
    /// Always.
    fn place_concats<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(PlaceConcats)
    }
}

/// Exposes the graph and the TPU compiler config to the shared pass framework.
impl Slots for DeviceModule<'_, Tpu> {
    /// What `fusion::slot_writes` accepts.
    fn writes(&self, concat: &Node, k: usize, producer: &Node, _: usize) -> bool {
        fusion::slot_writes(&self.graph, concat, k, producer)
    }
}
