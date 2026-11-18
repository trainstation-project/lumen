//! What Triton runs for this pass.

use crate::graph::{Node, Primitive};
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

    /// When operand `k`'s producer is a GEMM (a kernel of its own writing the slot), or when something but a
    /// fusible elementwise op or reshape reads the concatenation (it is in memory anyway). A concatenation of
    /// cheap producers read only elementwise is left to fusion, which builds it in registers inside its readers'
    /// kernel (as rotary's `cat([-x2, x1])` before `* sin`): no step and no buffer for it.
    fn worth(&self, concat: &Node, _: usize, producer: &Node, _: usize) -> bool {
        let graph = &self.graph;

        let fused = |reader: &Node| {
            fusion::fusible(graph, reader)
                && (crate::graph_compiler::elementwise(&reader.primitive)
                    || matches!(reader.primitive, Primitive::Reshape { .. }))
        };

        let mut readers = graph
            .nodes()
            .iter()
            .filter(|n| n.inputs.contains(&concat.output))
            .peekable();

        matches!(producer.primitive, Primitive::DotGeneral { .. })
            || graph.outputs().contains(&concat.output)
            || readers.peek().is_none()
            || readers.any(|r| !fused(r))
    }
}

/// With `fuse`.
impl DevicePass<Triton> for PlaceConcats {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
        when(config.fuse, PlaceConcats)
    }
}
