//! TPU's answers to the `alone` pass: which lone nodes only a generated kernel computes, and the kernels' names.

use crate::codegen::devices::tpu::CONCATENATE;
use crate::codegen::devices::tpu::Tpu;
use crate::graph::Primitive;
use crate::graph_compiler::device::DeviceModule;

/// TPU's lone kernels: each concatenation not written in place, copied by a `CONCATENATE` fusion of its own, and each
/// slice fusion left out and `place_views` made no view, copied by an `elementwise` one when that plans; TPU passes no
/// input by value.
impl super::Lone for DeviceModule<'_, Tpu> {
    /// Whether `node` is a concatenation not written in place, or a slice that is no view whose elementwise kernel
    /// plans (others stay as fusion left them).
    fn alone(graph: &crate::graph::Graph, node: &crate::graph::Node) -> bool {
        let planned = || {
            let i = graph
                .nodes()
                .iter()
                .position(|n| std::ptr::eq(n, node))
                .expect("a node of the graph");

            let (body, _) = crate::graph_compiler::passes::elementwise::body(graph, &[i], i, &[]);
            crate::codegen::elementwise::tpu::grid::plan(&body, 0).is_ok()
        };

        match node.primitive {
            Primitive::Concatenate {
                in_place: false, ..
            } => true,
            Primitive::Slice { view: false, .. } => planned(),
            _ => false,
        }
    }

    /// None: TPU kernels read every input from memory.
    fn scalars(&mut self) -> Option<&mut Vec<usize>> {
        None
    }

    /// `CONCATENATE` for a concatenation, else `elementwise`.
    fn kernel(&mut self, body: &crate::graph::Graph, _: &[bool]) -> String {
        let concatenation = body
            .nodes()
            .iter()
            .any(|n| matches!(n.primitive, Primitive::Concatenate { .. }));

        match concatenation {
            true => CONCATENATE.into(),
            false => "elementwise".into(),
        }
    }
}
