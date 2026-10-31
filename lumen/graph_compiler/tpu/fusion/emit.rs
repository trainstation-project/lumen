//! Turns the planned fusion groups into `elementwise` and `reduction` fusion nodes.
use super::groups::{Groups, planned};
use crate::codegen::reduction::tpu::reducible;
use crate::graph::Graph;
use crate::graph_compiler::common::elementwise::fused;

/// Rewrites `graph` with each planned group replaced by one fusion, named `reduction` when it reduces.
pub(super) fn elementwise_fusion(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let Groups { group, reduces } = planned(graph);

    fused(
        graph,
        &group,
        |root, members| {
            // A reducing member leads the fusion so the kernel is built around it.
            let main = members.iter().find(|&&j| reducible(graph, &nodes[j]));

            let name = match reduces[root] {
                true => "reduction",
                false => "elementwise",
            };

            (name, *main.unwrap_or(&root))
        },
        |_| Vec::new(),
    )
}
