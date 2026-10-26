use super::groups::{Groups, planned};
use crate::codegen::tpu::reduction::reducible;
use crate::graph::Graph;
use crate::graph_compiler::common::loops::fused;

pub(super) fn loop_fusion(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let Groups { group, reduces } = planned(graph);
    fused(
        graph,
        &group,
        |root, members| {
            let main = members.iter().find(|&&j| reducible(graph, &nodes[j]));
            let name = match reduces[root] {
                true => "reduction",
                false => "loop",
            };
            (name, *main.unwrap_or(&root))
        },
        |_| Vec::new(),
    )
}
