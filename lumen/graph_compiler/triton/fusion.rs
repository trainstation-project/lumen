use crate::codegen::triton::fusible;
use crate::graph::{Graph, Primitive};
use crate::graph_compiler::common::loops::fused;
use crate::graph_compiler::pass::{CompilerPass, Module};

pub(super) struct LoopFusion;

impl<M: Module> CompilerPass<M> for LoopFusion {
    fn name(&self) -> &'static str {
        "loop_fusion"
    }

    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let graph = m.graph();
        *m.graph_mut() = fused(graph, &groups(graph), |root, _| ("loop", root));
        Ok(())
    }
}

fn groups(graph: &Graph) -> Vec<Option<usize>> {
    let nodes = graph.nodes();
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        for &v in &node.inputs {
            if !readers[v].contains(&i) {
                readers[v].push(i);
            }
        }
    }
    let mut group: Vec<Option<usize>> = vec![None; nodes.len()];
    for (i, node) in nodes.iter().enumerate().rev() {
        if !fusible(graph, node) {
            continue;
        }
        let joined = match readers[node.output].split_first() {
            Some((&first, rest)) if !graph.outputs().contains(&node.output) => {
                group[first].filter(|&g| rest.iter().all(|&r| group[r] == Some(g)))
            }
            _ => None,
        };
        if joined.is_none() && matches!(node.primitive, Primitive::Reshape { .. }) {
            continue;
        }
        group[i] = Some(joined.unwrap_or(i));
    }
    group
}
