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
    let mut reduces = vec![false; nodes.len()];
    let mut dots = vec![false; nodes.len()];
    for (i, node) in nodes.iter().enumerate().rev() {
        let dot = matches!(node.primitive, Primitive::DotGeneral { .. });
        if !fusible(graph, node) && !dot {
            continue;
        }
        let mut joined = match readers[node.output].split_first() {
            Some((&first, rest)) if !graph.outputs().contains(&node.output) => {
                group[first].filter(|&g| rest.iter().all(|&r| group[r] == Some(g)))
            }
            _ => None,
        };
        let operand = readers[node.output]
            .iter()
            .any(|&r| matches!(nodes[r].primitive, Primitive::DotGeneral { .. }));
        joined = joined.filter(|_| !operand);
        let reduction = reduction(&node.primitive);
        if reduction || dot {
            joined = joined.filter(|&g| {
                !reduces[g] && !dots[g] && epilogue(graph, &readers, &group, node.output, g)
            });
        }
        if joined.is_none() && (dot || matches!(node.primitive, Primitive::Reshape { .. })) {
            continue;
        }
        let g = joined.unwrap_or(i);
        group[i] = Some(g);
        reduces[g] |= reduction;
        dots[g] |= dot;
    }
    group
}

fn reduction(p: &Primitive) -> bool {
    matches!(p, Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. })
}

fn epilogue(
    graph: &Graph,
    readers: &[Vec<usize>],
    group: &[Option<usize>],
    reduced: usize,
    g: usize,
) -> bool {
    let nodes = graph.nodes();
    let mut stack = vec![reduced];
    while let Some(v) = stack.pop() {
        for &r in &readers[v] {
            if group[r] != Some(g) {
                continue;
            }
            let p = &nodes[r].primitive;
            if matches!(
                p,
                Primitive::BroadcastInDim { .. }
                    | Primitive::Transpose { .. }
                    | Primitive::Slice { .. }
                    | Primitive::DynamicSlice { .. }
                    | Primitive::Gather { .. }
                    | Primitive::TakeAlongAxis { .. }
            ) {
                return false;
            }
            stack.push(nodes[r].output);
        }
    }
    true
}
