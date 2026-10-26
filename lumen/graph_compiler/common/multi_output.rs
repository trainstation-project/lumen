//! Multi-output fusion shared by the backends: a fusion also writes, as an extra output, a value
//! another fusion produced and others read, so that value needs no kernel of its own.

use crate::graph::{Graph, Primitive};
use crate::graph_compiler::elementwise;

/// A graph's producers and readers, as multi-output fusion walks them.
pub(crate) struct Readers<'a> {
    pub(crate) graph: &'a Graph,
    /// Node producing each value, if any.
    pub(crate) producer: &'a [Option<usize>],
    /// Nodes reading each value, in ascending order.
    pub(crate) users: &'a [Vec<usize>],
}

impl Readers<'_> {
    /// The fusion that can also write node `i`'s value: the one its first reader belongs to, when
    /// every other reader comes after that fusion (so it runs first) and the fusion reaches the
    /// value from its main node through elementwise members, writing it at an index it already
    /// computes. `fusion_of(reader)` gives the fusion's main node (a reduction, or its root) and the
    /// node its kernel is rooted at; `member(main, node)` whether `node` is computed inside it; and
    /// `hostable(main, fusion)` whether the backend lets that fusion take an extra output.
    pub(crate) fn host(
        &self,
        i: usize,
        fusion_of: impl Fn(usize) -> (usize, usize),
        member: impl Fn(usize, usize) -> bool,
        hostable: impl Fn(usize, usize) -> bool,
    ) -> Option<usize> {
        let value = self.graph.nodes()[i].output;
        let &reader = self.users[value].first()?;
        let (main, fusion) = fusion_of(reader);
        let after = self.users[value].iter().all(|&u| u == reader || u > fusion);
        let reached = || self.at_index(main, i, |p| member(main, p));
        (after && hostable(main, fusion) && reached()).then_some(fusion)
    }

    /// Whether node `i`'s value is reached from `main` (from a reduction's input) through
    /// elementwise nodes `member` accepts, so a fusion of `main` computes it at its own index.
    fn at_index(&self, main: usize, i: usize, member: impl Fn(usize) -> bool) -> bool {
        let nodes = self.graph.nodes();
        let value = nodes[i].output;
        let start = match nodes[main].primitive {
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. } => nodes[main].inputs[0],
            _ => nodes[main].output,
        };
        let mut stack = vec![start];
        while let Some(x) = stack.pop() {
            if x == value {
                return true;
            }
            let Some(p) = self.producer[x] else { continue };
            if member(p) && elementwise(&nodes[p].primitive) {
                stack.extend(&nodes[p].inputs);
            }
        }
        false
    }
}
