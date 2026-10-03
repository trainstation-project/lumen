//! Normalization diamonds (XLA's SoftmaxRewriterTriton, see
//! xla/backends/gpu/transforms/softmax_rewriter_triton.cc): whatever
//! normalizes rows by a reduction of them, however it is written, run as
//! one row kernel (`codegen.rs`). Nothing here knows about any one
//! normalization: an RMS norm (with `mean` or `sum`, `x / sqrt` or
//! `x * (1 / sqrt)`), a softmax written out and a layer norm are all
//! diamonds, or chains of them.
//!
//! A diamond is a root, an elementwise binary primitive, whose operands
//! both come from one producer `p`: one through trivial primitives (below),
//! the other through a reduction over the last dimension, trivial
//! primitives, and a broadcast back along it:
//!
//! ```text
//!              p
//!            /   \
//!     trivial     trivial -> reduce (last dim) -> trivial -> broadcast
//!            \   /
//!            root
//! ```
//!
//! A trivial primitive (XLA's `IsTriviallyFusible`) reads no more than it
//! writes, at the same index, and has one reader: a reshape, a unary
//! elementwise primitive, or a binary one of one value (`x * x`) or of one
//! value and a splat (a constant, or a graph input broadcast: `+ eps`,
//! `/ n`). Diamonds chain where one's root is (through elementwise
//! primitives read only by the next) the next one's producer: a softmax's
//! max then its sum, a layer norm's mean then its variance. Each chain is
//! one row kernel, its reductions inside.

use super::fusion::{constant, elementwise};
use crate::graph::{Graph, Primitive, Var};

/// A row fusion: a chain of diamonds, run as one row kernel. `root` (a
/// node index) is the fusion's root, `reductions` its reductions (in
/// order), `inner` the other nodes between them and the root, which are
/// read nowhere else.
#[derive(Debug, Clone)]
pub(crate) struct Row {
    pub root: usize,
    pub reductions: Vec<usize>,
    pub inner: Vec<usize>,
}

/// The chains of diamonds in `graph`, each a [`Row`].
pub(crate) fn diamonds(graph: &Graph) -> Vec<Row> {
    let finder = Finder::new(graph);
    let mut rows: Vec<Row> = Vec::new();
    for i in 0..graph.nodes().len() {
        let taken = |i: usize| rows.iter().any(|r| r.root == i || r.inner.contains(&i));
        if !finder.live[i] || taken(i) {
            continue;
        }
        let Some((mut row, producer)) = finder.diamond(i) else {
            continue;
        };
        if row.inner.iter().any(|&n| taken(n)) {
            continue;
        }
        // A diamond whose producer is an earlier one's root extends it.
        let earlier = finder.producer[producer].and_then(|p| rows.iter().position(|r| r.root == p));
        if let Some(k) = earlier {
            let mut inside = row.inner.clone();
            inside.push(row.root);
            if finder.read_only_by(producer, &inside) {
                let first = rows.remove(k);
                row.inner.extend(first.inner);
                row.inner.push(first.root);
                row.reductions.splice(0..0, first.reductions);
            }
        }
        rows.push(row);
    }
    rows
}

struct Finder<'a> {
    graph: &'a Graph,
    producer: Vec<Option<usize>>,
    /// Each value's live readers (once each), and whether it is an output.
    readers: Vec<Vec<usize>>,
    output: Vec<bool>,
    live: Vec<bool>,
}

impl<'a> Finder<'a> {
    fn new(graph: &'a Graph) -> Self {
        let nodes = graph.nodes();
        let n = graph.types.len();
        let mut producer = vec![None; n];
        for (i, node) in nodes.iter().enumerate() {
            producer[node.output] = Some(i);
        }
        let mut output = vec![false; n];
        graph.outputs().iter().for_each(|&v| output[v] = true);
        let mut live = vec![false; nodes.len()];
        let mut live_var = output.clone();
        for (i, node) in nodes.iter().enumerate().rev() {
            if live_var[node.output] {
                live[i] = true;
                node.inputs.iter().for_each(|&v| live_var[v] = true);
            }
        }
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, node) in nodes.iter().enumerate().filter(|&(i, _)| live[i]) {
            for &v in &node.inputs {
                if readers[v].last() != Some(&i) {
                    readers[v].push(i);
                }
            }
        }
        Finder {
            graph,
            producer,
            readers,
            output,
            live,
        }
    }

    /// Whether `v` is read only by `nodes` (and is no output).
    fn read_only_by(&self, v: Var, nodes: &[usize]) -> bool {
        !self.output[v] && self.readers[v].iter().all(|r| nodes.contains(r))
    }

    /// Whether `v` is a splat: the same everywhere along a row (a
    /// constant, or a broadcast graph input: a runtime scalar, a weight).
    fn splat(&self, v: Var) -> bool {
        let graph_input = |v: Var| self.producer[v].is_none();
        constant(self.graph, &self.producer, v)
            || self.producer[v].is_some_and(|i| {
                let node = &self.graph.nodes()[i];
                matches!(node.primitive, Primitive::BroadcastInDim { .. })
                    && graph_input(node.inputs[0])
            })
    }

    /// Whether node `i` is trivially fusible (XLA's `IsTriviallyFusible`):
    /// one reader, and a reshape, a unary elementwise primitive, or a
    /// binary one of one value or of one value and a splat.
    fn trivial(&self, i: usize) -> bool {
        let node = &self.graph.nodes()[i];
        if self.output[node.output] || self.readers[node.output].len() > 1 {
            return false;
        }
        match (&node.primitive, node.inputs.as_slice()) {
            (Primitive::Reshape { .. }, _) => true,
            (p, [_]) => elementwise(p),
            (p, &[a, b]) => elementwise(p) && (a == b || self.splat(a) != self.splat(b)),
            _ => false,
        }
    }

    /// The operand a trivial node `i` is followed through: its first that
    /// is not a splat.
    fn through(&self, i: usize) -> Var {
        let inputs = &self.graph.nodes()[i].inputs;
        match inputs.len() > 1 && self.splat(inputs[0]) {
            true => inputs[1],
            false => inputs[0],
        }
    }

    /// From value `v` back through trivial nodes (XLA's `TrivialEdge`) to
    /// one whose primitive passes `is`: it, and the trivial nodes passed.
    fn edge(&self, mut v: Var, is: fn(&Primitive) -> bool) -> Option<(usize, Vec<usize>)> {
        let mut path = Vec::new();
        loop {
            let i = self.producer[v]?;
            if is(&self.graph.nodes()[i].primitive) {
                return Some((i, path));
            }
            if !self.trivial(i) {
                return None;
            }
            path.push(i);
            v = self.through(i);
        }
    }

    /// The diamond rooted at node `i`, if it is one's root, and its
    /// producer.
    fn diamond(&self, i: usize) -> Option<(Row, Var)> {
        let nodes = self.graph.nodes();
        let root = &nodes[i];
        let (&[a, b], p) = (root.inputs.as_slice(), &root.primitive) else {
            return None;
        };
        if !elementwise(p) {
            return None;
        }
        // The broadcast side second, or (commutative) either.
        let commutative = matches!(p, Primitive::Add | Primitive::Mul | Primitive::Max);
        let sides: &[(Var, Var)] = if commutative {
            &[(a, b), (b, a)]
        } else {
            &[(a, b)]
        };
        sides
            .iter()
            .find_map(|&(side, reduced)| self.closed(i, side, reduced))
    }

    /// The diamond of root `i` whose operand `reduced` is a broadcast of a
    /// reduction of the producer `side` comes from.
    fn closed(&self, i: usize, side: Var, reduced: Var) -> Option<(Row, Var)> {
        let nodes = self.graph.nodes();
        let ty = self.graph.type_of(nodes[i].output);
        let (b, mut inner) =
            self.edge(reduced, |p| matches!(p, Primitive::BroadcastInDim { .. }))?;
        let (r, path) = self.edge(nodes[b].inputs[0], |p| {
            matches!(p, Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. })
        })?;
        inner.push(b);
        inner.extend(path);
        // Over the last dimension of a value of the root's shape, broadcast
        // back along it alone.
        let last = ty.shape.len().checked_sub(1)?;
        let x = nodes[r].inputs[0];
        let rows = ty.numel() / ty.shape[last].max(1);
        let (Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes }) =
            &nodes[r].primitive
        else {
            unreachable!("a reduction")
        };
        let Primitive::BroadcastInDim {
            broadcast_dimensions,
            ..
        } = &nodes[b].primitive
        else {
            unreachable!("a broadcast")
        };
        let along = axes.as_slice() == [last]
            && self.graph.type_of(x).shape == ty.shape
            && self.graph.type_of(nodes[b].output).shape == ty.shape
            && broadcast_dimensions.iter().enumerate().all(|(k, &d)| {
                // Along the last dimension only from a size-1 one (keepdim).
                d != last || self.graph.type_of(nodes[b].inputs[0]).shape[k] == 1
            })
            && self.graph.type_of(nodes[b].inputs[0]).numel() == rows;
        let single =
            |n: usize| self.readers[nodes[n].output].len() == 1 && !self.output[nodes[n].output];
        if !along || !single(b) || !single(r) {
            return None;
        }
        // The producer: the reduction's operand back through trivial nodes.
        let mut producer = x;
        while let Some(k) = self.producer[producer].filter(|&k| self.trivial(k)) {
            inner.push(k);
            producer = self.through(k);
        }
        // The root's other operand reaches it trivially too.
        let mut v = side;
        while v != producer {
            let k = self.producer[v].filter(|&k| self.trivial(k))?;
            inner.push(k);
            v = self.through(k);
        }
        inner.push(r);
        // Then further back, through elementwise nodes read only inside it
        // (a softmax's exp, read by its sum and its division): its producer
        // can be an earlier diamond's root.
        while let Some(k) = self.producer[producer] {
            let node = &nodes[k];
            let mut inside = inner.clone();
            inside.push(i);
            let follow = elementwise(&node.primitive)
                && !node.inputs.is_empty()
                && node
                    .inputs
                    .iter()
                    .all(|&u| u == self.through(k) || self.splat(u))
                && self.read_only_by(producer, &inside);
            if !follow {
                break;
            }
            inner.push(k);
            producer = self.through(k);
        }
        let row = Row {
            root: i,
            reductions: vec![r],
            inner,
        };
        Some((row, producer))
    }
}
