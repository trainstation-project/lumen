//! Detection of row-normalization patterns ("diamonds"): a value combined with a broadcast of its own
//! last-axis reduction, which a backend can run as one row kernel.

use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::{constant, elementwise};

/// Whether the backend can put a node in a row kernel.
pub(crate) type Fusible = fn(&Graph, &Node) -> bool;

/// Nodes of one row kernel, as node indices into the graph.
#[derive(Debug, Clone)]
pub(crate) struct Row {
    /// Last node of the row; its output is the kernel's main result.
    pub root: usize,
    /// Last-axis reductions computed inside the row.
    pub reductions: Vec<usize>,
    /// Other nodes computed inside the row.
    pub inner: Vec<usize>,
    /// Nodes whose values are also read outside the row (or are graph outputs).
    pub outputs: Vec<usize>,
    /// Partial column sums a backend attached to the row.
    pub partials: Vec<usize>,
}

/// Every row kernel in `graph`, built from diamonds and then grown backwards and forwards.
pub(crate) fn diamonds(graph: &Graph, fusible: Fusible) -> Vec<Row> {
    let finder = Finder::new(graph, fusible);
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
        // A diamond over the root of an earlier row, read only here, absorbs that row (chained norms).
        let earlier = finder.producer[producer].and_then(|p| rows.iter().position(|r| r.root == p));
        if let Some(k) = earlier {
            let mut inside = row.inner.clone();
            inside.push(row.root);
            if finder.read_only_by(producer, &inside) {
                let first = rows.remove(k);
                row.inner.extend(first.inner);
                row.inner.push(first.root);
                row.reductions.splice(0..0, first.reductions);
                row.outputs.splice(0..0, first.outputs);
            }
        }
        rows.push(row);
    }
    grown(&finder, &mut rows);
    consumed(&finder, &mut rows);
    rows
}

/// Grows rows backwards from each free elementwise root, absorbing same-shaped producers, per-row
/// values and earlier rows, so a pattern with several reductions becomes one row.
fn grown(finder: &Finder, rows: &mut Vec<Row>) {
    let graph = finder.graph;
    let nodes = graph.nodes();
    let mut taken = vec![false; nodes.len()];
    for r in rows.iter() {
        for &k in std::iter::once(&r.root)
            .chain(&r.inner)
            .chain(&r.reductions)
        {
            taken[k] = true;
        }
    }
    for i in (0..nodes.len()).rev() {
        let root = &nodes[i];
        let own = rows
            .iter()
            .position(|r| r.root == i && r.outputs.is_empty());
        let free = !taken[i] || own.is_some();
        if !finder.live[i]
            || !free
            || !elementwise(&root.primitive)
            || !(finder.fusible)(graph, root)
        {
            continue;
        }
        let is_reduction = |k: usize| {
            matches!(
                nodes[k].primitive,
                Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
            )
        };
        let ty = graph.type_of(root.output);
        let rows_of: &[Row] = rows;
        // Flood-fill earlier nodes that fit a layout of `rows` elements with rows of `n`.
        let grow = |rows: usize, n: usize| -> (Vec<bool>, Vec<usize>, Vec<usize>) {
            let each = rows / n;
            let shaped = |v: Var| {
                let t = graph.type_of(v);
                t.numel() == rows && t.shape.last() == Some(&n)
            };
            let per_row = |v: Var| graph.type_of(v).numel() == each;
            let mut inside = vec![false; nodes.len()];
            inside[i] = true;
            let mut absorbed: Vec<usize> = Vec::new();
            let mut outputs: Vec<usize> = Vec::new();
            if let Some(r) = own {
                let row = &rows_of[r];
                for &m in row.inner.iter().chain(&row.reductions) {
                    inside[m] = true;
                }
                absorbed.push(r);
            }
            for k in (0..i).rev() {
                let node = &nodes[k];
                let v = node.output;
                let read_inside = !finder.output[v]
                    && !finder.readers[v].is_empty()
                    && finder.readers[v].iter().all(|&r| inside[r]);
                // Per-row values may also be read after the root: they become row outputs.
                let read_after = !finder.output[v]
                    && per_row(v)
                    && finder.readers[v].iter().any(|&r| inside[r])
                    && finder.readers[v].iter().all(|&r| inside[r] || r > i);
                // Index arithmetic is recomputed by readers rather than claimed by the row.
                let shared = !read_inside && indices(graph, &finder.producer, v);
                if !finder.live[k] || inside[k] || !(read_inside || read_after) || shared {
                    continue;
                }
                let own_reads = |r: &Row| {
                    r.outputs.iter().all(|&o| {
                        let w = nodes[o].output;
                        !finder.output[w]
                            && finder.readers[w].iter().all(|&u| {
                                inside[u]
                                    || u == r.root
                                    || r.inner.contains(&u)
                                    || r.reductions.contains(&u)
                            })
                    })
                };
                let earlier = rows_of
                    .iter()
                    .position(|r| r.root == k && shaped(v) && own_reads(r));
                if let Some(r) = earlier {
                    let row = &rows_of[r];
                    for &m in std::iter::once(&row.root)
                        .chain(&row.inner)
                        .chain(&row.reductions)
                    {
                        inside[m] = true;
                    }
                    absorbed.push(r);
                    continue;
                }
                if taken[k] || !(finder.fusible)(graph, node) {
                    continue;
                }
                inside[k] = match &node.primitive {
                    Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => {
                        let x = node.inputs[0];
                        shaped(x)
                            && axes.as_slice() == [graph.type_of(x).shape.len() - 1]
                            && per_row(v)
                    }
                    _ => shaped(v) || per_row(v) || graph.type_of(v).numel() == 1,
                };
                if inside[k] && !read_inside {
                    outputs.push(k);
                }
            }
            (inside, absorbed, outputs)
        };
        let reduces = |inside: &[bool]| (0..i).any(|k| inside[k] && is_reduction(k));
        // Prefer the root's own row length; a per-row root takes the length of a reduction feeding it.
        let mut grown = match ty.shape.last() {
            Some(&n) if n > 1 => Some(grow(ty.numel(), n)),
            _ => None,
        };
        if !grown.as_ref().is_some_and(|(inside, _, _)| reduces(inside)) {
            grown = row_length(finder, i)
                .map(|n| grow(ty.numel() * n, n))
                .filter(|(inside, _, _)| {
                    (0..i).filter(|&k| inside[k] && is_reduction(k)).count() > 1
                });
        }
        let Some((inside, mut absorbed, mut outputs)) = grown else {
            continue;
        };
        let reductions: Vec<usize> = (0..i).filter(|&k| inside[k] && is_reduction(k)).collect();
        let grew = own.is_none_or(|r| {
            let row = &rows[r];
            (0..i).filter(|&k| inside[k]).count() > row.inner.len() + row.reductions.len()
        });
        if reductions.is_empty() || !grew {
            continue;
        }
        let inner: Vec<usize> = (0..i).filter(|&k| inside[k] && !is_reduction(k)).collect();
        for &k in inner.iter().chain(&reductions).chain([&i]) {
            taken[k] = true;
        }
        absorbed.sort_unstable();
        for r in absorbed.into_iter().rev() {
            rows.remove(r);
        }
        outputs.sort_unstable();
        rows.push(Row {
            root: i,
            reductions,
            inner,
            outputs,
            partials: Vec::new(),
        });
    }
}

/// Whether `v` is computed only from iotas and constants through layout and elementwise ops.
fn indices(graph: &Graph, producer: &[Option<usize>], v: Var) -> bool {
    let Some(p) = producer[v] else { return false };
    let node = &graph.nodes()[p];
    match node.primitive {
        Primitive::Iota { .. } | Primitive::Full { .. } => true,
        Primitive::Reshape { .. } | Primitive::BroadcastInDim { .. } => {
            indices(graph, producer, node.inputs[0])
        }
        ref q if elementwise(q) => node.inputs.iter().all(|&u| indices(graph, producer, u)),
        _ => false,
    }
}

/// Moves each row's root forward to a later elementwise consumer whose cheap input cone reaches the
/// row, so that consumer is computed in the row kernel too.
fn consumed(finder: &Finder, rows: &mut [Row]) {
    let graph = finder.graph;
    let nodes = graph.nodes();
    let is_reduction = |k: usize| {
        matches!(
            nodes[k].primitive,
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
        )
    };
    for r in 0..rows.len() {
        let row = &rows[r];
        let Some(&first) = row.reductions.first() else {
            continue;
        };
        let x = graph.type_of(nodes[first].inputs[0]);
        let Some(&n) = x.shape.last() else { continue };
        let (size, each) = (x.numel(), x.numel() / n.max(1));
        let mine: Vec<usize> = std::iter::once(row.root)
            .chain(row.inner.iter().copied())
            .chain(row.reductions.iter().copied())
            .collect();
        let taken = |k: usize| {
            rows.iter()
                .any(|o| o.root == k || o.inner.contains(&k) || o.reductions.contains(&k))
        };
        let shaped = |v: Var| {
            let t = graph.type_of(v);
            t.numel() == size && t.shape.last() == Some(&n)
        };
        let fits = |v: Var| shaped(v) || [each, 1].contains(&graph.type_of(v).numel());
        let cheap = |p: &Primitive| {
            elementwise(p)
                || matches!(
                    p,
                    Primitive::Reshape { .. }
                        | Primitive::BroadcastInDim { .. }
                        | Primitive::Iota { .. }
                )
        };
        // Candidates are the last elementwise nodes of a chain, tried from the end of the graph.
        let candidate = |c: usize| {
            finder.live[c]
                && elementwise(&nodes[c].primitive)
                && (finder.fusible)(graph, &nodes[c])
                && shaped(nodes[c].output)
                && !taken(c)
        };
        let candidates: Vec<usize> = (row.root + 1..nodes.len())
            .rev()
            .filter(|&c| {
                candidate(c)
                    && !finder.readers[nodes[c].output]
                        .iter()
                        .any(|&u| candidate(u))
            })
            .collect();
        let found = candidates.into_iter().find_map(|c| {
            let mut cone: Vec<usize> = vec![c];
            let (mut leaves, mut stack, mut reaches) = (Vec::new(), nodes[c].inputs.clone(), false);
            while let Some(v) = stack.pop() {
                match finder.producer[v] {
                    Some(p) if mine.contains(&p) => reaches = true,
                    Some(p) if cone.contains(&p) => {}
                    Some(p)
                        if !taken(p)
                            && !indices(graph, &finder.producer, v)
                            && (finder.fusible)(graph, &nodes[p])
                            && cheap(&nodes[p].primitive)
                            && fits(v) =>
                    {
                        cone.push(p);
                        stack.extend(&nodes[p].inputs);
                    }
                    _ => {
                        if !leaves.contains(&v) {
                            leaves.push(v);
                        }
                    }
                }
            }
            let inside = |u: usize| cone.contains(&u) || mine.contains(&u);
            let closed = cone[1..].iter().all(|&k| {
                let v = nodes[k].output;
                !finder.output[v] && finder.readers[v].iter().all(|&u| inside(u))
            });
            if !reaches || !closed {
                return None;
            }
            let mut outputs = Vec::new();
            let mut outside = Vec::new();
            for &k in &mine {
                let v = nodes[k].output;
                let readers: Vec<usize> = finder.readers[v]
                    .iter()
                    .copied()
                    .filter(|&u| !inside(u))
                    .collect();
                if finder.output[v] || !readers.is_empty() {
                    if graph.type_of(v).numel() != each {
                        return None;
                    }
                    outputs.push(k);
                    outside.extend(readers);
                }
            }
            // Leaves must not depend on readers of the row's outputs, or the row would wait on itself.
            let depends = |v: Var| {
                let (mut stack, mut seen) = (vec![v], vec![false; graph.types.len()]);
                while let Some(v) = stack.pop() {
                    if std::mem::replace(&mut seen[v], true) {
                        continue;
                    }
                    if let Some(p) = finder.producer[v] {
                        if outside.contains(&p) {
                            return true;
                        }
                        stack.extend(&nodes[p].inputs);
                    }
                }
                false
            };
            (!leaves.iter().any(|&v| depends(v))).then_some((c, cone, outputs))
        });
        let Some((c, cone, mut outputs)) = found else {
            continue;
        };
        let row = &mut rows[r];
        let old = row.root;
        row.inner.push(old);
        row.inner.extend(cone.iter().copied().filter(|&k| k != c));
        row.inner.retain(|&k| !is_reduction(k));
        row.inner.sort_unstable();
        row.inner.dedup();
        outputs.sort_unstable();
        row.outputs = outputs;
        row.root = c;
    }
}

/// Length of the reduced last axis behind per-row value `root`, through elementwise and layout ops.
fn row_length(finder: &Finder, root: usize) -> Option<usize> {
    let graph = finder.graph;
    let nodes = graph.nodes();
    let each = graph.type_of(nodes[root].output).numel();
    let mut stack = nodes[root].inputs.clone();
    let mut seen = vec![false; graph.types.len()];
    while let Some(v) = stack.pop() {
        let Some(k) = finder.producer[v].filter(|_| !std::mem::replace(&mut seen[v], true)) else {
            continue;
        };
        let node = &nodes[k];
        match &node.primitive {
            Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => {
                let x = graph.type_of(node.inputs[0]);
                let last = x.shape.len().checked_sub(1)?;
                if axes.as_slice() == [last]
                    && graph.type_of(v).numel() == each
                    && x.shape[last] > 1
                {
                    return Some(x.shape[last]);
                }
            }
            p if (elementwise(p)
                || matches!(
                    p,
                    Primitive::Reshape { .. } | Primitive::BroadcastInDim { .. }
                ))
                && graph.type_of(v).numel() == each =>
            {
                stack.extend(&node.inputs);
            }
            _ => {}
        }
    }
    None
}

/// Producers, readers and liveness of a graph's values, for row matching.
pub(crate) struct Finder<'a> {
    graph: &'a Graph,
    /// Whether the backend can fuse a node.
    fusible: Fusible,
    /// Node producing each value.
    pub(crate) producer: Vec<Option<usize>>,
    /// Live nodes reading each value.
    pub(crate) readers: Vec<Vec<usize>>,
    /// Whether each value is a graph output.
    pub(crate) output: Vec<bool>,
    /// Whether each node feeds a graph output.
    pub(crate) live: Vec<bool>,
}

impl<'a> Finder<'a> {
    /// Finder over `graph` with uses and liveness computed.
    pub(crate) fn new(graph: &'a Graph, fusible: Fusible) -> Self {
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
            fusible,
            producer,
            readers,
            output,
            live,
        }
    }

    /// Whether `v` isn't an output and only `nodes` read it.
    fn read_only_by(&self, v: Var, nodes: &[usize]) -> bool {
        !self.output[v] && self.readers[v].iter().all(|r| nodes.contains(r))
    }

    /// Whether `v` is a constant or a broadcast graph input (the same for every element of a row).
    fn splat(&self, v: Var) -> bool {
        let graph_input = |v: Var| self.producer[v].is_none();
        constant(self.graph, &self.producer, v)
            || self.producer[v].is_some_and(|i| {
                let node = &self.graph.nodes()[i];
                matches!(node.primitive, Primitive::BroadcastInDim { .. })
                    && graph_input(node.inputs[0])
            })
    }

    /// Whether node `i` is a reshape or an elementwise op on one non-splat value; it must be
    /// single-use unless `shared`.
    fn trivial(&self, i: usize, shared: bool) -> bool {
        let node = &self.graph.nodes()[i];
        if !shared && (self.output[node.output] || self.readers[node.output].len() > 1) {
            return false;
        }
        match (&node.primitive, node.inputs.as_slice()) {
            (Primitive::Reshape { .. }, _) => true,
            (p, [_]) => elementwise(p),
            (p, &[a, b]) => elementwise(p) && (a == b || self.splat(a) != self.splat(b)),
            _ => false,
        }
    }

    /// The non-splat operand of node `i`.
    fn through(&self, i: usize) -> Var {
        let inputs = &self.graph.nodes()[i].inputs;
        match inputs.len() > 1 && self.splat(inputs[0]) {
            true => inputs[1],
            false => inputs[0],
        }
    }

    /// Walks back from `v` through trivial nodes to one matching `is`; returns it and the path.
    fn edge(
        &self,
        mut v: Var,
        is: fn(&Primitive) -> bool,
        shared: bool,
    ) -> Option<(usize, Vec<usize>)> {
        let mut path = Vec::new();
        loop {
            let i = self.producer[v]?;
            if is(&self.graph.nodes()[i].primitive) {
                return Some((i, path));
            }
            if !self.trivial(i, shared) {
                return None;
            }
            path.push(i);
            v = self.through(i);
        }
    }

    /// The row rooted at elementwise node `i` combining a value with a broadcast of its reduction,
    /// and the value both sides derive from.
    fn diamond(&self, i: usize) -> Option<(Row, Var)> {
        let nodes = self.graph.nodes();
        let root = &nodes[i];
        let (&[a, b], p) = (root.inputs.as_slice(), &root.primitive) else {
            return None;
        };
        if !elementwise(p) {
            return None;
        }
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

    /// Matches `side op broadcast(reduce(x))` at node `i` where `side` and `x` derive from the same value
    /// through trivial ops; also absorbs single-use elementwise producers of that value.
    fn closed(&self, i: usize, side: Var, reduced: Var) -> Option<(Row, Var)> {
        let nodes = self.graph.nodes();
        let ty = self.graph.type_of(nodes[i].output);
        let (b, mut inner) = self.edge(
            reduced,
            |p| matches!(p, Primitive::BroadcastInDim { .. }),
            false,
        )?;
        let (r, path) = self.edge(
            nodes[b].inputs[0],
            |p| matches!(p, Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }),
            true,
        )?;
        let per_row: Vec<usize> = std::iter::once(r).chain(path.iter().copied()).collect();
        inner.push(b);
        inner.extend(path);
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
            && broadcast_dimensions
                .iter()
                .enumerate()
                .all(|(k, &d)| d != last || self.graph.type_of(nodes[b].inputs[0]).shape[k] == 1)
            && self.graph.type_of(nodes[b].inputs[0]).numel() == rows;
        if !along {
            return None;
        }
        let mut producer = x;
        while let Some(k) = self.producer[producer].filter(|&k| self.trivial(k, false)) {
            inner.push(k);
            producer = self.through(k);
        }
        let mut v = side;
        while v != producer {
            let k = self.producer[v].filter(|&k| self.trivial(k, false))?;
            inner.push(k);
            v = self.through(k);
        }
        inner.push(r);
        // Pull in elementwise ops above the shared value that only the row reads.
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
        let outputs = self.outputs(i, &inner, &per_row, b)?;
        let row = Row {
            root: i,
            reductions: vec![r],
            inner,
            outputs,
            partials: Vec::new(),
        };
        Some((row, producer))
    }

    /// Row nodes whose values are needed outside the row; `None` when one is read before the root
    /// or the broadcast is read by something that can't fuse.
    fn outputs(
        &self,
        root: usize,
        inner: &[usize],
        per_row: &[usize],
        b: usize,
    ) -> Option<Vec<usize>> {
        let nodes = self.graph.nodes();
        let mut inside = inner.to_vec();
        inside.push(root);
        let outside = |n: usize| -> Vec<usize> {
            let v = nodes[n].output;
            self.readers[v]
                .iter()
                .copied()
                .filter(|u| !inside.contains(u))
                .collect()
        };
        let mut outputs = Vec::new();
        for &n in per_row {
            let readers = outside(n);
            if self.output[nodes[n].output] || !readers.is_empty() {
                readers.iter().all(|&u| u > root).then_some(())?;
                outputs.push(n);
            }
        }
        let readers = outside(b);
        if self.output[nodes[b].output] {
            return None;
        }
        if !readers.is_empty() {
            let fused = |u: usize| u > root && (self.fusible)(self.graph, &nodes[u]);
            readers.iter().all(|&u| fused(u)).then_some(())?;
            let operand = self.producer[nodes[b].inputs[0]]?;
            if !outputs.contains(&operand) {
                outputs.push(operand);
            }
        }
        outputs.sort_unstable();
        Some(outputs)
    }
}
