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
//!     trivial     trivial → reduce (last dim) → trivial → broadcast
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
//!
//! A diamond's values of a row each (the reduction's, and the trivial
//! primitives' between it and the broadcast: an RMS norm's `sqrt(mean +
//! eps)`) may be read elsewhere too (a training step's backward): the row
//! kernel writes them as well, once a row (its `outputs`), as a fused RMS
//! norm writes each row's `rstd` for its backward. Its broadcast too, where
//! every other reader is a loop fusion's primitive: that fusion broadcasts
//! the written value itself.
//!
//! Beyond diamonds (XLA's rewriter stops there), any row fusion
//! ([`grown`]): an elementwise root of rows, and every fusible node before
//! it read by its nodes alone, each a value of the rows' shape or of a row
//! each (or a constant), reductions over a row among them, a reduction's
//! value broadcast back along its row: a normalization's backward
//! (`dx = g·w / rms - x · Σ_row(…)`, its row's values two, `g` and `x`,
//! not one producer's), a softmax's (`y · (g - Σ_row(g·y))`).

use super::fusion::{constant, elementwise, fusible};
use crate::graph::{Graph, Primitive, Var};
use crate::ops::reduce::mps::REDUCE_THREADS;

/// The most threadgroups a row kernel with partials runs (each a block of
/// rows): their partials, a column each, are summed by one thread a
/// column (`reduce_sum`'s, unsplit).
const MAX_GROUPS: usize = 64;

/// The fewest it runs, unless the rows are fewer: rows of no divisor in
/// between keep their sum's own reduction.
const MIN_GROUPS: usize = 16;

/// The most columns of a row each thread of a row kernel with partials
/// accumulates (in registers): rows of up to this x 256 elements.
const MAX_COLUMNS: usize = 16;

/// A row fusion: a chain of diamonds, run as one row kernel. `root` (a
/// node index) is the fusion's root, `reductions` its reductions (in
/// order), `inner` the other nodes between them and the root, which are
/// read nowhere else but `outputs`: values of a row each (of reductions,
/// or inner nodes) read after the root too, which the kernel writes.
/// `partials` ([`partials`]): sums of blocks of rows, each a column's
/// (`[groups, rows / groups, n]` over its middle), which the kernel
/// accumulates as it goes (a normalization's weight's gradient).
#[derive(Debug, Clone)]
pub(crate) struct Row {
    pub root: usize,
    pub reductions: Vec<usize>,
    pub inner: Vec<usize>,
    pub outputs: Vec<usize>,
    pub partials: Vec<usize>,
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
                row.outputs.splice(0..0, first.outputs);
            }
        }
        rows.push(row);
    }
    grown(&finder, &mut rows);
    consumed(&finder, &mut rows);
    rows
}

/// The row fusions beyond `rows` (diamonds): from each elementwise root of
/// rows (`[rows..., n]`) not in one, last first, every node before it read
/// by its nodes alone (no output), fusible, each a value of the root's
/// shape, of a row each, or one element; or a reduction over a row of the
/// root's shape (to a value of a row each); or a row fusion's root (read
/// by its nodes alone, writing nothing else), the row fusion inside it
/// whole (a chain: a layer norm's backward's two reductions). A diamond's
/// root (writing nothing else) grows so too, from its nodes. One with a
/// reduction is a row fusion: its value (of a row each) reaches the root
/// (of rows) through a broadcast back along the row.
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
        // A diamond's root grows from its row (writing nothing else).
        let own = rows
            .iter()
            .position(|r| r.root == i && r.outputs.is_empty());
        let free = !taken[i] || own.is_some();
        if !finder.live[i] || !free || !elementwise(&root.primitive) || !fusible(graph, root) {
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
        // From its rows (`rows` elements, `n` a row), the nodes inside (and
        // the row fusions it takes whole: its own, if a diamond's).
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
            // Values of a row each read after the root too (a cross
            // entropy's logsumexp, for its gradient): written as well.
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
                let read_after = !finder.output[v]
                    && per_row(v)
                    && finder.readers[v].iter().any(|&r| inside[r])
                    && finder.readers[v].iter().all(|&r| inside[r] || r > i);
                if !finder.live[k] || inside[k] || !(read_inside || read_after) {
                    continue;
                }
                // An earlier row fusion's root (of these rows, writing
                // nothing read but here: a softmax's max, its logsumexp's):
                // it, whole.
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
                if taken[k] || !fusible(graph, node) {
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
        // Of the root's rows; or, a value a row (a loss each), of the rows
        // a reduction it reads (through values a row) reduces.
        let mut grown = match ty.shape.last() {
            Some(&n) if n > 1 => Some(grow(ty.numel(), n)),
            _ => None,
        };
        // (Of several reductions: one, and its epilogue, is a reduction's
        // own fusion.)
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
        // A diamond that took nothing more stays as it is.
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

/// Each row fusion of `rows` extended to an elementwise consumer of its
/// values of the rows' shape after it, computed from them, from values of
/// its rows (a cross entropy's logits) and from values not reading its
/// own (`(exp(x - lse) - one_hot) · g`, its gradient, `g` a constant or an
/// input: XMA's forward-backward kernel): that consumer its root, its own
/// root and the values read elsewhere (of a row each) its outputs, written
/// once its reductions are done. Its readers before it wait for it (the
/// fused graph's order), as none of its values reads them.
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
        let taken = |k: usize| rows.iter().any(|o| o.root == k || o.inner.contains(&k) || o.reductions.contains(&k));
        let shaped = |v: Var| {
            let t = graph.type_of(v);
            t.numel() == size && t.shape.last() == Some(&n)
        };
        let fits = |v: Var| shaped(v) || [each, 1].contains(&graph.type_of(v).numel());
        let cheap = |p: &Primitive| {
            elementwise(p)
                || matches!(
                    p,
                    Primitive::Reshape { .. } | Primitive::BroadcastInDim { .. } | Primitive::Iota { .. }
                )
        };
        // The latest elementwise reader of its values of the rows' shape
        // that its cone (its producers outside the row, cheap and taken by
        // no other row) makes a fusion with it.
        let candidates: Vec<usize> = (row.root + 1..nodes.len())
            .rev()
            .filter(|&c| finder.live[c] && elementwise(&nodes[c].primitive) && fusible(graph, &nodes[c]) && shaped(nodes[c].output) && !taken(c))
            .collect();
        let found = candidates.into_iter().find_map(|c| {
            let mut cone: Vec<usize> = vec![c];
            let (mut leaves, mut stack, mut reaches) = (Vec::new(), nodes[c].inputs.clone(), false);
            while let Some(v) = stack.pop() {
                match finder.producer[v] {
                    Some(p) if mine.contains(&p) => reaches = true,
                    Some(p) if cone.contains(&p) => {}
                    Some(p) if !taken(p) && fusible(graph, &nodes[p]) && cheap(&nodes[p].primitive) && fits(v) => {
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
            // Its nodes read inside it alone (but its root).
            let inside = |u: usize| cone.contains(&u) || mine.contains(&u);
            let closed = cone[1..].iter().all(|&k| {
                let v = nodes[k].output;
                !finder.output[v] && finder.readers[v].iter().all(|&u| inside(u))
            });
            if !reaches || !closed {
                return None;
            }
            // The row's values read elsewhere (of a row each): its outputs;
            // nothing it reads computed from their readers.
            let mut outputs = Vec::new();
            let mut outside = Vec::new();
            for &k in &mine {
                let v = nodes[k].output;
                let readers: Vec<usize> = finder.readers[v].iter().copied().filter(|&u| !inside(u)).collect();
                if finder.output[v] || !readers.is_empty() {
                    if graph.type_of(v).numel() != each {
                        return None;
                    }
                    outputs.push(k);
                    outside.extend(readers);
                }
            }
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

/// The length of the rows a value a row (node `root`'s) is of: that of the
/// operand of a reduction over its last dimension to a value a row, read
/// back from it through elementwise primitives, reshapes and broadcasts
/// (a cross entropy's `logsumexp - picked`); None if none.
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

/// `graph` with each sum over rows (`Σ_rows v`: over every dimension but
/// the last) that a row fusion of `rows` can compute as it goes split in
/// two (Liger's and Apex's normalization backward, a persistent kernel):
/// the sums of `groups` blocks of rows, `reduce_sum(reshape(v, [groups,
/// rows / groups, n]), 1)`, that row fusion's `partials` (its kernel a
/// threadgroup a block, accumulating each column's sum in registers as
/// it goes, rows in order), then their sum (`reduce_sum(·, 0)`, a column
/// at a time, blocks in order): deterministic. The rows, of the graph
/// returned.
///
/// A sum is a row fusion's whose value reads (back through fusible nodes
/// no row fusion computes, which its kernel computes too) a value the row
/// fusion computes or reads (a normalization's weight's gradient, `Σ g·y`,
/// its `dx`'s `g`), never its root, nor one depending on it, and on which
/// nothing the row fusion computes depends.
pub(crate) fn partials(graph: &Graph, mut rows: Vec<Row>) -> (Graph, Vec<Row>) {
    let finder = Finder::new(graph);
    let nodes = graph.nodes();
    let mut row_of = vec![None; nodes.len()];
    for (r, row) in rows.iter().enumerate() {
        for &k in std::iter::once(&row.root)
            .chain(&row.inner)
            .chain(&row.reductions)
        {
            row_of[k] = Some(r);
        }
    }
    // Whether value `v` depends on a node `on` says, of index `floor` or
    // more (none before it can).
    let depends = |v: Var, on: &dyn Fn(usize) -> bool, floor: usize| {
        let (mut seen, mut stack) = (vec![false; nodes.len()], vec![v]);
        while let Some(v) = stack.pop() {
            let Some(k) = finder.producer[v].filter(|&k| k >= floor && !seen[k]) else {
                continue;
            };
            if on(k) {
                return true;
            }
            seen[k] = true;
            stack.extend(&nodes[k].inputs);
        }
        false
    };
    // Each sum's row fusion and its groups.
    let mut attach: Vec<Option<(usize, usize)>> = vec![None; nodes.len()];
    for (c, node) in nodes.iter().enumerate() {
        let Primitive::ReduceSum { axes, .. } = &node.primitive else {
            continue;
        };
        if !finder.live[c] || !fusible(graph, node) {
            continue;
        }
        let x = node.inputs[0];
        let ty = graph.type_of(x);
        let rank = ty.shape.len();
        let n = ty.shape.last().copied().unwrap_or(1);
        let over_rows = rank >= 2 && axes.iter().copied().eq(0..rank - 1);
        if !over_rows || n <= 1 || n.div_ceil(REDUCE_THREADS) > MAX_COLUMNS {
            continue;
        }
        let count = ty.numel() / n;
        let Some(groups) = (1..=MAX_GROUPS.min(count))
            .rev()
            .find(|&g| count.is_multiple_of(g))
            .filter(|&g| g >= MIN_GROUPS.min(count))
        else {
            continue;
        };
        // The values it reads: back from x through the fusible nodes no
        // row fusion computes.
        let (mut reads, mut stack) = (Vec::new(), vec![x]);
        while let Some(v) = stack.pop() {
            match finder.producer[v] {
                Some(k)
                    if row_of[k].is_none()
                        && fusible(graph, &nodes[k])
                        && !matches!(
                            nodes[k].primitive,
                            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
                        ) =>
                {
                    stack.extend(&nodes[k].inputs)
                }
                _ if !reads.contains(&v) => reads.push(v),
                _ => {}
            }
        }
        attach[c] = rows
            .iter()
            .enumerate()
            .find(|&(r, row)| {
                let in_row = |k: usize| row_of[k] == Some(r);
                let root = graph.type_of(nodes[row.root].output);
                let first = std::iter::once(row.root)
                    .chain(row.inner.iter().copied())
                    .chain(row.reductions.iter().copied())
                    .min()
                    .unwrap_or(row.root);
                let shares = reads.iter().any(|&v| {
                    finder.producer[v].is_some_and(in_row)
                        || finder.readers[v].iter().any(|&u| in_row(u))
                });
                // Its reads computed before the row fusion's kernel runs,
                // not after (nothing of it but what it recomputes).
                let before = reads.iter().all(|&v| match finder.producer[v] {
                    Some(k) if in_row(k) => k != row.root,
                    _ => !depends(v, &in_row, first),
                });
                // Nothing the row fusion computes depends on the sum.
                let after = std::iter::once(&row.root)
                    .chain(&row.inner)
                    .chain(&row.reductions)
                    .all(|&k| !nodes[k].inputs.iter().any(|&v| depends(v, &|k| k == c, c)));
                root.numel() == ty.numel()
                    && root.shape.last() == Some(&n)
                    && shares
                    && before
                    && after
            })
            .map(|(r, _)| (r, groups));
    }
    if attach.iter().all(Option::is_none) {
        return (graph.clone(), rows);
    }
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    let (mut index, mut added) = (vec![0; nodes.len()], Vec::new());
    for (i, node) in nodes.iter().enumerate() {
        out.set_origin(node);
        let mut inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let mut primitive = node.primitive.clone();
        if let Some((r, groups)) = attach[i] {
            let Primitive::ReduceSum { accum_dtype, .. } = node.primitive else {
                unreachable!("a sum")
            };
            let ty = out.type_of(inputs[0]);
            let n = *ty.shape.last().expect("a dimension");
            let new_sizes = vec![groups, ty.numel() / n / groups, n];
            let blocks = out.apply(Primitive::Reshape { new_sizes }, &inputs);
            let sum = |axes| Primitive::ReduceSum { axes, accum_dtype };
            let blocks = blocks.expect("a reshape of the rows");
            inputs = vec![out.apply(sum(vec![1]), &[blocks]).expect("their sums")];
            let at = out.nodes().len();
            added.push((r, at - 2, at - 1));
            primitive = sum(vec![0]);
        }
        index[i] = out.nodes().len();
        map[node.output] = out
            .apply(primitive, &inputs)
            .expect("a graph is typed as the original");
        out.set_label(map[node.output], node.label);
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs)
        .expect("outputs are values of the graph");
    for row in &mut rows {
        row.root = index[row.root];
        for k in row
            .inner
            .iter_mut()
            .chain(&mut row.reductions)
            .chain(&mut row.outputs)
        {
            *k = index[*k];
        }
    }
    for (r, reshape, partial) in added {
        rows[r].inner.push(reshape);
        rows[r].inner.sort_unstable();
        rows[r].partials.push(partial);
    }
    (out, rows)
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
    /// one reader (unless `shared`: a value of a row each, which the
    /// kernel can write for other readers), and a reshape, a unary
    /// elementwise primitive, or a binary one of one value or of one value
    /// and a splat.
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

    /// The operand a trivial node `i` is followed through: its first that
    /// is not a splat.
    fn through(&self, i: usize) -> Var {
        let inputs = &self.graph.nodes()[i].inputs;
        match inputs.len() > 1 && self.splat(inputs[0]) {
            true => inputs[1],
            false => inputs[0],
        }
    }

    /// From value `v` back through trivial nodes (XLA's `TrivialEdge`;
    /// `shared`: perhaps read elsewhere too) to one whose primitive passes
    /// `is`: it, and the trivial nodes passed.
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
        let (b, mut inner) = self.edge(
            reduced,
            |p| matches!(p, Primitive::BroadcastInDim { .. }),
            false,
        )?;
        // Values of a row each, which other nodes may read too.
        let (r, path) = self.edge(
            nodes[b].inputs[0],
            |p| matches!(p, Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }),
            true,
        )?;
        let per_row: Vec<usize> = std::iter::once(r).chain(path.iter().copied()).collect();
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
        if !along {
            return None;
        }
        // The producer: the reduction's operand back through trivial nodes.
        let mut producer = x;
        while let Some(k) = self.producer[producer].filter(|&k| self.trivial(k, false)) {
            inner.push(k);
            producer = self.through(k);
        }
        // The root's other operand reaches it trivially too.
        let mut v = side;
        while v != producer {
            let k = self.producer[v].filter(|&k| self.trivial(k, false))?;
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

    /// The values of a row each (`per_row`) that the row kernel rooted at
    /// `root` (`inner` inside it) writes too: those read elsewhere, after
    /// it (once it has written them), and the broadcast `b`'s operand if
    /// `b` is read elsewhere, by loop fusions' primitives alone (each
    /// broadcasting it itself); none if any is read otherwise.
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
            let fused = |u: usize| u > root && fusible(self.graph, &nodes[u]);
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
