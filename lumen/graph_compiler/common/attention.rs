//! Pattern matching of flash attention (forward and backward) in traced graphs, and the rewrite that
//! replaces each match with a single fusion kernel.

use std::collections::HashMap;

use crate::DType;
use crate::graph::{FUSION_SEPARATOR, Graph, Node, Primitive, Var};
use crate::graph_compiler::constant;
use crate::tensor::contiguous_strides;

/// How a kernel reads (or writes) a `[batch.., row, col]` operand as a strided view of `base`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Access {
    /// Graph value whose buffer holds the data.
    pub base: Var,
    /// Element offset of the view in `base`.
    pub offset: usize,
    /// `(stride, divisor)` per batch dim: index `i` addresses `i / divisor * stride`.
    pub batch: Vec<(usize, usize)>,
    /// Element stride of the row dim.
    pub row: usize,
    /// Element stride of the column dim.
    pub col: usize,
}

/// The batch extents as the attention kernels' grid launches them: the dims before the last merged into
/// one, and the last (the heads); 1 for an absent one.
pub(crate) fn grid_batch(batch: &[usize]) -> (usize, usize) {
    match batch.split_last() {
        Some((&heads, outer)) => (outer.iter().product(), heads),
        None => (1, 1),
    }
}

/// A term of an operand's element offset on the attention grid, each kernel writing it in its own syntax.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Term {
    /// The view's first element.
    Offset(usize),
    /// The outer batch index times this stride.
    Outer(usize),
    /// The head index, divided by `divisor` (heads sharing a KV head), times `stride`.
    Heads { stride: usize, divisor: usize },
}

impl Access {
    /// The non-zero batch terms of the access's element offset on the grid `grid_batch` gives `batch`: its
    /// offset, the merged outer dims' stride and the heads' stride with its divisor. Fails when the outer
    /// dims don't step as one dim (each over the whole of the one inside it) or carry a divisor.
    pub(crate) fn batch_terms(&self, batch: &[usize]) -> Result<Vec<Term>, String> {
        let mut terms = Vec::new();
        if self.offset != 0 {
            terms.push(Term::Offset(self.offset));
        }
        let Some((&(stride, divisor), outer)) = self.batch.split_last() else {
            return Ok(terms);
        };
        // The outer dims of more than one element, outermost first.
        let dims: Vec<(usize, (usize, usize))> = batch
            .iter()
            .copied()
            .zip(outer.iter().copied())
            .filter(|&(n, _)| n != 1)
            .collect();
        let steps = dims.windows(2).all(|w| w[0].1.0 == w[1].1.0 * w[1].0);
        if !steps || dims.iter().any(|&(_, (_, divisor))| divisor != 1) {
            return Err(
                "flash attention: batch dims before the heads that don't step as one dim".into(),
            );
        }
        if let Some(&(_, (outer, _))) = dims.last().filter(|&&(_, (s, _))| s != 0) {
            terms.push(Term::Outer(outer));
        }
        if stride != 0 {
            terms.push(Term::Heads { stride, divisor });
        }
        Ok(terms)
    }
}

/// A matched `softmax(scores(Q K^T)) V` attention.
#[derive(Debug, Clone)]
pub(crate) struct Attention {
    /// Node whose output is the attention as written (the second dot or its trailing transposes).
    pub root: usize,
    pub q: Access,
    pub k: Access,
    pub v: Access,
    /// Where the attention output is written.
    pub out: Access,
    /// Batch extents (leading dims of the scores).
    pub batch: Vec<usize>,
    /// Query count.
    pub sq: usize,
    /// Key count.
    pub sk: usize,
    /// Head dim of Q and K.
    pub h: usize,
    /// Head dim of V.
    pub hv: usize,
    /// Ops applied to the raw `Q K^T` before softmax, in order.
    pub scores: Vec<Score>,
    /// Causal diagonal: key `k` is kept for query `q` when `k <= q + causal (+ offset)`.
    pub causal: Option<i64>,
    /// Runtime scalar added to the causal diagonal.
    pub offset: Option<Var>,
    /// Node computing `max + log(sum)` when the graph also reads the logsumexp.
    pub lse: Option<usize>,
    /// The `Q K^T` dot node.
    pub dot1: usize,
    /// The `P V` dot node.
    pub dot2: usize,
    /// Row max of the scores.
    pub max: Var,
    /// Row sum of the exponentials.
    pub sum: Var,
    /// Dtype of Q, K and V.
    pub dtype: DType,
    /// Dtype of the written output.
    pub out_dtype: DType,
    pub dropout: Option<Dropout>,
}

/// Dropout on the attention probabilities, from `random_bits` compared against a threshold.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Dropout {
    /// Seed operand of `random_bits`.
    pub seed: Var,
    /// Counter start operand of `random_bits`.
    pub start: Var,
    /// Static counter offset of `random_bits`.
    pub offset: u64,
    /// Probabilities whose random bits are below this are dropped.
    pub threshold: u32,
    /// Scale applied to the kept probabilities.
    pub scale: f64,
    /// The boolean drop mask value in the graph.
    pub mask: Var,
}

/// One op applied to the attention scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Score {
    /// Rounds to the dtype (a cast).
    Round(DType),
    /// Multiplies by a constant, computing in the dtype.
    Mul(f64, DType),
}

/// A matched flash-attention backward: gradients of Q, K and V from the transposed scores `S^T`.
#[derive(Debug, Clone)]
pub(crate) struct Backward {
    /// Node writing dV and where it writes, when dV is computed.
    pub dv_out: Option<(usize, Access)>,
    /// Node writing dK and where it writes, when dK is computed.
    pub dk_out: Option<(usize, Access)>,
    /// Node writing dQ and where it writes, when dQ is computed.
    pub dq_out: Option<(usize, Access)>,
    pub q: Access,
    pub k: Access,
    pub v: Access,
    /// Gradient of the attention output.
    pub d_o: Access,
    /// Per-query logsumexp saved by the forward pass.
    pub lse: Access,
    /// Per-query `rowsum(dO * O)`.
    pub delta: Access,
    pub batch: Vec<usize>,
    pub sq: usize,
    pub sk: usize,
    pub h: usize,
    pub hv: usize,
    pub scores: Vec<Score>,
    pub causal: Option<i64>,
    /// Constant scale applied to dS.
    pub ds_scale: f64,
    pub dtype: DType,
    pub dropout: Option<Dropout>,
}

/// Largest head dim the backward kernel supports.
pub(crate) const MAX_BACKWARD_HEAD: usize = 128;

/// The first attention backward found in a fusion body.
pub(crate) fn backward_of_body(body: &Graph) -> Option<Backward> {
    backward_of_body_into(body, HashMap::new())
}

/// The first attention backward in `body`, where `views` gives `(offset, strides)` of body outputs
/// (by output index) that are written into a view of a larger buffer.
pub(crate) fn backward_of_body_into(
    body: &Graph,
    views: HashMap<usize, (usize, Vec<usize>)>,
) -> Option<Backward> {
    let strides = Strides {
        outputs: views,
        ..Strides::default()
    };
    backward_of_body_strided(body, &strides)
}

/// Where a fusion body's operands live: element strides of its inputs read through views and
/// `(offset, strides)` of its outputs written into views, by input and output index; others are
/// contiguous in their own buffers.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Strides {
    pub(crate) inputs: HashMap<usize, Vec<usize>>,
    pub(crate) outputs: HashMap<usize, (usize, Vec<usize>)>,
}

impl Strides {
    /// The layouts of `body`'s inputs and outputs as `inputs` and `outputs` strides give them: an
    /// input or output with contiguous strides (and no offset) is left out.
    pub(crate) fn of(body: &Graph, inputs: &[Vec<usize>], outputs: &[(usize, Vec<usize>)]) -> Self {
        let contiguous =
            |v: Var, strides: &[usize]| contiguous_strides(&body.type_of(v).shape) == strides;
        Strides {
            inputs: body
                .inputs()
                .iter()
                .zip(inputs)
                .enumerate()
                .filter(|(_, (v, s))| !contiguous(**v, s))
                .map(|(k, (_, s))| (k, s.clone()))
                .collect(),
            outputs: body
                .outputs()
                .iter()
                .zip(outputs)
                .enumerate()
                .filter(|(_, (v, (o, s)))| *o != 0 || !contiguous(**v, s))
                .map(|(k, (_, view))| (k, view.clone()))
                .collect(),
        }
    }
}

/// The first attention backward in `body`, its operands laid out as `strides` gives them.
pub(crate) fn backward_of_body_strided(body: &Graph, strides: &Strides) -> Option<Backward> {
    let m = Matcher::new(body).laid_out(strides);
    (0..body.nodes().len()).find_map(|i| m.backward(i))
}

/// Extracts the subgraph computing `outputs` from `bases` as a standalone body with `bases` as inputs.
pub(crate) fn body_of(
    graph: &Graph,
    producer: &[Option<usize>],
    outputs: &[Var],
    bases: &[Var],
) -> Graph {
    let nodes = graph.nodes();
    let mut members = Vec::new();
    let mut stack: Vec<Var> = outputs.to_vec();
    while let Some(v) = stack.pop() {
        if bases.contains(&v) {
            continue;
        }
        let n = producer[v].expect("a value the fusion computes");
        if !members.contains(&n) {
            members.push(n);
            stack.extend(&nodes[n].inputs);
        }
    }
    members.sort_unstable();
    let mut body = Graph::new();
    let mut var: HashMap<Var, Var> = HashMap::new();
    for &b in bases {
        var.insert(b, body.input(graph.type_of(b).clone()));
    }
    for &n in &members {
        let node = &nodes[n];
        let ins: Vec<Var> = node.inputs.iter().map(|v| var[v]).collect();
        body.set_origin(node);
        let v = body
            .apply(node.primitive.clone(), &ins)
            .expect("a node of the graph");
        var.insert(node.output, v);
    }
    let outputs: Vec<Var> = outputs.iter().map(|v| var[v]).collect();
    body.set_outputs(&outputs).expect("the fusion's outputs");
    body
}

/// Replaces each matched attention backward with fusion kernels named by `kernel`: one computing
/// dK, dV and dQ when `together` says so and the order allows, otherwise one for dK/dV and one for dQ.
pub(crate) fn fuse_backward(
    graph: &Graph,
    together: impl Fn(&Backward) -> bool,
    mut kernel: impl FnMut(&Graph) -> String,
) -> Graph {
    let m = Matcher::new(graph);
    let mut plain = Matcher::new(graph);
    plain.plain_dq = true;
    let nodes = graph.nodes();
    let mut fusions: Vec<(usize, Vec<usize>, Vec<Var>)> = Vec::new();
    for i in 0..nodes.len() {
        let Some(b) = m.backward(i) else { continue };
        let together = together(&b);
        let Some(b) = (if together { plain.backward(i) } else { Some(b) }) else {
            continue;
        };
        let (Some((dv, _)), Some((dk, _)), Some((dq, _))) = (&b.dv_out, &b.dk_out, &b.dq_out)
        else {
            continue;
        };
        let (dv, dk, dq) = (*dv, *dk, *dq);
        let mut bases: Vec<Var> = Vec::new();
        for acc in [&b.q, &b.k, &b.v, &b.d_o, &b.lse, &b.delta] {
            if !bases.contains(&acc.base) {
                bases.push(acc.base);
            }
        }
        if let Some(d) = &b.dropout {
            bases.extend([d.seed, d.start]);
        }
        // A fusion goes after its last operand is produced and before any of its outputs is read.
        let ready = bases.iter().filter_map(|&v| m.producer[v]).max();
        let read_after = |outs: &[usize]| {
            let first = *outs.iter().min().expect("a dot");
            let at = ready.map_or(first, |r| r.max(first));
            outs.iter()
                .all(|&o| m.readers[nodes[o].output].iter().all(|&r| r > at))
                .then_some(at)
        };
        let all = vec![dv, dk, dq];
        let groups = match (together, read_after(&all)) {
            (true, Some(_)) => vec![all],
            _ => vec![vec![dv, dk], vec![dq]],
        };
        for outs in groups {
            if let Some(at) = read_after(&outs) {
                fusions.push((at, outs, bases.clone()));
            }
        }
    }
    if fusions.is_empty() {
        return graph.clone();
    }
    // Rebuild the graph, dropping the fused dots and placing each fusion at its insertion point.
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for (i, node) in nodes.iter().enumerate() {
        out.set_origin(node);
        if !fusions.iter().any(|(_, outs, _)| outs.contains(&i)) {
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("a node of the graph");
            out.set_label(map[node.output], node.label);
        }
        for (_, outs, bases) in fusions.iter().filter(|(at, _, _)| *at == i) {
            let outputs: Vec<Var> = outs.iter().map(|&o| nodes[o].output).collect();
            let body = body_of(graph, &m.producer, &outputs, bases);
            let name = kernel(&body);
            let label = match outs.len() {
                3 => "flash_attention_backward(dk, dv, dq)",
                2 => "flash_attention_backward(dk, dv)",
                _ => "flash_attention_backward(dq)",
            };
            let fusion = Primitive::Fusion { name, label, body };
            let reads: Vec<Var> = bases.iter().map(|&b| map[b]).collect();
            let first = out
                .apply(fusion, &reads)
                .expect("a fusion typed as its first output");
            map[outputs[0]] = first;
            for (index, &o) in outputs.iter().enumerate().skip(1) {
                let ty = graph.type_of(o).clone();
                map[o] = out
                    .apply(Primitive::FusionOutput { index, ty }, &[first])
                    .expect("the fusion's output");
            }
        }
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Whether every element offset of a `rows x cols` view fits in 32 bits.
fn fits32(acc: &Access, rows: usize, cols: usize) -> bool {
    (acc.row * (rows - 1) + acc.col * (cols - 1)) < (1 << 32)
}

/// Largest head dim the forward kernel supports.
pub(crate) const MAX_HEAD: usize = 256;

/// Every live forward attention in `graph`; `runtime_offsets` allows a runtime causal offset.
pub(crate) fn attentions(graph: &Graph, runtime_offsets: bool) -> Vec<Attention> {
    attentions_in(Matcher::new(graph), runtime_offsets)
}

/// Every live forward attention `m`'s graph holds.
fn attentions_in(m: Matcher<'_>, runtime_offsets: bool) -> Vec<Attention> {
    let graph = m.graph;
    let m = Matcher {
        runtime_offsets,
        ..m
    };
    let mut found: Vec<Attention> = Vec::new();
    for (i, node) in graph.nodes().iter().enumerate() {
        let dot = m.live[i] && matches!(node.primitive, Primitive::DotGeneral { .. });
        if let Some(a) = dot.then(|| m.attention(i)).flatten() {
            found.push(a);
        }
    }
    found
}

/// Each attention traced in `graph` with the values computed between its operands and output.
pub(crate) fn traced(graph: &Graph) -> Vec<(Attention, Vec<Var>)> {
    let nodes = graph.nodes();
    let mut producer = vec![None; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    attentions(graph, false)
        .into_iter()
        .map(|a| {
            let (d1, d2) = (&nodes[a.dot1], &nodes[a.dot2]);
            let operands = [d1.inputs[0], d1.inputs[1], d2.inputs[1]];
            let mut members: Vec<Var> = Vec::new();
            let mut stack = vec![d2.output];
            while let Some(v) = stack.pop() {
                if operands.contains(&v) || members.contains(&v) {
                    continue;
                }
                let Some(n) = producer[v] else { continue };
                members.push(v);
                stack.extend(&nodes[n].inputs);
            }
            members.sort_unstable();
            (a, members)
        })
        .collect()
}

/// The attention in a fusion body, allowing a runtime causal offset.
pub(crate) fn of_body(body: &Graph) -> Option<Attention> {
    attentions(body, true).into_iter().next()
}

/// The attention in a fusion body whose operands are laid out as `strides` gives them.
pub(crate) fn of_body_strided(body: &Graph, strides: &Strides) -> Option<Attention> {
    attentions_in(Matcher::new(body).laid_out(strides), true)
        .into_iter()
        .next()
}

/// An attention to fuse: the match, its epilogue chain, the node the fusion replaces, the epilogue's
/// extra inputs, and whether the raw attention value is also kept as an output.
type Fused = (Attention, Vec<usize>, usize, Vec<Var>, bool);

/// Whether the kernel can read an epilogue input.
pub(crate) type Leaves = fn(&Graph, Var) -> bool;

/// Which elementwise epilogues may be fused into the attention kernel.
#[derive(Clone, Copy)]
pub(crate) struct Epilogues {
    pub(crate) leaves: Leaves,
    /// Only fuse chains that do more than reshape, keeping the value before the trailing reshapes.
    pub(crate) reshaped: bool,
    /// Write the attention through its one transpose though others read it too (they read it transposed
    /// back): a backend whose kernels load and store strided.
    pub(crate) transposed: bool,
}

/// Replaces each forward attention, with its elementwise epilogue when `epilogues` allows, by a fusion
/// kernel named by `kernel`. `scalars` are values passed by value, which the fusion can't read.
pub(crate) fn fuse(
    graph: &Graph,
    epilogues: Option<Epilogues>,
    scalars: &[Var],
    runtime_offsets: bool,
    mut kernel: impl FnMut(&Graph) -> String,
) -> Graph {
    let m = Matcher::new(graph);
    let found: Vec<Fused> = attentions(graph, runtime_offsets)
        .into_iter()
        // A runtime causal offset must be a buffer the kernel can read.
        .filter(|a| a.offset.is_none_or(|v| !scalars.contains(&v)))
        .map(|a| {
            let plan = |a: Attention| {
                let (mut chain, mut leaves, mut kept) = match epilogues {
                    Some(e) => m.epilogue(a.root, scalars, e.leaves, e.reshaped),
                    None => (Vec::new(), Vec::new(), false),
                };
                // Drop the epilogue when the raw attention or the logsumexp is read before the chain ends.
                let read_after = |v: Var, end: usize| {
                    m.readers[v].iter().all(|&r| r > end || chain.contains(&r))
                };
                let end = chain.last().copied().unwrap_or(a.root);
                let root_read = !kept || read_after(graph.nodes()[a.root].output, end);
                let lse_read = a
                    .lse
                    .is_none_or(|n| read_after(graph.nodes()[n].output, end));
                if !root_read || !lse_read {
                    (chain, leaves, kept) = (Vec::new(), Vec::new(), false);
                }
                let end = chain.last().copied().unwrap_or(a.root);
                (a, chain, end, leaves, kept)
            };
            // Written through the dot's one transpose, its other readers read it transposed back: they
            // must come after the fusion, and the chain must leave the written value to transpose.
            let nodes = graph.nodes();
            let dot = nodes[a.dot2].output;
            let through = epilogues.is_some_and(|e| e.transposed) && a.root == a.dot2;
            let shared = through
                .then(|| m.written(a.dot2, a.batch.len(), false, true))
                .flatten()
                .filter(|&(root, _)| root != a.dot2);
            if let Some((root, out)) = shared {
                let out_dtype = graph.type_of(nodes[root].output).dtype;
                let (t, chain, end, leaves, kept) = plan(Attention {
                    root,
                    out,
                    out_dtype,
                    ..a.clone()
                });
                let reshapes = chain
                    .iter()
                    .all(|&n| matches!(nodes[n].primitive, Primitive::Reshape { .. }));
                let after = m.readers[dot].iter().all(|&r| r == root || r > end);
                if after && (kept || reshapes) {
                    return (t, chain, end, leaves, kept);
                }
            }
            plan(a)
        })
        .filter(|(a, _, end, _, _)| {
            a.lse
                .is_none_or(|n| m.readers[graph.nodes()[n].output].iter().all(|r| r > end))
        })
        .collect();
    if found.is_empty() {
        return graph.clone();
    }
    let nodes = graph.nodes();
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for (i, node) in nodes.iter().enumerate() {
        out.set_origin(node);
        // The logsumexp is a fusion output, so its node goes unless it is read before the fusion.
        if found
            .iter()
            .any(|(a, _, end, _, _)| a.lse == Some(i) && i > *end)
        {
            continue;
        }
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let Some((a, chain, _, leaves, kept)) = found.iter().find(|(_, _, end, _, _)| *end == i)
        else {
            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("a node of the graph");
            out.set_label(map[node.output], node.label);
            continue;
        };
        let mut bases: Vec<Var> = Vec::new();
        for &b in [a.q.base, a.k.base, a.v.base]
            .iter()
            .chain(leaves)
            .chain(&a.offset)
        {
            if !bases.contains(&b) {
                bases.push(b);
            }
        }
        if let Some(d) = &a.dropout {
            bases.extend([d.seed, d.start]);
        }
        let root = nodes[a.root].output;
        let reshapes = chain
            .iter()
            .take_while(|&&n| matches!(nodes[n].primitive, Primitive::Reshape { .. }))
            .count();
        let reshaped =
            epilogues.is_some_and(|e| e.reshaped) && reshapes > 0 && reshapes < chain.len();
        let raw = kept.then(|| match reshaped {
            true => nodes[chain[reshapes - 1]].output,
            false => root,
        });
        let lse = a.lse.map(|n| nodes[n].output);
        let outputs: Vec<Var> = std::iter::once(node.output).chain(raw).chain(lse).collect();
        let body = body_of(graph, &m.producer, &outputs, &bases);
        let name = kernel(&body);
        let label =
            std::iter::once("flash_attention").chain(chain.iter().map(|&n| graph.label(&nodes[n])));
        let label = crate::graph::intern(label.collect::<Vec<_>>().join(FUSION_SEPARATOR));
        let fusion = Primitive::Fusion { name, label, body };
        out.set_origin(&nodes[a.root]);
        let reads: Vec<Var> = bases.iter().map(|&b| map[b]).collect();
        map[node.output] = out
            .apply(fusion, &reads)
            .expect("a fusion typed as the attention");
        let fused = map[node.output];
        for (index, &v) in outputs.iter().enumerate().skip(1) {
            let ty = graph.type_of(v).clone();
            map[v] = out
                .apply(Primitive::FusionOutput { index, ty }, &[fused])
                .expect("the fusion's output");
        }
        // Restore the attention value in its original shape for its other readers.
        if let Some(r) = raw.filter(|&r| r != root) {
            let new_sizes = graph.type_of(root).shape.clone();
            map[root] = out
                .apply(Primitive::Reshape { new_sizes }, &[map[r]])
                .expect("the attention's value");
        }
        // Written through its transpose, the attention's other readers read it transposed back.
        if a.root != a.dot2 {
            if raw.is_none() && i != a.root {
                let new_sizes = graph.type_of(root).shape.clone();
                map[root] = out
                    .apply(Primitive::Reshape { new_sizes }, &[map[node.output]])
                    .expect("the attention's value");
            }
            let Primitive::Transpose { permutation } = &nodes[a.root].primitive else {
                unreachable!("the attention's one transpose")
            };
            let mut inverse = vec![0; permutation.len()];
            for (d, &p) in permutation.iter().enumerate() {
                inverse[p] = d;
            }
            map[nodes[a.dot2].output] = out
                .apply(
                    Primitive::Transpose {
                        permutation: inverse,
                    },
                    &[map[root]],
                )
                .expect("the attention's value, heads first");
        }
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Graph indices used to match attention patterns.
struct Matcher<'a> {
    graph: &'a Graph,
    /// Node producing each value.
    producer: Vec<Option<usize>>,
    /// Live nodes reading each value, in order.
    readers: Vec<Vec<usize>>,
    /// Whether each value is a graph output.
    output: Vec<bool>,
    /// Whether each node feeds a graph output (all nodes while still tracing).
    live: Vec<bool>,
    /// Match dQ only as the dot or a single transpose of it (for a kernel writing all three gradients).
    plain_dq: bool,
    /// Allow a causal offset from a rank-0 integer input.
    runtime_offsets: bool,
    /// Body outputs written into a view: `(offset, strides)` of their buffer.
    views: HashMap<Var, (usize, Vec<usize>)>,
    /// Body inputs read through a view: their element strides.
    strided: HashMap<Var, Vec<usize>>,
}

impl<'a> Matcher<'a> {
    /// Follows the single-reader elementwise/reshape chain after `root` whose other operands are
    /// readable leaves. Returns the chain's nodes, its leaves, and whether `root` has other readers.
    fn epilogue(
        &self,
        root: usize,
        scalars: &[Var],
        readable: Leaves,
        reshaped: bool,
    ) -> (Vec<usize>, Vec<Var>, bool) {
        use Primitive::*;
        let nodes = self.graph.nodes();
        let (mut end, mut chain, mut leaves) = (root, Vec::new(), Vec::new());
        let mut kept = false;
        loop {
            let v = nodes[end].output;
            let u = match (&self.readers[v][..], end == root) {
                // The root may have other readers: then the raw value is kept as a fusion output.
                ([u], _) if !self.output[v] => *u,
                ([u, ..], true) => {
                    kept = true;
                    *u
                }
                _ => break,
            };
            let n = &nodes[u];
            let step = crate::graph_compiler::elementwise(&n.primitive)
                || matches!(n.primitive, Reshape { .. });
            if !step {
                break;
            }
            let mut found = Vec::new();
            let others = n.inputs.iter().filter(|&&x| x != v);
            if !others
                .into_iter()
                .all(|&x| self.leaves(x, scalars, &mut found))
                || !found.iter().all(|&leaf| readable(self.graph, leaf))
            {
                break;
            }
            for leaf in found {
                if !leaves.contains(&leaf) {
                    leaves.push(leaf);
                }
            }
            end = u;
            chain.push(u);
        }
        let layout = chain
            .iter()
            .all(|&n| matches!(nodes[n].primitive, Reshape { .. }));
        if reshaped && layout {
            return (Vec::new(), Vec::new(), false);
        }
        let kept = kept && !chain.is_empty();
        (chain, leaves, kept)
    }

    /// Collects the values `v` reads, looking through constants, broadcasts, reshapes and casts; false
    /// when one is a by-value scalar input.
    fn leaves(&self, v: Var, scalars: &[Var], leaves: &mut Vec<Var>) -> bool {
        use Primitive::*;
        let Some(n) = self.node(v) else {
            leaves.push(v);
            return !scalars.contains(&v);
        };
        match n.primitive {
            Full { .. } | Iota { .. } => true,
            BroadcastInDim { .. } | Reshape { .. } | Cast { .. } => {
                self.leaves(n.inputs[0], scalars, leaves)
            }
            _ => {
                leaves.push(v);
                true
            }
        }
    }

    /// Matcher over `graph` with producers, readers and liveness computed.
    fn new(graph: &'a Graph) -> Self {
        let nodes = graph.nodes();
        let n = graph.types.len();
        let mut producer = vec![None; n];
        for (i, node) in nodes.iter().enumerate() {
            producer[node.output] = Some(i);
        }
        let mut output = vec![false; n];
        graph.outputs().iter().for_each(|&v| output[v] = true);
        let tracing = graph.outputs().is_empty();
        let mut live = vec![tracing; nodes.len()];
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
        Matcher {
            plain_dq: false,
            runtime_offsets: false,
            views: HashMap::new(),
            strided: HashMap::new(),
            graph,
            producer,
            readers,
            output,
            live,
        }
    }

    /// The matcher with the body's inputs and outputs laid out as `strides` gives them.
    fn laid_out(mut self, strides: &Strides) -> Self {
        let graph = self.graph;
        self.views = strides
            .outputs
            .iter()
            .map(|(&k, view)| (graph.outputs()[k], view.clone()))
            .collect();
        self.strided = strides
            .inputs
            .iter()
            .map(|(&k, s)| (graph.inputs()[k], s.clone()))
            .collect();
        self
    }

    /// Node producing `v`, if any.
    fn node(&self, v: Var) -> Option<&'a Node> {
        self.producer[v].map(|i| &self.graph.nodes()[i])
    }

    /// Whether `v` has exactly one reader and isn't a graph output.
    fn single(&self, v: Var) -> bool {
        !self.output[v] && self.readers[v].len() == 1
    }

    /// `v` with a single-use cast removed.
    fn uncast(&self, v: Var) -> Var {
        match self.node(v) {
            Some(n) if matches!(n.primitive, Primitive::Cast { .. }) && self.single(v) => {
                n.inputs[0]
            }
            _ => v,
        }
    }

    /// `v` with single-use broadcasts and reshapes removed; `shared` may have several readers.
    fn unbroadcast(&self, mut v: Var, shared: Option<Var>) -> Var {
        while let Some(n) = self.node(v) {
            match n.primitive {
                Primitive::BroadcastInDim { .. } | Primitive::Reshape { .. }
                    if self.single(v) || shared == Some(v) =>
                {
                    v = n.inputs[0]
                }
                _ => break,
            }
        }
        v
    }

    /// The constant `v` holds when it is a broadcast, reshape or cast of a `full`.
    fn scalar(&self, v: Var) -> Option<f64> {
        if !constant(self.graph, &self.producer, v) {
            return None;
        }
        let mut v = v;
        loop {
            let n = self.node(v)?;
            match &n.primitive {
                Primitive::Full { fill_value, .. } => return Some(fill_value.to_f64()),
                Primitive::BroadcastInDim { .. }
                | Primitive::Reshape { .. }
                | Primitive::Cast { .. } => v = n.inputs[0],
                _ => return None,
            }
        }
    }

    /// Matches `select(mask, 0, x * scale)` dropout on a `shape`d tensor. Returns `x`, the dropout,
    /// and the dtype the scaling computes in.
    fn dropout(&self, v: Var, shape: &[usize], transposed: bool) -> Option<(Var, Dropout, DType)> {
        use Primitive::*;
        let n = self
            .node(v)
            .filter(|n| matches!(n.primitive, Select) && self.single(v))?;
        if self.scalar(n.inputs[1]) != Some(0.0) {
            return None;
        }
        let m = n.inputs[2];
        let mul = self
            .node(m)
            .filter(|n| matches!(n.primitive, Mul) && self.single(m))?;
        let (a, b) = (mul.inputs[0], mul.inputs[1]);
        let (scale, x) = match self.scalar(b) {
            Some(c) => (c, a),
            None => (self.scalar(a)?, b),
        };
        let mut dropout = self.mask(n.inputs[0], shape, transposed)?;
        dropout.scale = scale;
        Some((x, dropout, self.graph.type_of(m).dtype))
    }

    /// Matches a drop mask `random_bits < threshold` of `shape`, transposed in its last two dims when
    /// `transposed`.
    fn mask(&self, v: Var, shape: &[usize], transposed: bool) -> Option<Dropout> {
        use Primitive::*;
        let rank = shape.len();
        let swaps = |p: &[usize]| {
            let mut swapped: Vec<usize> = (0..rank).collect();
            swapped.swap(rank - 2, rank - 1);
            p == swapped
        };
        let (mut x, mut flipped, mut threshold) = (v, false, None);
        loop {
            let n = self.node(x)?;
            match &n.primitive {
                Transpose { permutation } if swaps(permutation) => flipped = !flipped,
                Lt if threshold.is_none() => threshold = Some(self.scalar(n.inputs[1])?),
                RandomBits { shape: s, offset } => {
                    let t = threshold?;
                    let whole = t.fract() == 0.0 && (0.0..4294967296.0).contains(&t);
                    return (s == shape && flipped == transposed && whole).then(|| Dropout {
                        seed: n.inputs[0],
                        start: n.inputs[1],
                        offset: *offset,
                        threshold: t as u32,
                        scale: 1.0,
                        mask: v,
                    });
                }
                _ => return None,
            }
            x = n.inputs[0];
        }
    }

    /// Matches the attention backward around `exp` node `i`, computing `P^T = exp(S^T - lse)`, with
    /// `dV = P^T dO`, `dS = P^T * (V dO^T - delta) * scale`, `dK = dS Q` and `dQ = dS^T K`.
    fn backward(&self, i: usize) -> Option<Backward> {
        use Primitive::*;
        let nodes = self.graph.nodes();
        let ty = |v: Var| self.graph.type_of(v);
        let dot = |n: usize| match &nodes[n].primitive {
            DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                accum_dtype,
                output_dtype,
            } => Some((
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                *accum_dtype,
                *output_dtype,
            )),
            _ => None,
        };
        let f32 = DType::F32;
        if !matches!(nodes[i].primitive, Exp) || !self.live[i] {
            return None;
        }
        let pt = nodes[i].output;
        let sub = self
            .node(nodes[i].inputs[0])
            .filter(|n| matches!(n.primitive, Sub) && self.single(nodes[i].inputs[0]))?;
        let (x, lse_b) = (sub.inputs[0], sub.inputs[1]);
        let nb = ty(x).shape.len().checked_sub(2)?;
        let (s, scores, causal) = self.scores(x, nb, true)?;
        let causal = causal.map(|(c, _)| c);
        let (lc1, rc1, lb1, rb1, acc1, _) = dot(s)?;
        let (k_, q_) = (nodes[s].inputs[0], nodes[s].inputs[1]);
        if lb1.len() != nb || lc1.len() != 1 || rc1.len() != 1 || acc1 != f32 || ty(x).dtype != f32
        {
            return None;
        }
        let free = |rank: usize, used: &[&[usize]]| {
            (0..rank).find(|d| used.iter().all(|u| !u.contains(d)))
        };
        let (kf, qf) = (
            free(ty(k_).shape.len(), &[lb1, lc1])?,
            free(ty(q_).shape.len(), &[rb1, rc1])?,
        );
        let dtype = ty(q_).dtype;
        let mut probabilities = ty(x).shape.clone();
        probabilities.swap(nb, nb + 1);
        let (mut m1, mut dv, mut dropout) = (None, None, None);
        let scaled = |r: usize| {
            nodes[r]
                .inputs
                .iter()
                .any(|&u| u != pt && self.scalar(u).is_some())
        };
        // Readers of P^T: the dV dot (possibly through dropout or a cast) and the dS product.
        for &r in &self.readers[pt] {
            match &nodes[r].primitive {
                Mul if dv.is_none() && scaled(r) => {
                    let [sel] = self.readers[nodes[r].output][..] else {
                        return None;
                    };
                    let p = nodes[sel].output;
                    let (x, d, t) = self.dropout(p, &probabilities, true)?;
                    let [c] = self.readers[p][..] else {
                        return None;
                    };
                    let dot = match &nodes[c].primitive {
                        Cast { .. } => match self.readers[nodes[c].output][..] {
                            [d] if self.single(nodes[c].output) => Some(d),
                            _ => None,
                        },
                        _ => Some(c),
                    }?;
                    let operand = nodes[dot].inputs[0];
                    (x == pt && t == f32 && (operand == p || operand == nodes[c].output))
                        .then_some(())?;
                    (dv, dropout) = (Some(dot), Some(d));
                }
                Mul if m1.is_none() => m1 = Some(r),
                Cast { .. } if dv.is_none() => {
                    let c = nodes[r].output;
                    let [d] = self.readers[c][..] else {
                        return None;
                    };
                    (self.single(c) && nodes[d].inputs[0] == c).then_some(())?;
                    dv = Some(d);
                }
                DotGeneral { .. } if dv.is_none() && nodes[r].inputs[0] == pt => dv = Some(r),
                _ => return None,
            }
        }
        if self.output[pt] {
            return None;
        }
        let m1 = &nodes[m1?];
        let other = *m1.inputs.iter().find(|&&v| v != pt)?;
        let sub2 = self
            .node(other)
            .filter(|n| matches!(n.primitive, Sub) && self.single(other))?;
        let (dpt, d_b) = (sub2.inputs[0], sub2.inputs[1]);
        let dpt = match self.dropout(dpt, &probabilities, true) {
            Some((x, d, t)) => {
                let same = |e: &Dropout| {
                    (e.seed, e.start, e.offset, e.threshold, e.scale)
                        == (d.seed, d.start, d.offset, d.threshold, d.scale)
                };
                (t == f32 && dropout.as_ref().is_none_or(same)).then_some(())?;
                dropout = Some(d);
                x
            }
            None if dropout.is_some() => return None,
            None => dpt,
        };
        let dpt_node = self.producer[dpt].filter(|_| self.single(dpt))?;
        let (lc2, rc2, lb2, rb2, acc2, out2) = dot(dpt_node)?;
        let (v_, do_) = (nodes[dpt_node].inputs[0], nodes[dpt_node].inputs[1]);
        if lb2.len() != nb || lc2.len() != 1 || rc2.len() != 1 || acc2 != f32 || out2 != f32 {
            return None;
        }
        let (vf, dof) = (
            free(ty(v_).shape.len(), &[lb2, lc2])?,
            free(ty(do_).shape.len(), &[rb2, rc2])?,
        );
        let m1_out = m1.output;
        let scaled = match self.readers[m1_out][..] {
            [m2] if !self.output[m1_out] && matches!(nodes[m2].primitive, Mul) => {
                let (a, b) = (nodes[m2].inputs[0], nodes[m2].inputs[1]);
                let c = match a == m1_out {
                    true => self.scalar(b),
                    false => self.scalar(a),
                };
                c.map(|c| (c, nodes[m2].output))
            }
            _ => None,
        };
        let (ds_scale, mut ds) = scaled.unwrap_or((1.0, m1_out));
        if let [c] = self.readers[ds][..]
            && !self.output[ds]
            && matches!(nodes[c].primitive, Cast { .. })
        {
            ds = nodes[c].output;
        }
        // Readers of dS are the dK and dQ dots, told apart by contracting dim and other operand.
        let (mut dk, mut dq) = (None, None);
        for &r in &self.readers[ds] {
            let (lc, rc, lb, _, acc, out) = dot(r)?;
            if nodes[r].inputs[0] != ds
                || *lb != (0..nb).collect::<Vec<_>>()
                || acc != f32
                || out != f32
            {
                return None;
            }
            match (lc.as_slice(), rc.as_slice()) {
                ([c], [d])
                    if *c == nb + 1 && nodes[r].inputs[1] == q_ && *d == qf && dk.is_none() =>
                {
                    dk = Some(r)
                }
                ([c], [d]) if *c == nb && nodes[r].inputs[1] == k_ && *d == kf && dq.is_none() => {
                    dq = Some(r)
                }
                _ => return None,
            }
        }
        if self.output[ds] || (dk.is_none() && dq.is_none()) {
            return None;
        }
        if let Some(d) = dv {
            let (lc, rc, lb, _, acc, out) = dot(d)?;
            let ok = *lb == (0..nb).collect::<Vec<_>>()
                && *lc == [nb + 1]
                && nodes[d].inputs[1] == do_
                && *rc == [dof]
                && acc == f32
                && out == f32;
            ok.then_some(())?;
        }
        let same = [k_, q_, v_, do_].iter().all(|&t| ty(t).dtype == dtype)
            && ty(ds).dtype == dtype
            && dv.is_none_or(|d| ty(nodes[d].inputs[0]).dtype == dtype);
        let shape = &ty(x).shape;
        let (sk, sq) = (shape[nb], shape[nb + 1]);
        let (h, hv) = (ty(q_).shape[rc1[0]], ty(v_).shape[lc2[0]]);
        let fits = |n: usize| n.is_multiple_of(8) && n > 0 && n <= MAX_BACKWARD_HEAD;
        let float = matches!(dtype, DType::F16 | DType::BF16 | DType::F32);
        if !same || !float || !fits(h) || !fits(hv) || sq == 0 || sk == 0 {
            return None;
        }
        let all: Vec<usize> = (0..nb).collect();
        let q = self.access(q_, rb1, qf, rc1[0])?;
        let k = self.access(k_, lb1, kf, lc1[0])?;
        let v = self.access(v_, lb2, vf, lc2[0])?;
        let d_o = self.access(do_, rb2, dof, rc2[0])?;
        let lse = self.access(lse_b, &all, nb, nb + 1)?;
        let delta = self.access(d_b, &all, nb, nb + 1)?;
        let per_query = |acc: &Access, v: Var| acc.row == 0 && ty(v).dtype == f32;
        if !per_query(&lse, lse_b) || !per_query(&delta, d_b) {
            return None;
        }
        if !fits32(&q, sq, h) || !fits32(&k, sk, h) || !fits32(&v, sk, hv) || !fits32(&d_o, sq, hv)
        {
            return None;
        }
        let out = |d: Option<usize>, chain: bool| match d {
            Some(d) => self.written(d, nb, chain, false).map(Some),
            None => Some(None),
        };
        Some(Backward {
            dv_out: out(dv, true)?,
            dk_out: out(dk, true)?,
            dq_out: out(dq, !self.plain_dq)?,
            q,
            k,
            v,
            d_o,
            lse,
            delta,
            batch: shape[..nb].to_vec(),
            sq,
            sk,
            h,
            hv,
            scores,
            causal,
            ds_scale,
            dtype,
            dropout,
        })
    }

    /// Follows dot `i` through transposes (and with `chain`, reshapes and one float cast) to where it
    /// is written; returns that node and the write as an [`Access`] over the dot's dims. With `shared`,
    /// the dot's one transpose is followed though others read the dot too.
    fn written(&self, i: usize, nb: usize, chain: bool, shared: bool) -> Option<(usize, Access)> {
        use Primitive::*;
        let nodes = self.graph.nodes();
        let mut path = vec![i];
        let mut cast = false;
        loop {
            let o = nodes[*path.last().expect("the dot")].output;
            let transposes: Vec<usize> = self.readers[o]
                .iter()
                .copied()
                .filter(|&r| matches!(nodes[r].primitive, Transpose { .. }))
                .collect();
            let r = match (&self.readers[o][..], &transposes[..]) {
                (&[r], _) if !self.output[o] && (chain || path.len() == 1) => r,
                (_, &[r]) if shared && !self.output[o] && path.len() == 1 => r,
                _ => break,
            };
            match &nodes[r].primitive {
                Transpose { .. } => {}
                Reshape { .. } if chain => {}
                Cast { new_dtype } if chain && !cast && new_dtype.is_float() => cast = true,
                _ => break,
            }
            path.push(r);
        }
        let root = *path.last().expect("the dot");
        let value = nodes[root].output;
        let (offset, strides) = match self.views.get(&value) {
            Some(view) => view.clone(),
            None => (0, contiguous_strides(&self.graph.type_of(value).shape)),
        };
        // Map the written buffer's (stride, divisor) dims back through the path to the dot's dims.
        let mut dims: Vec<(usize, usize)> = strides.into_iter().map(|s| (s, 1)).collect();
        for &k in path[1..].iter().rev() {
            let node = &nodes[k];
            dims = match &node.primitive {
                Transpose { permutation } => {
                    let mut operand = vec![(0, 1); dims.len()];
                    for (d, &p) in permutation.iter().enumerate() {
                        operand[p] = dims[d];
                    }
                    operand
                }
                Reshape { .. } => {
                    let from = &self.graph.type_of(node.output).shape;
                    let shape = &self.graph.type_of(node.inputs[0]).shape;
                    let operand = reshaped(from, &dims, shape)?;
                    operand
                        .iter()
                        .all(|&(_, div)| div == 1)
                        .then_some(operand)?
                }
                _ => dims,
            };
        }
        let out = Access {
            base: value,
            offset,
            batch: dims[..nb].to_vec(),
            row: dims[nb].0,
            col: dims[nb + 1].0,
        };
        Some((root, out))
    }

    /// Walks back from scores `x` through casts, one constant scale and one causal mask to the
    /// `Q K^T` dot. Returns the dot, the score ops in application order and the causal mask.
    fn scores(
        &self,
        x: Var,
        nb: usize,
        transposed: bool,
    ) -> Option<(usize, Vec<Score>, Option<Causal>)> {
        use Primitive::*;
        let ty = |v: Var| self.graph.type_of(v);
        let (mut v, mut scale, mut causal) = (x, None, None);
        let mut scores = Vec::new();
        let s = loop {
            let n = self.node(v)?;
            if v != x && !self.single(v) {
                return None;
            }
            let dtype = ty(n.output).dtype;
            match &n.primitive {
                DotGeneral { .. } => break self.producer[v]?,
                Cast { .. } => {
                    scores.push(Score::Round(dtype));
                    v = n.inputs[0];
                }
                Mul if scale.is_none() => {
                    let (a, b) = (n.inputs[0], n.inputs[1]);
                    let (c, rest) = match self.scalar(b) {
                        Some(c) => (c, a),
                        None => (self.scalar(a)?, b),
                    };
                    scale = Some(c);
                    scores.push(Score::Mul(c, dtype));
                    v = rest;
                }
                Select if causal.is_none() => {
                    if self.scalar(n.inputs[2]) != Some(f64::NEG_INFINITY) {
                        return None;
                    }
                    causal = Some(self.causal(n.inputs[0], nb, transposed)?);
                    v = n.inputs[1];
                }
                _ => return None,
            }
        };
        if scale.is_some_and(|c| !(c > 0.0 && c.is_finite())) {
            return None;
        }
        scores.push(Score::Round(ty(self.graph.nodes()[s].output).dtype));
        scores.reverse();
        Some((s, scores, causal))
    }

    /// Node computing the logsumexp `m + log(l)`, when the max and sum feed only it and the softmax.
    fn lse(&self, m: Var, l: Var) -> Option<usize> {
        let nodes = self.graph.nodes();
        let two = |v: Var| self.readers[v].len() == 2 && !self.output[v];
        if !two(m) || !two(l) {
            return None;
        }
        let add = *self.readers[m]
            .iter()
            .find(|&&r| matches!(nodes[r].primitive, Primitive::Add))?;
        let log = *nodes[add].inputs.iter().find(|&&v| v != m)?;
        let is_log = self.node(log).is_some_and(|n| {
            matches!(n.primitive, Primitive::Log) && n.inputs[0] == l && self.single(log)
        });
        let f32 = self.graph.type_of(nodes[add].output).dtype == DType::F32;
        (is_log && f32 && nodes[add].inputs.contains(&m)).then_some(add)
    }

    /// Operand of `v` when `v` is a last-axis reduction accepted by `is`.
    fn reduced(&self, v: Var, is: fn(&Primitive) -> bool, shared: Option<Var>) -> Option<Var> {
        let n = self.node(v)?;
        let x = *n.inputs.first()?;
        let last = self.graph.type_of(x).shape.len().checked_sub(1)?;
        let axes = match &n.primitive {
            Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => axes,
            _ => return None,
        };
        let read = self.single(v) || shared == Some(v);
        (is(&n.primitive) && axes.as_slice() == [last] && read).then_some(x)
    }

    /// Matches `softmax(scores) V` at dot node `i`, with optional dropout on the probabilities.
    fn attention(&self, i: usize) -> Option<Attention> {
        use Primitive::*;
        let nodes = self.graph.nodes();
        let ty = |v: Var| self.graph.type_of(v);
        let DotGeneral {
            lhs_contracting: lc2,
            rhs_contracting: rc2,
            lhs_batch: lb2,
            rhs_batch: rb2,
            ..
        } = &nodes[i].primitive
        else {
            return None;
        };
        let (p_, v_) = (nodes[i].inputs[0], nodes[i].inputs[1]);
        let nb = lb2.len();
        let (pr, vr) = (ty(p_).shape.len(), ty(v_).shape.len());
        if pr != nb + 2 || vr != nb + 2 || *lb2 != (0..nb).collect::<Vec<_>>() || *lc2 != [nb + 1] {
            return None;
        }
        if !self.single(p_) {
            return None;
        }
        let (rounded, dropout) = match self.dropout(p_, &ty(p_).shape, false) {
            Some((x, d, t)) => (x, Some((d, t))),
            None => (p_, None),
        };
        let p = self.uncast(rounded);
        let div = self
            .node(p)
            .filter(|n| matches!(n.primitive, Div) && self.single(p))?;
        let e = div.inputs[0];
        let exp = self.node(e).filter(|n| matches!(n.primitive, Exp))?;
        if self.readers[e].len() != 2 || self.output[e] {
            return None;
        }
        let sub = self
            .node(exp.inputs[0])
            .filter(|n| matches!(n.primitive, Sub) && self.single(exp.inputs[0]))?;
        let x = sub.inputs[0];
        let (l, m) = (
            self.unbroadcast(div.inputs[1], None),
            self.unbroadcast(sub.inputs[1], None),
        );
        let lse = self.lse(m, l);
        let shared = |v: Var| lse.is_some().then_some(v);
        let sum = self.unbroadcast(l, shared(l));
        let max = self.unbroadcast(m, shared(m));
        if self.reduced(sum, |p| matches!(p, ReduceSum { .. }), shared(l))? != e {
            return None;
        }
        if self.reduced(max, |p| matches!(p, ReduceMax { .. }), shared(m))? != x
            || self.readers[x].len() != 2
            || self.output[x]
        {
            return None;
        }
        let (dot1, scores, causal) = self.scores(x, nb, false)?;
        let s = &nodes[dot1];
        let f32 = |p: &Primitive| match p {
            DotGeneral { accum_dtype, .. } | ReduceSum { accum_dtype, .. } => {
                *accum_dtype == DType::F32
            }
            _ => false,
        };
        let (p1, p2, ps) = (
            &s.primitive,
            &nodes[i].primitive,
            &self.node(sum)?.primitive,
        );
        if ty(x).dtype != DType::F32 || !f32(p1) || !f32(p2) || !f32(ps) {
            return None;
        }
        let DotGeneral {
            lhs_contracting: lc1,
            rhs_contracting: rc1,
            lhs_batch: lb1,
            rhs_batch: rb1,
            ..
        } = &s.primitive
        else {
            unreachable!("a dot")
        };
        let (q_, k_) = (s.inputs[0], s.inputs[1]);
        let (qr, kr) = (ty(q_).shape.len(), ty(k_).shape.len());
        if lb1.len() != nb || lc1.len() != 1 || rc1.len() != 1 || qr != nb + 2 || kr != nb + 2 {
            return None;
        }
        let free = |rank: usize, used: &[&[usize]]| {
            (0..rank).find(|d| used.iter().all(|u| !u.contains(d)))
        };
        let (qf, kf, vf) = (
            free(qr, &[lb1, lc1])?,
            free(kr, &[rb1, rc1])?,
            free(vr, &[rb2, rc2])?,
        );
        let sty = ty(s.output);
        let (sq, sk) = (sty.shape[nb], sty.shape[nb + 1]);
        let (h, hv) = (ty(q_).shape[lc1[0]], ty(v_).shape[vf]);
        let batch: Vec<usize> = sty.shape[..nb].to_vec();
        let dtype = ty(q_).dtype;
        let float = matches!(dtype, DType::F16 | DType::BF16 | DType::F32);
        let fits = |n: usize| n.is_multiple_of(8) && n > 0 && n <= MAX_HEAD;
        if !float || ty(k_).dtype != dtype || ty(v_).dtype != dtype || ty(p_).dtype != dtype {
            return None;
        }
        if dropout.as_ref().is_some_and(|(_, t)| *t != dtype) {
            return None;
        }
        if !fits(h) || !fits(hv) || sq == 0 || sk == 0 || ty(v_).shape[rc2[0]] != sk {
            return None;
        }
        let q = self.access(q_, lb1, qf, lc1[0])?;
        let k = self.access(k_, rb1, kf, rc1[0])?;
        let v = self.access(v_, rb2, rc2[0], vf)?;
        if !fits32(&q, sq, h) || !fits32(&k, sk, h) || !fits32(&v, sk, hv) {
            return None;
        }
        let (root, out) = self.written(i, nb, false, false)?;
        Some(Attention {
            root,
            q,
            k,
            v,
            out,
            batch,
            sq,
            sk,
            h,
            hv,
            scores,
            causal: causal.map(|(c, _)| c),
            offset: causal.and_then(|(_, v)| v),
            lse,
            dot1,
            dot2: i,
            max: m,
            sum: l,
            dtype,
            out_dtype: ty(nodes[root].output).dtype,
            dropout: dropout.map(|(d, _)| d),
        })
    }

    /// Access of `v` with batch dims `b` and the given row/col dims, which must be undivided.
    fn access(&self, v: Var, b: &[usize], row: usize, col: usize) -> Option<Access> {
        let (base, offset, dims) = self.view(v);
        let one = |d: usize| (dims[d].1 == 1).then_some(dims[d].0);
        Some(Access {
            base,
            offset,
            batch: b.iter().map(|&d| dims[d]).collect(),
            row: one(row)?,
            col: one(col)?,
        })
    }

    /// Matches a causal mask `key < query + c` (or `<=` built as `lt | eq`) on iota coordinates.
    /// Returns the diagonal offset and an optional runtime offset; `transposed` swaps query and key dims.
    fn causal(&self, v: Var, nb: usize, transposed: bool) -> Option<Causal> {
        use Primitive::*;
        let shape = self.graph.type_of(v).shape.clone();
        let (v, dims) = self.broadcast_dims(v);
        let n = self.node(v)?;
        let (a, b, inclusive) = match n.primitive {
            Lt => (n.inputs[0], n.inputs[1], false),
            Select => {
                let less = self
                    .node(n.inputs[0])
                    .filter(|l| matches!(l.primitive, Lt))?;
                let eq = self
                    .node(n.inputs[2])
                    .filter(|e| matches!(e.primitive, Eq))?;
                let swapped = [eq.inputs[1], eq.inputs[0]];
                if n.inputs[1] != n.inputs[0]
                    || (less.inputs != eq.inputs && less.inputs != swapped)
                {
                    return None;
                }
                (less.inputs[0], less.inputs[1], true)
            }
            _ => return None,
        };
        let (a, b) = (self.affine(a)?, self.affine(b)?);
        let mut outer = vec![0; nb + 2];
        for (k, (x, y)) in a.coefficients.iter().zip(&b.coefficients).enumerate() {
            match (dims[k], x - y) {
                (_, 0) => {}
                (Some(d), c) if d < outer.len() => outer[d] = c,
                _ => return None,
            }
        }
        // The mask must compare exactly `key - query`; size-1 dims accept either coefficient.
        let (query, key) = if transposed {
            (nb + 1, nb)
        } else {
            (nb, nb + 1)
        };
        for (d, c) in [(query, -1), (key, 1)] {
            if shape.get(d) == Some(&1) {
                outer[d] = c;
            }
        }
        if outer[..nb].iter().any(|&c| c != 0) || (outer[query], outer[key]) != (-1, 1) {
            return None;
        }
        let offset = match sum_terms(a.term, b.term.map(|(v, k)| (v, -k))) {
            None => None,
            Some((v, -1)) => Some(v),
            Some(_) => return None,
        };
        // Normalize to "kept when key <= query + c".
        let c0 = a.constant - b.constant;
        Some((if inclusive { -c0 } else { -c0 - 1 }, offset))
    }

    /// `v` with broadcasts and unit reshapes removed, and for each of its dims the outer dim it maps to.
    fn broadcast_dims(&self, mut v: Var) -> (Var, Vec<Option<usize>>) {
        let mut dims: Vec<Option<usize>> =
            (0..self.graph.type_of(v).shape.len()).map(Some).collect();
        while let Some(n) = self.node(v) {
            let inner = &self.graph.type_of(n.inputs[0]).shape;
            let mapped = match &n.primitive {
                Primitive::BroadcastInDim {
                    broadcast_dimensions,
                    ..
                } => broadcast_dimensions.iter().map(|&d| dims[d]).collect(),
                Primitive::Reshape { .. } => {
                    let outer = &self.graph.type_of(v).shape;
                    let Some(map) = unit_reshape(inner, outer) else {
                        break;
                    };
                    map.iter().map(|d| d.and_then(|d| dims[d])).collect()
                }
                _ => break,
            };
            dims = mapped;
            v = n.inputs[0];
        }
        (v, dims)
    }

    /// `v` as an affine function of its coordinates plus a constant and an optional runtime scalar.
    fn affine(&self, v: Var) -> Option<Affine> {
        use Primitive::*;
        let ty = self.graph.type_of(v);
        let rank = ty.shape.len();
        let constant = |c: i64| Affine {
            coefficients: vec![0; rank],
            constant: c,
            term: None,
        };
        let Some(n) = self.node(v) else {
            let runtime = self.runtime_offsets && rank == 0 && ty.dtype.is_int();
            return runtime.then_some(Affine {
                term: Some((v, 1)),
                ..constant(0)
            });
        };
        match &n.primitive {
            Iota { dimension, .. } => {
                let mut a = constant(0);
                a.coefficients[*dimension] = 1;
                Some(a)
            }
            Full { fill_value, .. } => {
                let c = fill_value.to_f64();
                (c.fract() == 0.0).then(|| constant(c as i64))
            }
            Cast { .. } => self.affine(n.inputs[0]),
            BroadcastInDim {
                broadcast_dimensions,
                ..
            } => {
                let a = self.affine(n.inputs[0])?;
                let inner = &self.graph.type_of(n.inputs[0]).shape;
                let mut out = constant(a.constant);
                out.term = a.term;
                for (k, &d) in broadcast_dimensions.iter().enumerate() {
                    if inner[k] == ty.shape[d] {
                        out.coefficients[d] = a.coefficients[k];
                    }
                }
                Some(out)
            }
            Reshape { .. } => {
                let a = self.affine(n.inputs[0])?;
                let map = unit_reshape(&self.graph.type_of(n.inputs[0]).shape, &ty.shape)?;
                let mut out = constant(a.constant);
                out.term = a.term;
                for (k, d) in map.into_iter().enumerate() {
                    if let Some(d) = d {
                        out.coefficients[d] = a.coefficients[k];
                    }
                }
                Some(out)
            }
            Add | Sub => {
                let (a, b) = (self.affine(n.inputs[0])?, self.affine(n.inputs[1])?);
                let s = if matches!(n.primitive, Add) { 1 } else { -1 };
                let term = match (a.term, b.term) {
                    (Some((x, _)), Some((y, _))) if x != y => return None,
                    (at, bt) => sum_terms(at, bt.map(|(v, k)| (v, s * k))),
                };
                let coefficients = (a.coefficients.iter().zip(&b.coefficients))
                    .map(|(x, y)| x + s * y)
                    .collect();
                Some(Affine {
                    coefficients,
                    constant: a.constant + s * b.constant,
                    term,
                })
            }
            _ => None,
        }
    }

    /// `v` as a view of a base value: the base, element offset and `(stride, divisor)` per dim.
    fn view(&self, v: Var) -> (Var, usize, Vec<(usize, usize)>) {
        use Primitive::*;
        let own = || {
            let shape = &self.graph.type_of(v).shape;
            let strides = match self.strided.get(&v) {
                Some(strides) => strides.clone(),
                None => contiguous_strides(shape),
            };
            (v, 0, strides.into_iter().map(|s| (s, 1)).collect())
        };
        let Some(n) = self.node(v) else {
            return own();
        };
        let layout = matches!(
            n.primitive,
            Transpose { .. } | Slice { .. } | BroadcastInDim { .. } | Reshape { .. }
        );
        if !layout {
            return own();
        }
        let x = n.inputs[0];
        let (base, mut offset, dims) = self.view(x);
        let xs = &self.graph.type_of(x).shape;
        let shape = &self.graph.type_of(v).shape;
        let dims = match &n.primitive {
            Transpose { permutation } => permutation.iter().map(|&p| dims[p]).collect(),
            Slice { start_indices, .. } => {
                for (d, &s) in start_indices.iter().enumerate() {
                    if s != 0 && dims[d].1 != 1 {
                        return own();
                    }
                    offset += s * dims[d].0;
                }
                dims
            }
            BroadcastInDim {
                broadcast_dimensions,
                ..
            } => {
                let mut out = vec![(0, 1); shape.len()];
                for (k, &d) in broadcast_dimensions.iter().enumerate() {
                    if xs[k] == shape[d] {
                        out[d] = dims[k];
                    }
                }
                out
            }
            Reshape { .. } => match reshaped(xs, &dims, shape) {
                Some(out) => out,
                None => return own(),
            },
            _ => unreachable!("a layout primitive"),
        };
        (base, offset, dims)
    }
}

/// Causal diagonal offset and optional runtime offset value.
type Causal = (i64, Option<Var>);

/// An integer tensor as `coefficients . coords + constant + coefficient * var`.
struct Affine {
    coefficients: Vec<i64>,
    constant: i64,
    term: Option<(Var, i64)>,
}

/// Sum of two runtime terms; `None` when they cancel or name different values.
fn sum_terms(a: Option<(Var, i64)>, b: Option<(Var, i64)>) -> Option<(Var, i64)> {
    match (a, b) {
        (Some((v, x)), Some((w, y))) if v == w => (x + y != 0).then_some((v, x + y)),
        (Some(_), Some(_)) => None,
        (t, None) | (None, t) => t,
    }
}

/// For a reshape that only adds or drops size-1 dims, the outer dim each inner dim maps to.
fn unit_reshape(inner: &[usize], outer: &[usize]) -> Option<Vec<Option<usize>>> {
    let kept =
        |shape: &[usize]| -> Vec<usize> { (0..shape.len()).filter(|&d| shape[d] != 1).collect() };
    let (a, b) = (kept(inner), kept(outer));
    if a.len() != b.len() || a.iter().zip(&b).any(|(&x, &y)| inner[x] != outer[y]) {
        return None;
    }
    let mut map = vec![None; inner.len()];
    for (&x, &y) in a.iter().zip(&b) {
        map[x] = Some(y);
    }
    Some(map)
}

/// Re-expresses per-dim `(stride, divisor)` of a view of shape `from` after reshaping it to `shape`,
/// or `None` when the result isn't a strided view.
fn reshaped(
    from: &[usize],
    dims: &[(usize, usize)],
    shape: &[usize],
) -> Option<Vec<(usize, usize)>> {
    let ins: Vec<(usize, (usize, usize))> = from
        .iter()
        .zip(dims)
        .filter(|(n, _)| **n != 1)
        .map(|(&n, &d)| (n, d))
        .collect();
    let outs: Vec<usize> = (0..shape.len()).filter(|&d| shape[d] != 1).collect();
    let mut out = vec![(0, 1); shape.len()];
    let (mut a, mut b) = (0, 0);
    // Pair up runs of input and output dims with equal products.
    while a < ins.len() || b < outs.len() {
        let (mut pa, mut pb) = (ins.get(a)?.0, shape[*outs.get(b)?]);
        let (a0, b0) = (a, b);
        while pa != pb {
            if pa < pb {
                a += 1;
                pa *= ins.get(a)?.0;
            } else {
                b += 1;
                pb *= shape[*outs.get(b)?];
            }
        }
        let group = &ins[a0..=a];
        let targets = &outs[b0..=b];
        match (group, targets) {
            ([(_, d)], [t]) => out[*t] = *d,
            // Splitting an undivided dim: strides follow the output shape.
            ([(_, (s, 1))], ts) => {
                let mut stride = *s;
                for &t in ts.iter().rev() {
                    out[t] = (stride, 1);
                    stride *= shape[t];
                }
            }
            // Merging dims: trailing broadcast dims become a divisor, otherwise they must be contiguous.
            (gs, [t]) => {
                let (_, (s0, d0)) = gs[0];
                let rest = &gs[1..];
                if rest.iter().all(|(_, (s, d))| *s == 0 && *d == 1) {
                    let inner: usize = rest.iter().map(|(n, _)| n).product();
                    out[*t] = (s0, d0 * inner);
                } else if gs
                    .windows(2)
                    .all(|w| w[0].1.1 == 1 && w[1].1.1 == 1 && w[0].1.0 == w[1].1.0 * w[1].0)
                {
                    out[*t] = (gs.last()?.1.0, 1);
                } else {
                    return None;
                }
            }
            _ => return None,
        }
        a += 1;
        b += 1;
    }
    Some(out)
}
