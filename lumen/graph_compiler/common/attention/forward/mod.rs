//! Flash attention forward: matching attention in traced graphs and the rewrite replacing each with a
//! `flash_attention` fusion.

#[cfg(lumen_mps_linked)]
mod mps;
pub(super) mod tpu;
mod triton;

use super::{Access, Dropout, Matcher, Score, Strides, body_of, fits32};
use crate::DType;
use crate::graph::{FUSION_SEPARATOR, Graph, Primitive, Var};
use crate::graph_compiler::common::simplify::rounds;
use crate::graph_compiler::pass::{CompilerPass, Module};

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

/// Each attention traced in `graph` with its output (the second dot's, or its rounding cast's) and the
/// values computed between its operands and output.
pub(crate) fn traced(graph: &Graph) -> Vec<(Attention, Var, Vec<Var>)> {
    let nodes = graph.nodes();
    let mut producer = vec![None; graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }

    let m = Matcher::new(graph);

    attentions(graph, false)
        .into_iter()
        .map(|a| {
            let (d1, d2) = (&nodes[a.dot1], &nodes[a.dot2]);
            let operands = [d1.inputs[0], d1.inputs[1], d2.inputs[1]];
            let mut members: Vec<Var> = Vec::new();
            let out = nodes[m.rounding(a.dot2).unwrap_or(a.dot2)].output;
            let mut stack = vec![out];

            while let Some(v) = stack.pop() {
                if operands.contains(&v) || members.contains(&v) {
                    continue;
                }

                let Some(n) = producer[v] else { continue };
                members.push(v);
                stack.extend(&nodes[n].inputs);
            }

            members.sort_unstable();
            (a, out, members)
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
            let start = m.rounding(a.dot2).unwrap_or(a.dot2);
            let dot = nodes[start].output;
            let through = epilogues.is_some_and(|e| e.transposed) && a.root == start;

            let shared = through
                .then(|| m.written(start, a.batch.len(), false, true))
                .flatten()
                .filter(|&(root, _)| root != start);

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
        let start = m.rounding(a.dot2).unwrap_or(a.dot2);

        if a.root != start {
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

            map[nodes[start].output] = out
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

    /// `v` with a single-use cast removed.
    fn uncast(&self, v: Var) -> Var {
        match self.node(v) {
            Some(n) if matches!(n.primitive, Primitive::Cast { .. }) && self.single(v) => {
                n.inputs[0]
            }
            _ => v,
        }
    }

    /// Dot `i`'s sole reader when it is the dot's rounding cast ([`rounds`]), which the kernel does as it
    /// stores; `None` when the dot writes its accumulator.
    fn rounding(&self, i: usize) -> Option<usize> {
        let (nodes, dot) = (self.graph.nodes(), &self.graph.nodes()[i]);

        match self.readers[dot.output][..] {
            [r] if !self.output[dot.output]
                && matches!(nodes[r].primitive, Primitive::Cast { new_dtype }
                    if rounds(self.graph, &dot.primitive, &dot.inputs, new_dtype)) =>
            {
                Some(r)
            }
            _ => None,
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

        let (root, out) = self.written(self.rounding(i).unwrap_or(i), nb, false, false)?;

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
}

/// A module whose backend can generate flash-attention kernels.
pub(crate) trait Attentions: Module {
    /// Whether elementwise epilogues may be fused into the attention.
    fn contraction_epilogues(&self) -> bool;
    /// Which epilogues the backend accepts: by default any leaf, without reshapes.
    fn epilogues(&self) -> Epilogues {
        Epilogues {
            leaves: |_, _| true,
            reshaped: false,
            transposed: true,
        }
    }
    /// Values passed to kernels by value rather than as buffers.
    fn scalars(&self) -> Vec<Var>;
    /// Requests a kernel for the next fusion and returns its name.
    fn kernel(&mut self) -> String;
    /// Whether the causal diagonal may include a runtime offset: by default the kernel reads it at run time.
    fn runtime_offsets(&self) -> bool {
        true
    }
}

/// Compiler pass running [`fuse`].
pub(crate) struct FlashAttention;

impl<M: Attentions> CompilerPass<M> for FlashAttention {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "flash_attention"
    }

    /// Replaces each forward attention in the module's graph with a flash-attention fusion.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let epilogues = m.contraction_epilogues().then(|| m.epilogues());
        let (graph, scalars) = (m.graph().clone(), m.scalars());
        let runtime_offsets = m.runtime_offsets();

        *m.graph_mut() = fuse(&graph, epilogues, &scalars, runtime_offsets, |_| m.kernel());

        Ok(())
    }
}
