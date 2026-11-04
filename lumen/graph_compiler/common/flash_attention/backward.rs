//! Flash attention backward: matching attention's gradients in traced graphs and the rewrite replacing
//! them with `flash_attention_backward` fusions.

use std::collections::HashMap;

use super::{Access, Dropout, Matcher, Score, Strides, body_of, fits32};
use crate::DType;
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};

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

/// The first attention backward in `body`, its operands laid out as `strides` gives them.
pub(crate) fn backward_of_body_strided(body: &Graph, strides: &Strides) -> Option<Backward> {
    let m = Matcher::new(body).laid_out(strides);
    (0..body.nodes().len()).find_map(|i| m.backward(i))
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

impl<'a> Matcher<'a> {
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
}

/// A module whose backend can generate flash-attention backward kernels.
pub(crate) trait AttentionBackwards: Module {
    /// Which backwards run as one kernel computing dK, dV and dQ; the others get a dK/dV kernel and
    /// a dQ kernel.
    fn together(&self) -> impl Fn(&Backward) -> bool + 'static;
    /// Requests a kernel for the next backward fusion and returns its name.
    fn backward_kernel(&mut self) -> String;
}

/// Compiler pass running [`fuse_backward`].
pub(crate) struct FlashAttentionBackward;

impl<M: AttentionBackwards> CompilerPass<M> for FlashAttentionBackward {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "flash_attention_backward"
    }

    /// Replaces each attention backward in the module's graph with its flash-attention fusions.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let graph = m.graph().clone();
        let together = m.together();
        *m.graph_mut() = fuse_backward(&graph, together, |_| m.backward_kernel());
        Ok(())
    }
}
