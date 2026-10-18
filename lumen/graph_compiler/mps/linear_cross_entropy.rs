//! A linear layer's cross entropy, matched as written: `F.cross_entropy`
//! of a dot's logits (`F.matmul(h, w.t(), "float32", "float32")`, an LM
//! head's), however traced, run as [`Primitive::LinearCrossEntropy`] (two
//! kernels, Cut Cross-Entropy's) when nothing else reads the logits: they
//! never reach memory. As flash attention's ([`crate::graph_compiler::attention`]),
//! no call of its own needed (trained, `lumen/autograd/cross_entropy.py`
//! makes the gradient not read the logits).
//!
//! What is matched, from the dot (`F.cross_entropy`'s forward as traced):
//!
//! ```text
//! x    = dot_general(h, w')         [B, V], h [B, D], w' [D, V] or [V, D], in and to float32
//! m    = reduce_max(x, 1)           read as reshape(m, [B, 1]): r
//! lse  = reshape(r, [B]) + log(reduce_sum(exp(x - broadcast(r)), 1))
//! pick = gather(reshape(x, [B V]), iota(B) * V + target, 0)
//! loss = lse - pick
//! ```
//!
//! `loss` and `lse` are replaced by the primitive's rows (its kernels
//! write the loss); each other value read by the pattern alone. Its logits
//! accumulate in float32 as the dot's, but its log-sum-exp adds them in
//! another order (each tile's classes' first), as the row kernel's online
//! softmax does.
//!
//! Trained a chunk of rows at a time (its gradient recomputing a chunk's
//! logits, `lumen.config.compiler.fused_linear_cross_entropy_chunk_size`),
//! it is computed from those chunks' logits instead
//! ([`chunked_cross_entropy`]): each chunk's logits once, its losses and
//! gradient one row kernel of them (XMA's fused linear cross entropy).

use crate::graph::{Graph, Node, Primitive, Var};
use crate::{DType, Scalar};

/// A match: the dot's operands and the pattern's nodes (by index), `loss`
/// and `lse` its values read outside it (`pick`, the first of `lse` and the
/// class's logit: where it is computed).
struct Found {
    h: Var,
    w: Var,
    /// Whether `w` is `[D, V]` (read transposed).
    transposed: bool,
    target: Var,
    members: Vec<usize>,
    loss: usize,
    lse: usize,
    first: usize,
    /// The gather of the class's logit.
    pick: usize,
}

/// `graph` with each linear cross entropy whose logits nothing else reads
/// as a [`Primitive::LinearCrossEntropy`].
pub(crate) fn linear_cross_entropy(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let (producer, readers) = uses(graph);
    let found: Vec<Found> = (0..nodes.len())
        .filter_map(|i| matched(graph, &producer, &readers, i))
        .collect();
    if found.is_empty() {
        return graph.clone();
    }
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for (i, node) in nodes.iter().enumerate() {
        let Some(f) = found.iter().find(|f| f.members.contains(&i)) else {
            out.set_origin(node);
            map[node.output] = out
                .apply(
                    node.primitive.clone(),
                    &node.inputs.iter().map(|&v| map[v]).collect::<Vec<_>>(),
                )
                .expect("well typed");
            out.set_label(map[node.output], node.label);
            continue;
        };
        // At the first of its values read outside it (the target defined
        // before: `matched`'s).
        if i != f.first {
            continue;
        }
        let members: Vec<&Node> = f.members.iter().map(|&k| &nodes[k]).collect();
        out.set_origins(&members);
        let apply =
            |out: &mut Graph, p: Primitive, ins: &[Var]| out.apply(p, ins).expect("well typed");
        let w = match f.transposed {
            true => apply(
                &mut out,
                Primitive::Transpose {
                    permutation: vec![1, 0],
                },
                &[map[f.w]],
            ),
            false => map[f.w],
        };
        let rows = apply(
            &mut out,
            Primitive::LinearCrossEntropy,
            &[map[f.h], w, map[f.target]],
        );
        let b = graph.type_of(f.h).shape[0];
        for (k, row) in [(f.loss, 0), (f.lse, 1)] {
            let slice = Primitive::Slice {
                start_indices: vec![row, 0],
                limit_indices: vec![row + 1, b],
                view: false,
            };
            let v = apply(&mut out, slice, &[rows]);
            map[nodes[k].output] = apply(&mut out, Primitive::Reshape { new_sizes: vec![b] }, &[v]);
        }
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Each value's producing node and its readers (by node index).
fn uses(graph: &Graph) -> (Vec<Option<usize>>, Vec<Vec<usize>>) {
    let mut producer = vec![None; graph.types.len()];
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];
    for (i, node) in graph.nodes().iter().enumerate() {
        producer[node.output] = Some(i);
        for &v in &node.inputs {
            if !readers[v].contains(&i) {
                readers[v].push(i);
            }
        }
    }
    (producer, readers)
}

/// `graph` with each linear cross entropy whose gradient a chunk of rows
/// at a time recomputes their logits (`lumen/autograd/cross_entropy.py`:
/// a dot of the chunk's rows of `h`, the forward's) computed from those
/// chunks' logits: each chunk's losses and log-sum-exp from the logits
/// its gradient reads, beside it (the rows kernel writing both), the
/// whole logits never computed. XMA's fused linear cross entropy: the
/// logits of a chunk computed once, its gradient formed with its loss.
/// Not where the gradient's `g` reads the losses (waiting for every
/// chunk's, each chunk's logits would be kept until then).
pub(crate) fn chunked_cross_entropy(graph: &Graph) -> Graph {
    let (producer, readers) = uses(graph);
    for i in 0..graph.nodes().len() {
        if let Some(f) = matched(graph, &producer, &readers, i)
            && let Some(g) = chunked(graph, &producer, i, &f)
        {
            return chunked_cross_entropy(&g);
        }
    }
    graph.clone()
}

/// `graph` with the linear cross entropy `f` of the dot at node `dot`
/// computed from its chunks' logits, if its gradient recomputes them (a
/// dot as `dot` of a slice of rows of `h` a chunk, the chunks its rows).
fn chunked(graph: &Graph, producer: &[Option<usize>], dot: usize, f: &Found) -> Option<Graph> {
    let nodes = graph.nodes();
    let (b, d) = (graph.type_of(f.h).shape[0], graph.type_of(f.h).shape[1]);
    // The chunks' dots: (node, first row, row past the last), by rows.
    let mut chunks: Vec<(usize, usize, usize)> = (0..nodes.len())
        .filter(|&k| {
            k != dot
                && nodes[k].primitive == nodes[dot].primitive
                && nodes[k].inputs[1] == nodes[dot].inputs[1]
        })
        .filter_map(|k| {
            let s = &nodes[producer[nodes[k].inputs[0]]?];
            match &s.primitive {
                Primitive::Slice {
                    start_indices,
                    limit_indices,
                    ..
                } if s.inputs[0] == f.h && start_indices[1] == 0 && limit_indices[1] == d => {
                    Some((k, start_indices[0], limit_indices[0]))
                }
                _ => None,
            }
        })
        .collect();
    chunks.sort_by_key(|c| c.1);
    let covered = chunks
        .iter()
        .try_fold(0, |row, c| (c.1 == row).then_some(c.2));
    if chunks.is_empty() || covered != Some(b) {
        return None;
    }
    // The forward's rows, copied for a chunk: the pattern (but the dot)
    // and its gather's index, `iota(B) * V + target`.
    let mut region: Vec<usize> = f.members.iter().copied().filter(|&k| k != dot).collect();
    let mut stack = vec![nodes[f.pick].inputs[1]];
    while let Some(v) = stack.pop() {
        if v == f.target {
            continue;
        }
        let k = producer[v]?;
        if !region.contains(&k) {
            region.push(k);
            stack.extend(&nodes[k].inputs);
        }
    }
    region.sort_unstable();
    let (lse, loss) = (nodes[f.lse].output, nodes[f.loss].output);
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    // Values waiting for every chunk's rows (the losses, the log-sum-exp,
    // what reads them), and those computed from a chunk's logits.
    let mut waiting = vec![false; graph.types.len()];
    let mut logits = vec![false; graph.types.len()];
    let mut deferred = Vec::new();
    // Each chunk's (first row, row past the last, log-sum-exp, losses).
    let mut rows: Vec<(usize, usize, Var, Var)> = Vec::new();
    let emit = |out: &mut Graph, map: &mut [Var], node: &Node| {
        out.set_origin(node);
        let ins: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out.apply(node.primitive.clone(), &ins).expect("well typed");
        out.set_label(map[node.output], node.label);
    };
    for (k, node) in nodes.iter().enumerate() {
        if f.members.contains(&k) {
            continue;
        }
        if let Some(&(_, start, end)) = chunks.iter().find(|c| c.0 == k) {
            // The chunk's logits, the forward's too (its dot's lines).
            out.set_origins(&[node, &nodes[dot]]);
            let ins: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
            map[node.output] = out.apply(node.primitive.clone(), &ins).ok()?;
            out.set_label(map[node.output], node.label);
            logits[node.output] = true;
            // The chunk's rows of the forward, of its logits.
            out.set_origin(&nodes[f.pick]);
            let target = Primitive::Slice {
                start_indices: vec![start],
                limit_indices: vec![end],
                view: false,
            };
            let mut local = vec![usize::MAX; graph.types.len()];
            local[nodes[dot].output] = map[node.output];
            local[f.target] = out.apply(target, &[map[f.target]]).ok()?;
            for &j in &region {
                let n = &nodes[j];
                let p = of_rows(&n.primitive, b, end - start)?;
                let ins: Vec<Var> = n
                    .inputs
                    .iter()
                    .map(|&v| {
                        if local[v] != usize::MAX {
                            local[v]
                        } else {
                            map[v]
                        }
                    })
                    .collect();
                out.set_origin(n);
                local[n.output] = out.apply(p, &ins).ok()?;
                out.set_label(local[n.output], n.label);
            }
            rows.push((start, end, local[lse], local[loss]));
            if rows.len() < chunks.len() {
                continue;
            }
            // Every chunk's: the losses and log-sum-exp (what reads them
            // after every chunk's gradient, read after its rows' kernel).
            rows.sort_by_key(|r| r.0);
            out.set_origin(&nodes[f.lse]);
            map[lse] = joined(&mut out, rows.iter().map(|r| r.2).collect())?;
            out.set_origin(&nodes[f.loss]);
            map[loss] = joined(&mut out, rows.iter().map(|r| r.3).collect())?;
            continue;
        }
        // A chunk's rows of the log-sum-exp or losses: its own.
        if let Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } = &node.primitive
            && let Some(r) = rows
                .iter()
                .find(|r| start_indices == &[r.0] && limit_indices == &[r.1])
        {
            match node.inputs[0] {
                v if v == lse => {
                    map[node.output] = r.2;
                    continue;
                }
                v if v == loss => {
                    map[node.output] = r.3;
                    continue;
                }
                _ => {}
            }
        }
        logits[node.output] = node.inputs.iter().any(|&v| logits[v]);
        if node
            .inputs
            .iter()
            .any(|&v| waiting[v] || v == lse || v == loss)
        {
            // A chunk's gradient waiting for every chunk's losses.
            if logits[node.output] {
                return None;
            }
            waiting[node.output] = true;
            deferred.push(k);
            continue;
        }
        emit(&mut out, &mut map, node);
    }
    if rows.len() < chunks.len() {
        return None;
    }
    for j in deferred {
        emit(&mut out, &mut map, &nodes[j]);
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).ok()?;
    Some(out.prune().0)
}

/// `values` (chunks' rows, in order) concatenated: at most 16 a
/// concatenate (a kernel binds few buffers), those concatenated in turn.
fn joined(out: &mut Graph, mut values: Vec<Var>) -> Option<Var> {
    while values.len() > 1 {
        values = values
            .chunks(16)
            .map(|part| match part {
                [v] => Some(*v),
                _ => out
                    .apply(
                        Primitive::Concatenate {
                            dimension: 0,
                            in_place: false,
                        },
                        part,
                    )
                    .ok(),
            })
            .collect::<Option<_>>()?;
    }
    values.pop()
}

/// `p` of a chunk of `c` of the `b` rows (each value's first dimension
/// a multiple of the rows), if it computes each row's from its own.
fn of_rows(p: &Primitive, b: usize, c: usize) -> Option<Primitive> {
    use Primitive::*;
    let rows = |shape: &[usize]| {
        let mut shape = shape.to_vec();
        if let Some(n) = shape.first_mut() {
            if *n % b != 0 {
                return None;
            }
            *n = *n / b * c;
        }
        Some(shape)
    };
    Some(match p {
        Reshape { new_sizes } => Reshape {
            new_sizes: rows(new_sizes)?,
        },
        BroadcastInDim {
            shape,
            broadcast_dimensions,
        } => BroadcastInDim {
            shape: rows(shape)?,
            broadcast_dimensions: broadcast_dimensions.clone(),
        },
        Iota {
            dtype,
            shape,
            dimension: 0,
        } => Iota {
            dtype: *dtype,
            shape: rows(shape)?,
            dimension: 0,
        },
        Full {
            shape,
            fill_value,
            dtype,
        } => Full {
            shape: rows(shape)?,
            fill_value: *fill_value,
            dtype: *dtype,
        },
        ReduceMax { .. } | ReduceSum { .. } | Gather { axis: 0 } | Sub | Add | Mul | Exp | Log => {
            p.clone()
        }
        _ => return None,
    })
}

/// The linear cross entropy of the dot at node `dot`, if it is one (see
/// the module's pattern) whose values but `lse` and `pick` nothing else
/// reads, its target defined before both.
fn matched(
    graph: &Graph,
    producer: &[Option<usize>],
    readers: &[Vec<usize>],
    dot: usize,
) -> Option<Found> {
    use Primitive::*;
    let nodes = graph.nodes();
    let node = &nodes[dot];
    let DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
        ..
    } = &node.primitive
    else {
        return None;
    };
    let (h, w) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));
    let transposed = rhs_contracting == &[0];
    let fits = lhs_batch.is_empty()
        && h.shape.len() == 2
        && w.shape.len() == 2
        && lhs_contracting == &[1]
        && (transposed || rhs_contracting == &[1])
        && h.dtype.is_float()
        && h.dtype == w.dtype;
    if !fits {
        return None;
    }
    let x = node.output;
    let (b, v) = (h.shape[0], graph.type_of(x).shape[1]);
    // The one node reading `v` (not an output), if it is one.
    let only = |v: Var| match (&readers[v][..], graph.outputs().contains(&v)) {
        (&[k], false) => Some(k),
        _ => None,
    };
    // The logits' readers: the max, the shift, the rows flattened.
    if readers[x].len() != 3 || graph.outputs().contains(&x) {
        return None;
    }
    let find = |f: &dyn Fn(&Node) -> bool| readers[x].iter().copied().find(|&k| f(&nodes[k]));
    let max = find(&|n| matches!(&n.primitive, ReduceMax { axes } if axes == &[1]))?;
    let shift = find(&|n| matches!(n.primitive, Sub) && n.inputs[0] == x)?;
    let flat = find(&|n| matches!(&n.primitive, Reshape { new_sizes } if new_sizes == &[b * v]))?;
    // The max's reshapes (to [B, 1], back to [B]): the broadcast for the
    // shift reads one, lse another (or the max).
    let mut views = vec![max];
    let mut k = 0;
    while k < views.len() {
        let u = nodes[views[k]].output;
        views.extend(
            readers[u]
                .iter()
                .filter(|&&r| matches!(nodes[r].primitive, Reshape { .. })),
        );
        k += 1;
    }
    let values: Vec<Var> = views.iter().map(|&k| nodes[k].output).collect();
    let broadcast = nodes[shift].inputs[1];
    let broadcast = producer[broadcast].filter(|&k| {
        matches!(&nodes[k].primitive, BroadcastInDim { shape, broadcast_dimensions } if shape == &[b, v] && broadcast_dimensions == &[0, 1])
            && values.contains(&nodes[k].inputs[0])
            && only(nodes[k].output) == Some(shift)
    })?;
    // exp, its sum over the classes, its log, plus the max.
    let exp = only(nodes[shift].output).filter(|&k| matches!(nodes[k].primitive, Exp))?;
    let sum = only(nodes[exp].output).filter(|&k| matches!(&nodes[k].primitive, ReduceSum { axes, accum_dtype: DType::F32 } if axes == &[1]))?;
    let log = only(nodes[sum].output).filter(|&k| matches!(nodes[k].primitive, Log))?;
    let lse = only(nodes[log].output).filter(|&k| matches!(nodes[k].primitive, Add))?;
    let max_read = match nodes[lse].inputs[..] {
        [l, r] if r == nodes[log].output => l,
        [l, r] if l == nodes[log].output => r,
        _ => return None,
    };
    if !values.contains(&max_read) || graph.type_of(max_read).shape != [b] {
        return None;
    }
    // Nothing else reads the max.
    let inside = |r: &usize| views.contains(r) || *r == broadcast || *r == lse;
    if values
        .iter()
        .any(|&u| graph.outputs().contains(&u) || !readers[u].iter().all(inside))
    {
        return None;
    }
    // The class's logit: the rows flattened at row * V + target.
    let pick = only(nodes[flat].output).filter(|&k| {
        matches!(nodes[k].primitive, Gather { axis: 0 }) && nodes[k].inputs[0] == nodes[flat].output
    })?;
    let target = row_offsets_plus(graph, producer, nodes[pick].inputs[1], b, v)?;
    // The loss: lse less the class's logit, its only reader.
    let loss = only(nodes[pick].output).filter(|&k| {
        matches!(nodes[k].primitive, Sub)
            && nodes[k].inputs == [nodes[lse].output, nodes[pick].output]
    })?;
    let defined = producer[target].is_none_or(|k| k < lse.min(pick));
    if !defined {
        return None;
    }
    Some(Found {
        h: node.inputs[0],
        w: node.inputs[1],
        transposed,
        target,
        members: [
            vec![dot, shift, broadcast, exp, sum, log, lse, flat, pick, loss],
            views,
        ]
        .concat(),
        loss,
        lse,
        first: lse.min(pick),
        pick,
    })
}

/// `target` if `at` is `iota(B) * V + target` (`target` `[B]`, int32 or
/// int64, either operand order).
fn row_offsets_plus(
    graph: &Graph,
    producer: &[Option<usize>],
    at: Var,
    b: usize,
    v: usize,
) -> Option<Var> {
    use Primitive::*;
    let nodes = graph.nodes();
    let node = |u: Var| producer[u].map(|k| &nodes[k]);
    let add = node(at).filter(|n| matches!(n.primitive, Add))?;
    let offsets = |u: Var| {
        let Some(mul) = node(u).filter(|n| matches!(n.primitive, Mul)) else {
            return false;
        };
        let iota = |u: Var| {
            node(u).is_some_and(
                |n| matches!(&n.primitive, Iota { shape, dimension: 0, .. } if shape == &[b]),
            )
        };
        let classes = |u: Var| {
            node(u).is_some_and(|n| {
                matches!(n.primitive, BroadcastInDim { .. })
                    && node(n.inputs[0]).is_some_and(|f| matches!(f.primitive, Full { fill_value: Scalar::Int(c), .. } if c == v as i64))
            })
        };
        let [l, r] = mul.inputs[..] else { return false };
        (iota(l) && classes(r)) || (iota(r) && classes(l))
    };
    let [l, r] = add.inputs[..] else { return None };
    let target = match (offsets(l), offsets(r)) {
        (true, false) => r,
        (false, true) => l,
        _ => return None,
    };
    let t = graph.type_of(target);
    (t.shape == [b] && matches!(t.dtype, DType::I32 | DType::I64)).then_some(target)
}
