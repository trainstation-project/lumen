//! A linear layer's cross entropy, matched as written: `F.cross_entropy`
//! of a dot's logits (`F.matmul(h, w.t(), "float32", "float32")`, an LM
//! head's), however traced, run as [`Primitive::LinearCrossEntropy`] (two
//! kernels, Cut Cross-Entropy's) when nothing else reads the logits: they
//! never reach memory. As flash attention's ([`crate::compiler::attention`]),
//! no call of its own needed (`F.linear_cross_entropy` traces as this).
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
}

/// `graph` with each linear cross entropy whose logits nothing else reads
/// as a [`Primitive::LinearCrossEntropy`].
pub(crate) fn linear_cross_entropy(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let mut producer = vec![None; graph.types.len()];
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
        for &v in &node.inputs {
            if !readers[v].contains(&i) {
                readers[v].push(i);
            }
        }
    }
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
            };
            let v = apply(&mut out, slice, &[rows]);
            map[nodes[k].output] = apply(&mut out, Primitive::Reshape { new_sizes: vec![b] }, &[v]);
        }
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
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
