//! Pass replacing an unfused `cross_entropy(h @ w, target)` with one `LinearCrossEntropy` op that
//! never materializes the logits.

use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::passes::{CompilerPass, Module};
use crate::{DType, Scalar};

/// A matched linear cross entropy.
pub(super) struct Found {
    /// Hidden states `[b, d]`.
    pub(super) h: Var,
    /// Weight, `[v, d]` or `[d, v]` when `transposed`.
    w: Var,
    /// The dot contracts dim 0 of the weight, so it is `[d, v]`.
    transposed: bool,
    /// Class index per row.
    pub(super) target: Var,
    /// Nodes the fused op replaces.
    pub(super) members: Vec<usize>,
    /// Node computing the per-row loss `lse - logit[target]`.
    pub(super) loss: usize,
    /// Node computing the per-row logsumexp.
    pub(super) lse: usize,
    /// Earliest node producing a result, where the fused op is emitted.
    first: usize,
    /// The gather reading the target logits.
    pub(super) pick: usize,
}

/// Emits the `LinearCrossEntropy` of `h`, weight `w` (`[d, v]` when `transposed`) and `target` into `out`.
fn emit(out: &mut Graph, h: Var, w: Var, transposed: bool, target: Var) -> Var {
    let w = match transposed {
        true => {
            let transpose = Primitive::Transpose {
                permutation: vec![1, 0],
            };

            out.apply(transpose, &[w]).expect("well typed")
        }
        false => w,
    };

    out.apply(Primitive::LinearCrossEntropy, &[h, w, target])
        .expect("well typed")
}

/// Row `at` (0: the loss, 1: the lse) of a `LinearCrossEntropy`'s `[2, b]` result `rows`, as a `[b]` value.
fn row(out: &mut Graph, rows: Var, at: usize, b: usize) -> Var {
    let slice = Primitive::Slice {
        start_indices: vec![at, 0],
        limit_indices: vec![at + 1, b],
        view: false,
    };

    let v = out.apply(slice, &[rows]).expect("well typed");

    out.apply(Primitive::Reshape { new_sizes: vec![b] }, &[v])
        .expect("well typed")
}

/// Producer node of each value and the nodes reading it.
pub(super) fn uses(graph: &Graph) -> (Vec<Option<usize>>, Vec<Vec<usize>>) {
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

/// The max-shifted logsumexp along the classes of logits `h @ w` computed by dot node `dot`.
struct Logsumexp {
    /// Hidden states `[b, d]`.
    h: Var,
    /// Weight, `[v, d]` or `[d, v]` when `transposed`.
    w: Var,
    /// The dot contracts dim 0 of the weight, so it is `[d, v]`.
    transposed: bool,
    /// Rows `b` and classes `v`.
    b: usize,
    v: usize,
    /// The dot, the shift, its broadcast, exp, sum, log and the lse, then the row max and its reshapes.
    members: Vec<usize>,
    /// Node computing the per-row logsumexp.
    lse: usize,
    /// The reshape flattening the logits for the target gather, the logits' one other reader.
    flat: Option<usize>,
}

/// Matches the logsumexp of logits `h @ w` computed by dot node `dot`: the logits read only by the
/// row max, the shift and (for a loss's target gather) a flattening reshape.
fn logsumexp(
    graph: &Graph,
    producer: &[Option<usize>],
    readers: &[Vec<usize>],
    dot: usize,
) -> Option<Logsumexp> {
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

    let only = |v: Var| match (&readers[v][..], graph.outputs().contains(&v)) {
        (&[k], false) => Some(k),
        _ => None,
    };

    let find = |f: &dyn Fn(&Node) -> bool| readers[x].iter().copied().find(|&k| f(&nodes[k]));
    let max = find(&|n| matches!(&n.primitive, ReduceMax { axes } if axes == &[1]))?;
    let shift = find(&|n| matches!(n.primitive, Sub) && n.inputs[0] == x)?;
    let flat = find(&|n| matches!(&n.primitive, Reshape { new_sizes } if new_sizes == &[b * v]));

    // The logits are read exactly by the row max, the shift and the flattening reshape, if any.
    if readers[x].len() != 2 + usize::from(flat.is_some()) || graph.outputs().contains(&x) {
        return None;
    }

    // The row max and its reshapes may only feed the shift's broadcast and the final lse add.
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

    let inside = |r: &usize| views.contains(r) || *r == broadcast || *r == lse;

    if values
        .iter()
        .any(|&u| graph.outputs().contains(&u) || !readers[u].iter().all(inside))
    {
        return None;
    }

    Some(Logsumexp {
        h: node.inputs[0],
        w: node.inputs[1],
        transposed,
        b,
        v,
        members: [vec![dot, shift, broadcast, exp, sum, log, lse], views].concat(),
        lse,
        flat,
    })
}

/// Matches the cross entropy over logits `h @ w` computed by dot node `dot`: max-shifted
/// `logsumexp` along the classes, and the target logits gathered from the flattened logits.
pub(super) fn matched(
    graph: &Graph,
    producer: &[Option<usize>],
    readers: &[Vec<usize>],
    dot: usize,
) -> Option<Found> {
    use Primitive::*;
    let nodes = graph.nodes();
    let l = logsumexp(graph, producer, readers, dot)?;
    let flat = l.flat?;

    let only = |v: Var| match (&readers[v][..], graph.outputs().contains(&v)) {
        (&[k], false) => Some(k),
        _ => None,
    };

    let pick = only(nodes[flat].output).filter(|&k| {
        matches!(nodes[k].primitive, Gather { axis: 0 }) && nodes[k].inputs[0] == nodes[flat].output
    })?;

    let target = row_offsets_plus(graph, producer, nodes[pick].inputs[1], l.b, l.v)?;

    let loss = only(nodes[pick].output).filter(|&k| {
        matches!(nodes[k].primitive, Sub)
            && nodes[k].inputs == [nodes[l.lse].output, nodes[pick].output]
    })?;

    // The target must exist before the fused op is emitted.
    let defined = producer[target].is_none_or(|k| k < l.lse.min(pick));

    if !defined {
        return None;
    }

    Some(Found {
        h: l.h,
        w: l.w,
        transposed: l.transposed,
        target,
        members: [l.members, vec![flat, pick, loss]].concat(),
        loss,
        lse: l.lse,
        first: l.lse.min(pick),
        pick,
    })
}

/// The target `t` when index `at` is `iota(b) * v + t`, with `t` an int32/int64 `[b]` tensor.
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

/// Compiler pass running [`linear_cross_entropy`].
pub(crate) struct LinearCrossEntropy;

impl<M: Module> CompilerPass<M> for LinearCrossEntropy {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "linear_cross_entropy"
    }

    /// Replaces every matched cross entropy with a `LinearCrossEntropy` whose `[2, b]` result holds the
    /// loss row and the lse row, and every logsumexp of logits nothing else reads (a gradient's, its loss
    /// unused) with one of class-0 targets, its lse row alone used.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let graph = m.graph();

        let nodes = graph.nodes();
        let (producer, readers) = uses(graph);

        let found: Vec<Found> = (0..nodes.len())
            .filter_map(|i| matched(graph, &producer, &readers, i))
            .collect();

        let alone: Vec<Logsumexp> = (0..nodes.len())
            .filter_map(|i| logsumexp(graph, &producer, &readers, i))
            .filter(|l| l.flat.is_none())
            .collect();

        if found.is_empty() && alone.is_empty() {
            return Ok(());
        }

        let mut out = Graph::new();
        let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

        for &v in graph.inputs() {
            map[v] = out.input(graph.type_of(v).clone());
        }

        for (i, node) in nodes.iter().enumerate() {
            if let Some(l) = alone.iter().find(|l| l.members.contains(&i)) {
                if i != l.lse {
                    continue;
                }

                let members: Vec<&Node> = l.members.iter().map(|&k| &nodes[k]).collect();
                out.set_origins(&members);

                let classes = Primitive::Full {
                    shape: vec![l.b],
                    fill_value: Scalar::Int(0),
                    dtype: DType::I32,
                };

                let target = out.apply(classes, &[]).expect("well typed");
                let rows = emit(&mut out, map[l.h], map[l.w], l.transposed, target);
                map[node.output] = row(&mut out, rows, 1, l.b);
                continue;
            }

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

            if i != f.first {
                continue;
            }

            let members: Vec<&Node> = f.members.iter().map(|&k| &nodes[k]).collect();
            out.set_origins(&members);

            let rows = emit(&mut out, map[f.h], map[f.w], f.transposed, map[f.target]);
            let b = graph.type_of(f.h).shape[0];

            for (k, at) in [(f.loss, 0), (f.lse, 1)] {
                map[nodes[k].output] = row(&mut out, rows, at, b);
            }
        }

        let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
        out.set_outputs(&outputs).expect("values of the graph");

        *m.graph_mut() = out;
        Ok(())
    }
}
