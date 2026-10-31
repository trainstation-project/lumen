//! Split-K shared by the backends: long-K matmuls with too few output tiles become partial products
//! over K chunks summed over the chunks.

use crate::DType;
use crate::graph::{Graph, Node, Primitive, TensorType, Var};

/// Most K chunks a matmul is split into.
const MAX_SPLIT: usize = 8;
/// Smallest K chunk length.
const MIN_CHUNK: usize = 256;
/// Smallest K worth splitting.
const MIN_K: usize = 2048;

/// A backend's rules for splitting K.
pub(crate) struct Rules<'a> {
    /// Output tiles (threadgroups or programs) an `m` x `n` matmul runs as.
    pub(crate) tiles: &'a dyn Fn(usize, usize) -> usize,
    /// Tiles a split aims to fill the GPU with; a matmul of half as many or more is not split.
    pub(crate) target: usize,
    /// Whether the backend runs a dot of these operand and output types on a kernel of its own, unsplit.
    pub(crate) unsplit: &'a dyn Fn(&Primitive, &TensorType, &TensorType, &TensorType) -> bool,
    /// Whether a node fuses into the kernel computing its operands.
    pub(crate) fusible: &'a dyn Fn(&Graph, &Node) -> bool,
    /// Whether an f32 split whose partials outweigh its operands may accumulate atomically.
    pub(crate) atomic: bool,
    /// Whether a fusible reader keeps a split out of the atomic form (the sum takes it as an epilogue).
    pub(crate) fuse: bool,
}

/// Rewrites eligible unbatched float matmuls into a batched dot over K chunks plus a sum over chunks,
/// by the backend's `rules`. With `rules.atomic`, an f32 split whose partials would outweigh its
/// operands becomes a `split_k` fusion that accumulates atomically, unless `rules.fuse` is set and
/// a fusible node reads the result.
pub(crate) fn split_k(graph: &Graph, rules: &Rules) -> Graph {
    let nodes = graph.nodes();

    let sliced = |v: Var| {
        nodes
            .iter()
            .any(|n| n.output == v && matches!(n.primitive, Primitive::Slice { .. }))
    };

    let fused = |v: Var| {
        nodes
            .iter()
            .any(|n| n.inputs.contains(&v) && (rules.fusible)(graph, n))
    };

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    for node in nodes {
        out.set_origin(node);
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

        let apply =
            |out: &mut Graph, p: Primitive, ins: &[Var]| out.apply(p, ins).expect("well typed");

        let Primitive::DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            accum_dtype,
            output_dtype,
            ..
        } = &node.primitive
        else {
            map[node.output] = apply(&mut out, node.primitive.clone(), &inputs);
            out.set_label(map[node.output], node.label);
            continue;
        };

        let (lhs, rhs) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));

        let split = match (&lhs_contracting[..], &rhs_contracting[..]) {
            (&[l], &[r])
                if lhs_batch.is_empty()
                    && accum_dtype.is_float()
                    && !node.inputs.iter().any(|&v| sliced(v))
                    && !(rules.unsplit)(&node.primitive, lhs, rhs, graph.type_of(node.output)) =>
            {
                let m = lhs.numel() / lhs.shape[l];
                let n = rhs.numel() / rhs.shape[r];
                chunks((rules.tiles)(m, n), rules.target, lhs.shape[l]).map(|s| (l, r, s))
            }
            _ => None,
        };

        let Some((l, r, s)) = split else {
            map[node.output] = apply(&mut out, node.primitive.clone(), &inputs);
            out.set_label(map[node.output], node.label);
            continue;
        };

        // Split contracting dim `d` of `shape` into `[s, k / s]`, the chunk index becoming a batch dim.
        let reshaped = |shape: &[usize], d: usize| {
            let mut shape = shape.to_vec();
            let k = shape[d];
            shape.splice(d..=d, [s, k / s]);
            Primitive::Reshape { new_sizes: shape }
        };

        let a = apply(&mut out, reshaped(&lhs.shape, l), &[inputs[0]]);
        let b = apply(&mut out, reshaped(&rhs.shape, r), &[inputs[1]]);

        let dot = Primitive::DotGeneral {
            lhs_contracting: vec![l + 1],
            rhs_contracting: vec![r + 1],
            lhs_batch: vec![l],
            rhs_batch: vec![r],
            accum_dtype: *accum_dtype,
            output_dtype: *accum_dtype,
        };

        let sum = Primitive::ReduceSum {
            axes: vec![0],
            accum_dtype: *accum_dtype,
        };

        // Bytes of the partial products written and read back, against the operand bytes.
        let partials = 2 * s * lhs.numel() / lhs.shape[l]
            * (rhs.numel() / rhs.shape[r])
            * accum_dtype.size_of();

        let operands = (lhs.numel() + rhs.numel()) * lhs.dtype.size_of();

        let atomic =
            rules.atomic && *accum_dtype == DType::F32 && !(rules.fuse && fused(node.output));

        let mut value = match atomic && partials > operands {
            true => {
                let mut body = Graph::new();
                let ins = [a, b].map(|v| body.input(out.type_of(v).clone()));
                let partials = body.apply(dot, &ins).expect("well typed");
                let sum = body.apply(sum, &[partials]).expect("well typed");
                body.set_outputs(&[sum]).expect("its value");

                let fusion = Primitive::Fusion {
                    name: "split_k".into(),
                    label: "dot_general (split-K)",
                    body,
                };

                out.apply(fusion, &[a, b]).expect("well typed")
            }
            false => {
                let partials = apply(&mut out, dot, &[a, b]);
                out.set_label(partials, Some("dot_general (split-K)"));
                let sum = apply(&mut out, sum, &[partials]);
                out.set_label(sum, Some("reduce_sum (split-K)"));
                sum
            }
        };

        if output_dtype != accum_dtype {
            value = apply(
                &mut out,
                Primitive::Cast {
                    new_dtype: *output_dtype,
                },
                &[value],
            );
        }

        map[node.output] = value;
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Number of K chunks for an output of `groups` tiles with contraction `k`, aiming at `target` tiles,
/// or `None` when there are already enough tiles, K is short, or no power-of-two split leaves chunks
/// of at least `MIN_CHUNK`.
fn chunks(groups: usize, target: usize, k: usize) -> Option<usize> {
    if groups == 0 || 2 * groups >= target || k < MIN_K {
        return None;
    }

    let mut s = target.div_ceil(groups).next_power_of_two().min(MAX_SPLIT);

    while s >= 2 && (!k.is_multiple_of(s) || k / s < MIN_CHUNK) {
        s /= 2;
    }

    (s >= 2).then_some(s)
}
