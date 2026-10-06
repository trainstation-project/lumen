//! Split-K (XLA: `SplitKRewriter`): a float dot of few output tiles and a
//! long contraction (small M and N, large K: a decode step's) leaves most of
//! the GPU idle, each threadgroup looping over all of K. Its K is split into
//! `S` chunks, a new batch dimension (a reshape of each operand: no copy):
//! one dot writes the `S` partial products in float32, its accumulation
//! dtype, a sum adds them in it, and a cast rounds to the dot's output
//! dtype once. Not what the program computes (the partials are added in
//! another order): `lumen.config.compiler.split_k`.
//!
//! With `atomic` (unless `lumen.config.compiler.deterministic`), the dot
//! and its sum are one fusion ([`atomic_dot`]): its kernel adds each
//! chunk's products to the float32 output (zeroed first) atomically, in no
//! fixed order, rather than writing them for a sum to read back.

use crate::DType;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::ops::dot_general::mps::SMALL_TILE;

/// Dots of fewer output tiles (of the small matmul kernel's: 32 x 32) are
/// split.
const MAX_TILES: usize = 256;
/// The threadgroups a split dot launches, at most (on an M4 Pro: about 13
/// per GPU core).
const GROUPS: usize = 256;
/// The most chunks.
const MAX_SPLIT: usize = 32;
/// The fewest elements of K a chunk takes.
const MIN_CHUNK: usize = 64;
/// The shortest contraction split (a weight's gradient's, over a batch's
/// tokens): a shorter one is a few steps of the matmul's loop.
const MIN_K: usize = 512;

/// `graph` with each dot of few output tiles and a long contraction split
/// along it, its chunks added atomically if `atomic`.
pub(crate) fn split_k(graph: &Graph, atomic: bool) -> Graph {
    let nodes = graph.nodes();
    let sliced = |v: Var| {
        nodes
            .iter()
            .any(|n| n.output == v && matches!(n.primitive, Primitive::Slice { .. }))
    };
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in nodes {
        out.set_scope(node.scope);
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
            continue;
        };
        let (lhs, rhs) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));
        let split = match (&lhs_contracting[..], &rhs_contracting[..]) {
            // No batch dimensions (the split's would not collapse with
            // them), operands read whole (not a slice in place).
            (&[l], &[r])
                if lhs_batch.is_empty()
                    && *accum_dtype == DType::F32
                    && !node.inputs.iter().any(|&v| sliced(v)) =>
            {
                let m = lhs.numel() / lhs.shape[l];
                let n = rhs.numel() / rhs.shape[r];
                chunks(m, n, lhs.shape[l]).map(|s| (l, r, s))
            }
            _ => None,
        };
        let Some((l, r, s)) = split else {
            map[node.output] = apply(&mut out, node.primitive.clone(), &inputs);
            continue;
        };
        // Contracting dimension d as [S, K / S]: S a batch dimension.
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
            accum_dtype: DType::F32,
            output_dtype: DType::F32,
        };
        let sum = Primitive::ReduceSum {
            axes: vec![0],
            accum_dtype: DType::F32,
        };
        let mut value = match atomic {
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
                // Labelled as what they are (the atomic one's fusion is
                // `dot_general (split-K)`): ordinary primitives otherwise.
                let partials = apply(&mut out, dot, &[a, b]);
                out.set_label(partials, Some("dot_general (split-K)"));
                let sum = apply(&mut out, sum, &[partials]);
                out.set_label(sum, Some("reduce_sum (split-K)"));
                sum
            }
        };
        if *output_dtype != DType::F32 {
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

/// The dot of a fusion's `body` if it is a split dot whose chunks are added
/// atomically (see [`split_k`]): the dot, then the sum over its chunks.
pub(crate) fn atomic_dot(body: &Graph) -> Option<&Node> {
    match body.nodes() {
        [dot, sum] => {
            let split = matches!(dot.primitive, Primitive::DotGeneral { .. })
                && matches!(&sum.primitive, Primitive::ReduceSum { axes, .. } if axes == &[0])
                && sum.inputs == [dot.output];
            split.then_some(dot)
        }
        _ => None,
    }
}

/// The chunks to split the contraction of an `m` by `k` by `n` matmul
/// into, if it has few output tiles: as many as fill [`GROUPS`]
/// threadgroups (a power of two, at most [`MAX_SPLIT`], dividing `k`, of
/// at least [`MIN_K`], into chunks of at least [`MIN_CHUNK`]).
fn chunks(m: usize, n: usize, k: usize) -> Option<usize> {
    let tiles = m.div_ceil(SMALL_TILE.0) * n.div_ceil(SMALL_TILE.1);
    if tiles == 0 || tiles >= MAX_TILES || k < MIN_K {
        return None;
    }
    let mut s = 1 << (GROUPS / tiles).min(MAX_SPLIT).ilog2();
    while s >= 2 && (!k.is_multiple_of(s) || k / s < MIN_CHUNK) {
        s /= 2;
    }
    (s >= 2).then_some(s)
}
