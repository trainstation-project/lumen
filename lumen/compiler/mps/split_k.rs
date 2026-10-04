//! Split-K (XLA: `SplitKRewriter`): a float dot of few output tiles and a
//! long contraction (small M and N, large K: a decode step's) leaves most of
//! the GPU idle, each threadgroup looping over all of K. Its K is split into
//! `S` chunks, a new batch dimension (a reshape of each operand: no copy):
//! one dot writes the `S` partial products in float32, its accumulation
//! dtype, a sum adds them in it, and a cast rounds to the dot's output
//! dtype once. Not what the program computes (the partials are added in
//! another order): `lumen.config.compiler.split_k`.

use crate::DType;
use crate::graph::{Graph, Primitive, Var};
use crate::ops::dot_general::mps::SMALL_TILE;

/// Dots of fewer output tiles (of the small matmul kernel's) are split.
const MAX_TILES: usize = 64;
/// The threadgroups a split dot launches, at most (on an M4 Pro: about 13
/// per GPU core).
const GROUPS: usize = 256;
/// The most chunks.
const MAX_SPLIT: usize = 32;
/// The fewest elements of K a chunk takes.
const MIN_CHUNK: usize = 256;

/// `graph` with each dot of few output tiles and a long contraction split
/// along it.
pub(crate) fn split_k(graph: &Graph) -> Graph {
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
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let mut apply = |p: Primitive, ins: &[Var]| out.apply(p, ins).expect("well typed");
        let Primitive::DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            accum_dtype,
            output_dtype,
            ..
        } = &node.primitive
        else {
            map[node.output] = apply(node.primitive.clone(), &inputs);
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
            map[node.output] = apply(node.primitive.clone(), &inputs);
            continue;
        };
        // Contracting dimension d as [S, K / S]: S a batch dimension.
        let reshaped = |shape: &[usize], d: usize| {
            let mut shape = shape.to_vec();
            let k = shape[d];
            shape.splice(d..=d, [s, k / s]);
            Primitive::Reshape { new_sizes: shape }
        };
        let a = apply(reshaped(&lhs.shape, l), &[inputs[0]]);
        let b = apply(reshaped(&rhs.shape, r), &[inputs[1]]);
        let dot = Primitive::DotGeneral {
            lhs_contracting: vec![l + 1],
            rhs_contracting: vec![r + 1],
            lhs_batch: vec![l],
            rhs_batch: vec![r],
            accum_dtype: DType::F32,
            output_dtype: DType::F32,
        };
        let partials = apply(dot, &[a, b]);
        let sum = Primitive::ReduceSum {
            axes: vec![0],
            accum_dtype: DType::F32,
        };
        let mut value = apply(sum, &[partials]);
        if *output_dtype != DType::F32 {
            value = apply(
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

/// The chunks to split the contraction of an `m` by `k` by `n` matmul
/// into, if it has few output tiles: as many as fill [`GROUPS`]
/// threadgroups (a power of two, at most [`MAX_SPLIT`], dividing `k` into
/// chunks of at least [`MIN_CHUNK`]).
fn chunks(m: usize, n: usize, k: usize) -> Option<usize> {
    let tiles = m.div_ceil(SMALL_TILE.0) * n.div_ceil(SMALL_TILE.1);
    if tiles == 0 || tiles >= MAX_TILES {
        return None;
    }
    let mut s = 1 << (GROUPS / tiles).min(MAX_SPLIT).ilog2();
    while s >= 2 && (!k.is_multiple_of(s) || k / s < MIN_CHUNK) {
        s /= 2;
    }
    (s >= 2).then_some(s)
}
