//! `SplitK` (pass name `split_k`) splits a matmul with a long contraction but too few output tiles to fill
//! the GPU into partial products over K chunks, summed over the chunks. It runs after `linear_reordering`
//! and before `canonicalize_dots`, behind the `split_k` config flag.
//!
//! ## Rewrite
//!
//! A dot is split when it is unbatched, has one contracting dim per side, accumulates in a float dtype,
//! neither operand is a `Slice` output, and the backend's `unsplit` rule does not keep it whole.
//!
//! - The chunk count is a power of two from `target / tiles(m, n)`, capped at 8, and only when the output has
//!   fewer than half the target tiles and K is at least 2048. A backend's preferred count (`Rules::chunks`)
//!   wins when larger.
//! - `divides` halves the count until every chunk has at least 256 elements and is a multiple of 8 elements
//!   (whole 16-byte vectors of 2-byte elements); fewer than 2 chunks means no split.
//! - Each operand's contracting dim `d` is reshaped to `[s, k / s]`; the dot then batches over the chunk
//!   index and contracts the chunk length, writing the accumulation dtype. A `ReduceSum` over axis 0 sums the
//!   chunks, and a `Cast` follows when the original output dtype differs from the accumulator.
//! - The new nodes are labelled `dot_general (split-K)` and `reduce_sum (split-K)`.
//!
//! Devices: it runs, with `split_k`, on each device whose codegen implements `codegen::dot_general::SplitsK` (its tiles, target,
//! the products it keeps whole and a chunk count it prefers): MPS (tiles counted with the GEMM's `float_groups`, a
//! target of 256 threadgroups, GEMV and skinny products of at most 4 rows kept whole, MLX's preferred chunks) and
//! Triton (tiles of the untuned GEMM config, a target of 128 tiles, matrix-vector products kept whole for the GEMV
//! kernel; partials always summed by a reduction, so results stay deterministic). The others get the identity pass.
use crate::codegen::dot_general::SplitsK;
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::CompilerPass;

/// Most K chunks a matmul is split into.
pub(crate) const MAX_SPLIT: usize = 8;
/// Smallest K chunk length.
const MIN_CHUNK: usize = 256;
/// Smallest K worth splitting.
const MIN_K: usize = 2048;
/// Elements every K chunk is a multiple of: a 16-byte vector of 2-byte elements, so the split product
/// keeps its vector loads (M4 Pro, f32 300 x 20000 x 200: 32 chunks of 625 run 25% slower than 4 of 5000).
const CHUNK_ALIGN: usize = 8;

/// A backend's rules for splitting K.
pub(crate) struct Rules<'a> {
    /// Output tiles an `m` x `n` matmul runs as, each one kernel instance.
    pub(crate) tiles: &'a dyn Fn(usize, usize) -> usize,
    /// Tiles a split aims to fill the GPU with; a matmul of half as many or more is not split.
    pub(crate) target: usize,
    /// Whether the backend runs a dot of these operand and output types on a kernel of its own, unsplit.
    pub(crate) unsplit: &'a dyn Fn(&Primitive, &TensorType, &TensorType, &TensorType) -> bool,
    /// A chunk count the backend would rather split an `m` x `n` output with contraction `k` into, when
    /// it is split at all: the larger of it and the `target` count is used.
    pub(crate) chunks: Option<&'a dyn Fn(usize, usize, usize) -> Option<usize>>,
}

/// Rewrites eligible unbatched float matmuls into a batched dot over K chunks plus a sum over chunks,
/// by the backend's `rules`.
pub(crate) fn split_k(graph: &Graph, rules: &Rules) -> Graph {
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
                let k = lhs.shape[l];

                let base = chunks((rules.tiles)(m, n), rules.target, k);
                let wanted = rules.chunks.and_then(|f| f(m, n, k));

                match (base, wanted) {
                    (Some(b), Some(w)) => divides(b.max(w), k),
                    (b, _) => b,
                }
                .map(|s| (l, r, s))
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

        let partials = apply(&mut out, dot, &[a, b]);
        out.set_label(partials, Some("dot_general (split-K)"));
        let mut value = apply(&mut out, sum, &[partials]);
        out.set_label(value, Some("reduce_sum (split-K)"));

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

    let s = target.div_ceil(groups).next_power_of_two().min(MAX_SPLIT);
    divides(s, k)
}

/// The largest power of two up to `s` dividing `k` into chunks of at least `MIN_CHUNK`, each a multiple
/// of `CHUNK_ALIGN` elements (whole vector loads), if at least 2.
pub(crate) fn divides(mut s: usize, k: usize) -> Option<usize> {
    while s >= 2 && (!k.is_multiple_of(s * CHUNK_ALIGN) || k / s < MIN_CHUNK) {
        s /= 2;
    }

    (s >= 2).then_some(s)
}

/// Splits `graph`'s matmuls by device `D`'s rules (`SplitsK`).
pub(crate) fn split<D: SplitsK>(graph: &Graph) -> Graph {
    let rules = Rules {
        tiles: &|m, n| D::tiles(m, n),
        target: D::TARGET,
        unsplit: &|dot, lhs, rhs, out| D::unsplit(dot, lhs, rhs, out),
        chunks: Some(&|m, n, k| D::chunks(m, n, k)),
    };

    split_k(graph, &rules)
}

/// Compiler pass splitting long-K matmuls with too few output tiles, by the device's rules.
pub(crate) struct SplitK;

impl<D: SplitsK> CompilerPass<DeviceModule<'_, D>> for SplitK {
    /// With `split_k`.
    fn configured(config: &CompilerConfig) -> Option<Self> {
        (config.split_k).then_some(SplitK)
    }

    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "split_k"
    }

    /// Splits the module's matmuls.
    fn apply(&mut self, m: &mut DeviceModule<'_, D>) -> Result<(), String> {
        m.graph = split::<D>(&m.graph);
        Ok(())
    }
}
