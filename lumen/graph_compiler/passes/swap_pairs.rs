//! Pass rewriting a swap of a last axis's partners written as slices and a concatenation into one `SwapPairs`:
//! `concatenate([slice(p, [.., δ:2δ]), slice(p, [.., 0:δ])], last)` of a value `p` whose last axis has length `2δ`
//! (`δ` a power of two) becomes `swap_pairs(p, distance=δ)`. With `δ` 1 that is an interleaved rotary embedding's
//! `cat([x[..., 1:], x[..., :1]])` over its pairs; with `δ` `d / 2` a rotate-half rotary embedding's
//! `cat([x[..., d/2:], x[..., :d/2]])` over its heads. When the first part is negated
//! (`cat([-x[..., d/2:], x[..., :d/2]])`, rotate-half's sign), the swap is multiplied by `-1` on the first `δ` elements
//! and `1` on the rest, an exact sign flip. A kernel then reads each element's partner at index `j ^ δ`, or, when the
//! value is in registers (a GEMM accumulator), swaps the partners in place; slices and a concatenation would need the
//! value in memory.
//!
//! Devices: it runs on each device whose codegen implements `codegen::emitter::SwapsPairs` (MPS and Triton), right
//! after the full `simplify`. Other devices keep the slices and concatenation.

use crate::codegen::emitter::SwapsPairs;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::{CompilerPass, Module};
use crate::{DType, Scalar};

/// Compiler pass turning each pair swap written with slices and a concatenation into `SwapPairs`.
pub(crate) struct SwapPairs;

impl<D: SwapsPairs> CompilerPass<DeviceModule<'_, D>> for SwapPairs {
    /// Always.
    fn configured(_: &CompilerConfig) -> Option<Self> {
        Some(SwapPairs)
    }

    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "swap_pairs"
    }

    /// Rewrites every pair swap of the module's graph.
    fn apply(&mut self, m: &mut DeviceModule<'_, D>) -> Result<(), String> {
        if let Some(graph) = swapped(m.graph()) {
            *m.graph_mut() = graph;
        }

        Ok(())
    }
}

/// A swap `node` makes: the value whose partners it swaps, their distance `δ`, and whether the first part is negated
/// (`concatenate([-slice(p, [.., δ:2δ]), slice(p, [.., 0:δ])])`), when it is one: a concatenation along the last axis of
/// a `p` whose last axis has length `2δ` (`δ` a power of two), each slice whole on the other axes.
fn swap_of(graph: &Graph, producer: &[Option<usize>], node: &Node) -> Option<(Var, usize, bool)> {
    let Primitive::Concatenate { dimension, .. } = node.primitive else {
        return None;
    };

    let [a, b] = node.inputs[..] else {
        return None;
    };

    // The slice making `v` (through a negation when `negated` allows one): its value, part and whether negated.
    let part = |v: Var, negated: bool| {
        let mut slice = &graph.nodes()[producer[v]?];
        let mut sign = false;

        // A negated part is undone by a sign multiply, so only for floats.
        if negated
            && matches!(slice.primitive, Primitive::Neg)
            && graph.type_of(slice.output).dtype.is_float()
        {
            slice = &graph.nodes()[producer[slice.inputs[0]]?];
            sign = true;
        }

        let Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } = &slice.primitive
        else {
            return None;
        };

        let p = slice.inputs[0];
        let shape = &graph.type_of(p).shape;
        let last = shape.len().checked_sub(1)?;
        let half = shape[last] / 2;

        let whole = (0..last).all(|d| start_indices[d] == 0 && limit_indices[d] == shape[d]);
        let fits = dimension == last && whole && half.is_power_of_two() && shape[last] == 2 * half;
        let from = start_indices[last];

        (fits && limit_indices[last] == from + half).then_some((p, from / half.max(1), half, sign))
    };

    let (p, high, half, negated) = part(a, true)?;
    let (q, low, _, _) = part(b, false)?;

    (p == q && high == 1 && low == 0).then_some((p, half, negated))
}

/// `graph` with each pair swap written as slices and a concatenation replaced by `SwapPairs` (the slices pruned when
/// nothing else reads them), or `None` when it has none.
pub(crate) fn swapped(graph: &Graph) -> Option<Graph> {
    let mut producer = vec![None; graph.types.len()];

    for (i, node) in graph.nodes().iter().enumerate() {
        producer[node.output] = Some(i);
    }

    let swaps: Vec<Option<(Var, usize, bool)>> = graph
        .nodes()
        .iter()
        .map(|node| swap_of(graph, &producer, node))
        .collect();

    if swaps.iter().all(Option::is_none) {
        return None;
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    for (node, swap) in graph.nodes().iter().zip(&swaps) {
        out.set_origin(node);

        let Some(&(p, distance, negated)) = swap.as_ref() else {
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("a node of the graph");

            continue;
        };

        let swapped = out
            .apply(Primitive::SwapPairs { distance }, &[map[p]])
            .expect("a swap keeps its operand's type");

        map[node.output] = match negated {
            false => swapped,
            true => signed(&mut out, swapped, distance),
        };
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("the graph's outputs");
    Some(out.prune().0)
}

/// `x` (a swap) times `-1` on the first `distance` elements of its last axis and `1` on the rest: the negated part of
/// `concatenate([-hi, lo])`, an exact sign flip.
fn signed(out: &mut Graph, x: Var, distance: usize) -> Var {
    let ty = out.type_of(x).clone();
    let last = ty.shape.len() - 1;
    let mut apply = |p: Primitive, args: &[Var]| out.apply(p, args).expect("the sign's nodes");

    let full = |value: Scalar, dtype: DType| Primitive::Full {
        shape: ty.shape.clone(),
        fill_value: value,
        dtype,
    };

    let at = apply(
        Primitive::Iota {
            dtype: DType::I32,
            shape: ty.shape.clone(),
            dimension: last,
        },
        &[],
    );

    let bound = apply(full(Scalar::Int(distance as i64), DType::I32), &[]);
    let first = apply(Primitive::Lt, &[at, bound]);
    let minus = apply(full(Scalar::Float(-1.0), ty.dtype), &[]);
    let plus = apply(full(Scalar::Float(1.0), ty.dtype), &[]);
    let sign = apply(Primitive::Select, &[first, minus, plus]);
    apply(Primitive::Mul, &[x, sign])
}
