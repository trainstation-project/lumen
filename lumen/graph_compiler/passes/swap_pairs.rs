//! Pass rewriting a swap of adjacent pairs written as slices and a concatenation into one `SwapPairs`:
//! `concatenate([slice(p, [.., 1:2]), slice(p, [.., 0:1])], last)` of a value `p` whose last axis has length 2 (an
//! interleaved rotary embedding's `cat([x[..., 1:], x[..., :1]])` over its pairs) becomes `swap_pairs(p)`. A kernel
//! then reads each element's partner at index `j ^ 1`, or, when the value is in registers (a GEMM accumulator),
//! swaps the two elements a thread holds; slices and a concatenation would need the value in memory.
//!
//! Devices: it runs on each device whose codegen implements `codegen::SwapsPairs` (MPS and Triton), right after the
//! full `simplify`. Other devices keep the slices and concatenation.

use crate::codegen::SwapsPairs;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::{CompilerPass, DevicePass, Module};
use crate::graph_compiler::pipeline::Pass;

/// Compiler pass turning each pair swap written with slices and a concatenation into `SwapPairs`.
pub(crate) struct SwapPairs;

/// Always, on every device whose codegen implements `SwapsPairs`.
impl<D: SwapsPairs> DevicePass<D> for SwapPairs {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, D>> {
        Box::new(SwapPairs)
    }
}

impl<M: Module> CompilerPass<M> for SwapPairs {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "swap_pairs"
    }

    /// Rewrites every pair swap of the module's graph.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        if let Some(graph) = swapped(m.graph()) {
            *m.graph_mut() = graph;
        }

        Ok(())
    }
}

/// The value whose pairs `node` swaps, when it is `concatenate([slice(p, [.., 1:2]), slice(p, [.., 0:1])])` along
/// the last axis of a `p` whose last axis has length 2, each slice whole on the other axes.
fn swap_of(graph: &Graph, producer: &[Option<usize>], node: &Node) -> Option<Var> {
    let Primitive::Concatenate { dimension, .. } = node.primitive else {
        return None;
    };

    let [a, b] = node.inputs[..] else {
        return None;
    };

    let half = |v: Var, from: usize| {
        let slice = &graph.nodes()[producer[v]?];

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

        let whole = (0..last).all(|d| start_indices[d] == 0 && limit_indices[d] == shape[d]);
        let one = start_indices[last] == from && limit_indices[last] == from + 1;

        (shape[last] == 2 && dimension == last && whole && one).then_some(p)
    };

    let p = half(a, 1)?;
    (half(b, 0)? == p).then_some(p)
}

/// `graph` with each pair swap written as slices and a concatenation replaced by `SwapPairs` (the slices pruned when
/// nothing else reads them), or `None` when it has none.
pub(crate) fn swapped(graph: &Graph) -> Option<Graph> {
    let mut producer = vec![None; graph.types.len()];

    for (i, node) in graph.nodes().iter().enumerate() {
        producer[node.output] = Some(i);
    }

    let swaps: Vec<Option<Var>> = graph
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

        let (primitive, inputs) = match swap {
            Some(p) => (Primitive::SwapPairs, vec![map[*p]]),
            None => (
                node.primitive.clone(),
                node.inputs.iter().map(|&v| map[v]).collect(),
            ),
        };

        map[node.output] = out
            .apply(primitive, &inputs)
            .expect("a rewrite keeps every value's type");
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("the graph's outputs");
    Some(out.prune().0)
}
