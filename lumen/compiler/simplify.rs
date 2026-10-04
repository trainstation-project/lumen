//! Algebraic simplification (XLA: `AlgebraicSimplifier`):
//!
//! - `x * 1`, `1 * x`, `x / 1`, `x + 0`, `0 + x`, `x - 0`, `neg(neg(x))`,
//!   `select(p, x, x)`, a cast to `x`'s own dtype: `x` (each value as it
//!   was, but `-0 + 0`: `-0`, not `+0`, as XLA's);
//! - a reshape of a reshape, a transpose of a transpose, a broadcast of a
//!   broadcast: one; a reshape to `x`'s shape, an identity transpose: `x`;
//! - with `swap_dots` (the MPS compiler's, once attention is matched: its
//!   backward's transposes of dots are its kernels' layouts), a transpose
//!   of a dot swapping its operands' free dimensions (as autodiff's
//!   transpose of a dot gives a weight's gradient): the dot of the operands
//!   swapped, so no copy.
//!
//! Dead nodes are dropped, and repeated ones merged again (CSE).

use std::collections::HashMap;

use super::cse::cse;
use crate::graph::{Graph, Node, Primitive, Var};

/// `graph` simplified.
pub(crate) fn simplify(graph: &Graph) -> Graph {
    simplify_with(graph, false)
}

/// `graph` simplified, transposes of dots swapping their operands too if
/// `swap_dots`.
pub(crate) fn simplify_with(graph: &Graph, swap_dots: bool) -> Graph {
    let nodes = graph.nodes();
    // The values read once, by a node (not an output): a dot's that a
    // transpose alone reads is rewritten, never computed twice.
    let mut reads = vec![0usize; graph.types.len()];
    for v in nodes.iter().flat_map(|n| &n.inputs).chain(graph.outputs()) {
        reads[*v] += 1;
    }
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    // Each value of `out`'s node (to look at a value's producer), and
    // whether it is read once (a dot's, by its transpose, may be swapped:
    // never computed twice).
    let mut producer: HashMap<Var, usize> = HashMap::new();
    let mut once: HashMap<Var, bool> = HashMap::new();
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in nodes {
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let value = match rewrite(&out, &producer, &once, swap_dots, &node.primitive, &inputs) {
            Some(Rewrite::Value(v)) => v,
            Some(Rewrite::Node(p, ins)) => {
                out.apply(p, &ins).expect("a rewrite is typed as the node")
            }
            None => out
                .apply(node.primitive.clone(), &inputs)
                .expect("a node of the graph"),
        };
        if out.nodes().last().is_some_and(|n| n.output == value) {
            producer.insert(value, out.nodes().len() - 1);
        }
        // A value a node simplified to has that node's readers too.
        let fresh = !once.contains_key(&value);
        once.insert(value, fresh && reads[node.output] == 1);
        map[node.output] = value;
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    cse(&out.prune().0)
}

enum Rewrite {
    /// The node's value is this one.
    Value(Var),
    /// The node is this one.
    Node(Primitive, Vec<Var>),
}

/// What `primitive` applied to `inputs` (values of `out`) simplifies to.
fn rewrite(
    out: &Graph,
    producer: &HashMap<Var, usize>,
    once: &HashMap<Var, bool>,
    swap_dots: bool,
    primitive: &Primitive,
    inputs: &[Var],
) -> Option<Rewrite> {
    use Primitive::*;
    let node = |v: Var| producer.get(&v).map(|&i| &out.nodes()[i]);
    let is = |v: Var, c: f64| constant(out, producer, v) == Some(c);
    match primitive {
        Mul if is(inputs[1], 1.0) => Some(Rewrite::Value(inputs[0])),
        Mul if is(inputs[0], 1.0) => Some(Rewrite::Value(inputs[1])),
        Div if is(inputs[1], 1.0) => Some(Rewrite::Value(inputs[0])),
        Sub if is(inputs[1], 0.0) => Some(Rewrite::Value(inputs[0])),
        Add if is(inputs[1], 0.0) => Some(Rewrite::Value(inputs[0])),
        Add if is(inputs[0], 0.0) => Some(Rewrite::Value(inputs[1])),
        Neg => match node(inputs[0]) {
            Some(Node {
                primitive: Neg,
                inputs: x,
                ..
            }) => Some(Rewrite::Value(x[0])),
            _ => None,
        },
        Select if inputs[1] == inputs[2] => Some(Rewrite::Value(inputs[1])),
        Cast { new_dtype } if out.type_of(inputs[0]).dtype == *new_dtype => {
            Some(Rewrite::Value(inputs[0]))
        }
        Reshape { new_sizes } => {
            if out.type_of(inputs[0]).shape == *new_sizes {
                return Some(Rewrite::Value(inputs[0]));
            }
            match node(inputs[0]) {
                Some(Node {
                    primitive: Reshape { .. },
                    inputs: x,
                    ..
                }) => Some(Rewrite::Node(primitive.clone(), vec![x[0]])),
                _ => None,
            }
        }
        Transpose { permutation } => {
            if permutation.iter().enumerate().all(|(i, &d)| i == d) {
                return Some(Rewrite::Value(inputs[0]));
            }
            match node(inputs[0]) {
                Some(Node {
                    primitive: Transpose { permutation: inner },
                    inputs: x,
                    ..
                }) => {
                    let permutation: Vec<usize> = permutation.iter().map(|&d| inner[d]).collect();
                    // Composing to the identity: x.
                    match permutation.iter().enumerate().all(|(i, &d)| i == d) {
                        true => Some(Rewrite::Value(x[0])),
                        false => Some(Rewrite::Node(Transpose { permutation }, vec![x[0]])),
                    }
                }
                Some(Node {
                    primitive: dot @ DotGeneral { .. },
                    inputs: x,
                    output,
                }) if swap_dots && once.get(output) == Some(&true) => {
                    swapped(out, dot, x, permutation)
                }
                _ => None,
            }
        }
        BroadcastInDim {
            shape,
            broadcast_dimensions: outer,
        } => match node(inputs[0]) {
            Some(Node {
                primitive:
                    BroadcastInDim {
                        broadcast_dimensions: inner,
                        ..
                    },
                inputs: x,
                ..
            }) => {
                let broadcast_dimensions: Vec<usize> = inner.iter().map(|&d| outer[d]).collect();
                let composed = BroadcastInDim {
                    shape: shape.clone(),
                    broadcast_dimensions,
                };
                // Only if it types (each kept dimension 1 or the size).
                composed
                    .infer(&[out.type_of(x[0])])
                    .is_ok()
                    .then(|| Rewrite::Node(composed, vec![x[0]]))
            }
            _ => None,
        },
        _ => None,
    }
}

/// The dot `dot(x[0], x[1])` transposed by `permutation`, if that swaps
/// its free dimensions (batch, lhs free, rhs free → batch, rhs free, lhs
/// free, each in order): the dot of the operands swapped.
fn swapped(out: &Graph, dot: &Primitive, x: &[Var], permutation: &[usize]) -> Option<Rewrite> {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
        accum_dtype,
        output_dtype,
    } = dot
    else {
        unreachable!("a dot")
    };
    let free = |v: Var, used: usize| out.type_of(v).shape.len() - used;
    let nb = lhs_batch.len();
    let fl = free(x[0], nb + lhs_contracting.len());
    let fr = free(x[1], nb + rhs_contracting.len());
    let expected: Vec<usize> = (0..nb)
        .chain(nb + fl..nb + fl + fr)
        .chain(nb..nb + fl)
        .collect();
    (permutation == expected.as_slice()).then(|| {
        let swapped = Primitive::DotGeneral {
            lhs_contracting: rhs_contracting.clone(),
            rhs_contracting: lhs_contracting.clone(),
            lhs_batch: rhs_batch.clone(),
            rhs_batch: lhs_batch.clone(),
            accum_dtype: *accum_dtype,
            output_dtype: *output_dtype,
        };
        Rewrite::Node(swapped, vec![x[1], x[0]])
    })
}

/// The value of `v` if it is a constant everywhere: a `full`, through
/// broadcasts, reshapes and casts.
fn constant(out: &Graph, producer: &HashMap<Var, usize>, v: Var) -> Option<f64> {
    let mut v = v;
    loop {
        let node = &out.nodes()[*producer.get(&v)?];
        match &node.primitive {
            Primitive::Full { fill_value, .. } => return Some(fill_value.to_f64()),
            Primitive::BroadcastInDim { .. }
            | Primitive::Reshape { .. }
            | Primitive::Cast { .. } => v = node.inputs[0],
            _ => return None,
        }
    }
}
