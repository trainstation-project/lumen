//! Algebraic simplification (XLA: `AlgebraicSimplifier`):
//!
//! - `x * 1`, `1 * x`, `x / 1`, `x + 0`, `0 + x`, `x - 0`, `neg(neg(x))`,
//!   `select(p, x, x)`, a cast to `x`'s own dtype: `x` (each value as it
//!   was, but `-0 + 0`: `-0`, not `+0`, as XLA's);
//! - a reshape of a reshape, a transpose of a transpose, a broadcast of a
//!   broadcast: one; a reshape to `x`'s shape, an identity transpose: `x`;
//! - with `matched` (once attention is matched: the MPS compiler's after its
//!   matchers, whose patterns these change, or a device with none):
//!   - a transpose of a dot swapping its operands' free dimensions (as
//!     autodiff's transpose of a dot gives a weight's gradient): the dot of
//!     the operands swapped, so no copy;
//!   - an elementwise primitive (not a cast) of the same reshape or
//!     transpose of values of one shape, or of constants: the reshape or
//!     transpose of it, on those values (XLA: `ReshapeMover`), so it joins
//!     the fusion before (a dot's epilogue);
//!   - a reshape, transpose, slice or concatenate of widening casts: the
//!     cast of it; a narrowing cast of one: it of narrowing casts (XLA:
//!     `ConvertMover`), so the layout primitive moves the narrow values.
//!
//! Each moved value is read by the primitive moved past alone. A dot's
//! batch dimensions are not merged (XLA: `DotDimensionMerger`): the MPS
//! matmul collapses them itself (`collapsed`, `ops/dot_general/mps`), and
//! a reshape after a dot would keep its epilogue out of its kernel.
//!
//! Dead nodes are dropped, and repeated ones merged again (CSE).

use std::collections::HashMap;

use super::cse::cse;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::tensor::dtype::dispatch_dtype;
use crate::{DType, Element, Scalar};

/// `graph` simplified.
pub(crate) fn simplify(graph: &Graph) -> Graph {
    simplify_with(graph, false)
}

/// `graph` simplified, with the rewrites changing what the attention
/// matcher matches too if `matched`.
pub(crate) fn simplify_with(graph: &Graph, matched: bool) -> Graph {
    let nodes = graph.nodes();
    // The values read once, by a node (not an output): a dot's that a
    // transpose alone reads is rewritten, never computed twice.
    let mut reads = vec![0usize; graph.types.len()];
    for n in nodes {
        for (k, &v) in n.inputs.iter().enumerate() {
            reads[v] += usize::from(!n.inputs[..k].contains(&v));
        }
    }
    for &v in graph.outputs() {
        reads[v] += 1;
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
        out.set_scope(node.scope);
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let first = out.nodes().len();
        let value = match rewrite(&out, &producer, &once, matched, &node.primitive, &inputs) {
            Some(Rewrite::Value(v)) => v,
            Some(Rewrite::Node(p, ins)) => {
                out.apply(p, &ins).expect("a rewrite is typed as the node")
            }
            None => match matched {
                true => moved(&mut out, &producer, &once, &node.primitive, &inputs),
                false => None,
            }
            .unwrap_or_else(|| {
                out.apply(node.primitive.clone(), &inputs)
                    .expect("a node of the graph")
            }),
        };
        // The nodes added: those before the last read by the next alone.
        for (k, n) in out.nodes().iter().enumerate().skip(first) {
            producer.insert(n.output, k);
            if n.output != value {
                once.insert(n.output, true);
            }
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
    matched: bool,
    primitive: &Primitive,
    inputs: &[Var],
) -> Option<Rewrite> {
    use Primitive::*;
    let node = |v: Var| node_of(out, producer, v);
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
                    ..
                }) if matched && once.get(output) == Some(&true) => {
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

/// The node of `out` computing `v`, if one does.
fn node_of<'a>(out: &'a Graph, producer: &HashMap<Var, usize>, v: Var) -> Option<&'a Node> {
    producer.get(&v).map(|&i| &out.nodes()[i])
}

/// `primitive` of `inputs`, a primitive moved past (the rewrites only
/// `matched` makes, see the module's), added to `out`: its value, if it is.
fn moved(
    out: &mut Graph,
    producer: &HashMap<Var, usize>,
    once: &HashMap<Var, bool>,
    primitive: &Primitive,
    inputs: &[Var],
) -> Option<Var> {
    use Primitive::*;
    let layout = |p: &Primitive| {
        matches!(
            p,
            Reshape { .. } | Transpose { .. } | Slice { .. } | Concatenate { .. }
        )
    };
    let read_once = |v: Var| once.get(&v) == Some(&true);
    // Each new operand: a value, or a node of these inputs.
    let mut operands: Vec<(Option<Primitive>, Vec<Var>)> = Vec::new();
    let then: Primitive = match primitive {
        // ReshapeMover: op(r(x), r(y), c) = r(op(x, y, c')), a reshape or
        // transpose `r` read by `op` alone, not of a constant.
        p if super::elementwise(p) && !matches!(p, Cast { .. }) => {
            let rearranged = |v: Var| {
                node_of(out, producer, v).filter(|n| {
                    matches!(n.primitive, Reshape { .. } | Transpose { .. })
                        && !matches!(
                            node_of(out, producer, n.inputs[0]).map(|m| &m.primitive),
                            Some(Full { .. })
                        )
                })
            };
            let r = inputs.iter().find_map(|&v| rearranged(v))?;
            let shape = &out.type_of(r.inputs[0]).shape;
            for &v in inputs {
                let n = node_of(out, producer, v)?;
                let same = n.primitive == r.primitive && out.type_of(n.inputs[0]).shape == *shape;
                operands.push(match same && read_once(v) {
                    true => (None, vec![n.inputs[0]]),
                    false => (
                        Some(unrearranged(n, &r.primitive, shape)?),
                        n.inputs.clone(),
                    ),
                });
            }
            r.primitive.clone()
        }
        // ConvertMover: l(widen(x), widen(y), c) = widen(l(x, y, c')), a
        // layout primitive `l` of casts from one dtype read by it alone, of
        // constants exact in it.
        p if layout(p) => {
            let cast = |v: Var| {
                node_of(out, producer, v)
                    .filter(|n| matches!(n.primitive, Cast { .. }) && read_once(v))
            };
            let c = inputs.iter().find_map(|&v| cast(v))?;
            let (from, to) = (out.type_of(c.inputs[0]).dtype, out.type_of(c.output).dtype);
            if from.size_of() >= to.size_of() {
                return None;
            }
            for &v in inputs {
                match (cast(v), &node_of(out, producer, v)?.primitive) {
                    (Some(n), _) if out.type_of(n.inputs[0]).dtype == from => {
                        operands.push((None, vec![n.inputs[0]]))
                    }
                    (
                        None,
                        Full {
                            shape, fill_value, ..
                        },
                    ) => {
                        let value = round(to, *fill_value);
                        let fill_value = round(from, value);
                        if round(to, fill_value) != value {
                            return None;
                        }
                        let full = Full {
                            shape: shape.clone(),
                            fill_value,
                            dtype: from,
                        };
                        operands.push((Some(full), Vec::new()));
                    }
                    _ => return None,
                }
            }
            Cast { new_dtype: to }
        }
        // ConvertMover: narrow(l(x)) = l(narrow(x)), up through the layout
        // primitives read by the next alone.
        Cast { new_dtype } => {
            let x = inputs[0];
            let n = node_of(out, producer, x).filter(|n| layout(&n.primitive) && read_once(x))?;
            if out.type_of(x).dtype.size_of() <= new_dtype.size_of() {
                return None;
            }
            let (p, ins) = (n.primitive.clone(), n.inputs.clone());
            let ins: Vec<Var> = ins
                .iter()
                .map(|&u| cast_up(out, producer, once, u, *new_dtype))
                .collect();
            return Some(
                out.apply(p, &ins)
                    .expect("a layout primitive of its operands cast"),
            );
        }
        _ => return None,
    };
    let operands: Vec<Var> = operands
        .into_iter()
        .map(|(p, ins)| match p {
            Some(p) => out.apply(p, &ins).expect("an operand moved"),
            None => ins[0],
        })
        .collect();
    let x = out
        .apply(primitive.clone(), &operands)
        .expect("the primitive on the operands moved");
    Some(out.apply(then, &[x]).expect("the primitive moved past"))
}

/// `v` cast to `dtype`, up through the layout primitives (reshapes,
/// transposes, slices and concatenates) read by the next alone.
fn cast_up(
    out: &mut Graph,
    producer: &HashMap<Var, usize>,
    once: &HashMap<Var, bool>,
    v: Var,
    dtype: DType,
) -> Var {
    use Primitive::*;
    let layout = node_of(out, producer, v)
        .filter(|n| {
            once.get(&v) == Some(&true)
                && matches!(
                    n.primitive,
                    Reshape { .. } | Transpose { .. } | Slice { .. } | Concatenate { .. }
                )
        })
        .map(|n| (n.primitive.clone(), n.inputs.clone()));
    match layout {
        Some((p, ins)) => {
            let ins: Vec<Var> = ins
                .iter()
                .map(|&u| cast_up(out, producer, once, u, dtype))
                .collect();
            out.apply(p, &ins)
                .expect("a layout primitive of its operands cast")
        }
        None => out.apply(Cast { new_dtype: dtype }, &[v]).expect("a cast"),
    }
}

/// The node `n` (of a constant) computes, of the shape `shape` that
/// `rearrange` (a reshape or transpose) takes to its own, if it computes
/// that as simply (XLA: `ReshapeMover::CanTriviallyRearrange`): a `full`,
/// a scalar's broadcast, a broadcast a transpose keeps in order.
fn unrearranged(n: &Node, rearrange: &Primitive, shape: &[usize]) -> Option<Primitive> {
    use Primitive::*;
    let shape = shape.to_vec();
    match (&n.primitive, rearrange) {
        (
            Full {
                fill_value, dtype, ..
            },
            _,
        ) => Some(Full {
            shape,
            fill_value: *fill_value,
            dtype: *dtype,
        }),
        (
            BroadcastInDim {
                broadcast_dimensions,
                ..
            },
            _,
        ) if broadcast_dimensions.is_empty() => Some(BroadcastInDim {
            shape,
            broadcast_dimensions: Vec::new(),
        }),
        (
            BroadcastInDim {
                broadcast_dimensions,
                ..
            },
            Transpose { permutation },
        ) => {
            let dims: Vec<usize> = broadcast_dimensions
                .iter()
                .map(|&d| permutation[d])
                .collect();
            (broadcast_dimensions.is_sorted() && dims.is_sorted()).then_some(BroadcastInDim {
                shape,
                broadcast_dimensions: dims,
            })
        }
        _ => None,
    }
}

/// `value` as an element of `dtype`.
fn round(dtype: DType, value: Scalar) -> Scalar {
    dispatch_dtype!(dtype, T => T::from_scalar(value).to_scalar())
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
