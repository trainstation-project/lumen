//! Algebraic simplification: identities, composed layout ops, transposes folded into dots, and casts
//! and layout ops moved to where they are cheaper.

use std::collections::HashMap;

use super::cse::cse;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};
use crate::tensor::dtype::dispatch_dtype;
use crate::{DType, Element, Scalar};

/// Simplifies `graph` node by node, then prunes and deduplicates it. `matched` says pattern-matching
/// passes have run, enabling the rewrites that restructure (cast-into-dot, moved casts and layouts,
/// transposed dots); otherwise only transposes of graph inputs fold into dots.
pub(crate) fn simplify_with(graph: &Graph, matched: bool) -> Graph {
    let nodes = graph.nodes();
    // Distinct readers of each value, counting graph outputs.
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
    // Producer index (in `out`) of each rewritten value, and whether it is a fresh value read once.
    let mut producer: HashMap<Var, usize> = HashMap::new();
    let mut once: HashMap<Var, bool> = HashMap::new();
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in nodes {
        out.set_origin(node);
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let first = out.nodes().len();
        let value = match rewrite(&out, &producer, &once, matched, &node.primitive, &inputs) {
            Some(Rewrite::Value(v)) => v,
            Some(Rewrite::Node(p, ins)) => {
                let replaced: Vec<Node> = std::iter::once(node.clone())
                    .chain(
                        inputs
                            .iter()
                            .filter(|v| !ins.contains(v))
                            .filter_map(|v| producer.get(v).map(|&k| out.nodes()[k].clone())),
                    )
                    .collect();
                out.set_origins(&replaced.iter().collect::<Vec<_>>());
                out.apply(p, &ins).expect("a rewrite is typed as the node")
            }
            None => match &node.primitive {
                Primitive::DotGeneral { .. } => {
                    folded(&mut out, &producer, matched, &node.primitive, &inputs)
                }
                _ if matched => moved(&mut out, &producer, &once, &node.primitive, &inputs),
                _ => None,
            }
            .unwrap_or_else(|| {
                let v = out
                    .apply(node.primitive.clone(), &inputs)
                    .expect("a node of the graph");
                out.set_label(v, node.label);
                v
            }),
        };
        // Intermediate nodes a rewrite created have a single reader by construction.
        for (k, n) in out.nodes().iter().enumerate().skip(first) {
            producer.insert(n.output, k);
            if n.output != value {
                once.insert(n.output, true);
            }
        }
        let fresh = !once.contains_key(&value);
        once.insert(value, fresh && reads[node.output] == 1);
        map[node.output] = value;
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    cse(&out.prune().0)
}

/// The result of a local rewrite.
enum Rewrite {
    /// The node is replaced by an existing value.
    Value(Var),
    /// The node is replaced by this primitive on these operands.
    Node(Primitive, Vec<Var>),
}

/// Local rewrite of `primitive` on `inputs`: identities (`x * 1`, `x + 0`, double negation, no-op casts,
/// reshapes and transposes) and composition of consecutive layout ops.
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
        Cast { new_dtype } => match node(inputs[0]) {
            Some(Node {
                primitive:
                    DotGeneral {
                        lhs_contracting,
                        rhs_contracting,
                        lhs_batch,
                        rhs_batch,
                        accum_dtype,
                        output_dtype,
                    },
                inputs: x,
                output,
                ..
            // A narrowing cast of a dot's sole use becomes the dot's output dtype.
            }) if matched
                && accum_dtype == output_dtype
                && accum_dtype.is_float()
                && new_dtype.size_of() < output_dtype.size_of()
                && out.type_of(x[0]).dtype == *new_dtype
                && once.get(output) == Some(&true) =>
            {
                let dot = DotGeneral {
                    lhs_contracting: lhs_contracting.clone(),
                    rhs_contracting: rhs_contracting.clone(),
                    lhs_batch: lhs_batch.clone(),
                    rhs_batch: rhs_batch.clone(),
                    accum_dtype: *accum_dtype,
                    output_dtype: *new_dtype,
                };
                Some(Rewrite::Node(dot, x.clone()))
            }
            _ => None,
        },
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
                // Transposing a dot's result to (batch, rhs free, lhs free) is the dot with swapped operands.
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

/// Node in `out` producing `v`.
fn node_of<'a>(out: &'a Graph, producer: &HashMap<Var, usize>, v: Var) -> Option<&'a Node> {
    producer.get(&v).map(|&i| &out.nodes()[i])
}

/// Moves ops so they run on the cheaper side: an elementwise op of reshaped/transposed operands runs
/// before the rearrangement; a widening cast moves after layout ops; a narrowing cast moves before them.
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
    let mut operands: Vec<(Option<Primitive>, Vec<Var>)> = Vec::new();
    let then: Primitive = match primitive {
        p if crate::graph_compiler::elementwise(p) && !matches!(p, Cast { .. }) => {
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

/// `v` cast to `dtype`, pushing the cast through single-use layout ops to their operands.
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

/// The constant operand `n` restated in the shape before `rearrange`, or `None` when it isn't a constant.
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

/// `value` rounded through `dtype`.
fn round(dtype: DType, value: Scalar) -> Scalar {
    dispatch_dtype!(dtype, T => T::from_scalar(value).to_scalar())
}

/// `transpose(dot(x0, x1))` as `dot(x1, x0)` when the permutation swaps the two operands' free dims.
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

/// The dot with operand transposes (possibly under a cast) folded into its dimension numbers, when the
/// free dims keep their order; `None` when nothing folds.
fn folded(
    out: &mut Graph,
    producer: &HashMap<Var, usize>,
    matched: bool,
    dot: &Primitive,
    x: &[Var],
) -> Option<Var> {
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
    let (mut dims, mut operands) = (
        [
            (lhs_contracting.clone(), lhs_batch.clone()),
            (rhs_contracting.clone(), rhs_batch.clone()),
        ],
        x.to_vec(),
    );
    let mut changed = false;
    for side in 0..2 {
        let (cast, transposed) = match node_of(out, producer, x[side]) {
            Some(Node {
                primitive: cast @ Primitive::Cast { .. },
                inputs,
                ..
            }) => (Some(cast.clone()), inputs[0]),
            _ => (None, x[side]),
        };
        let Some(Node {
            primitive: Primitive::Transpose { permutation },
            inputs,
            ..
        }) = node_of(out, producer, transposed)
        else {
            continue;
        };
        let (permutation, input) = (permutation.clone(), inputs[0]);
        if !matched && !out.inputs().contains(&input) {
            continue;
        }
        let (contracting, batch) = &dims[side];
        let rank = permutation.len();
        let free: Vec<usize> = (0..rank)
            .filter(|d| !contracting.contains(d) && !batch.contains(d))
            .map(|d| permutation[d])
            .collect();
        if !free.is_sorted() {
            continue;
        }
        let map = |ds: &[usize]| ds.iter().map(|&d| permutation[d]).collect::<Vec<_>>();
        dims[side] = (map(contracting), map(batch));
        operands[side] = match cast {
            Some(cast) => out.apply(cast, &[input]).expect("a cast of the input"),
            None => input,
        };
        changed = true;
    }
    let [(lhs_contracting, lhs_batch), (rhs_contracting, rhs_batch)] = dims;
    changed.then(|| {
        let dot = Primitive::DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            rhs_batch,
            accum_dtype: *accum_dtype,
            output_dtype: *output_dtype,
        };
        out.apply(dot, &operands).expect("the dot, folded")
    })
}

/// The constant `v` holds when it is a broadcast, reshape or cast of a `full`.
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

/// Compiler pass running [`simplify_with`].
pub(crate) struct Simplify {
    /// Pattern-matching passes have run, so structural rewrites are allowed.
    pub(crate) matched: bool,
}

impl<M: Module> CompilerPass<M> for Simplify {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "simplify"
    }

    /// Replaces the module's graph with its simplified version.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = simplify_with(m.graph(), self.matched);
        Ok(())
    }
}
