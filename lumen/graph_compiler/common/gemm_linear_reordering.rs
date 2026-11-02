//! Pass moving a factor of a matmul's operand to its output (CODA's reordering, `f(x) W = f(x W)` for a
//! linear `f` acting on the axes the matmul doesn't contract): a factor constant along the contracted
//! axis, per lhs row (a norm's `1 / rms`, a softmax's `1 / sum`, an activation's per-token scale) or per
//! rhs column (a weight's per-channel dequantization scale), scales the matmul's output rows or columns
//! instead, in its epilogue, so the scaled operand is never needed whole. The rounding moves with it: a
//! cast between the factor and the matmul then rounds the unscaled value, so results differ from the
//! traced program's in the last bits; `gemm_linear_reordering` (off by default) turns it on.

use crate::DType;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};

/// How an operand is rebuilt without its factor: the original value, or an op of the rebuilt value and an
/// original operand.
enum Expr {
    /// An original value, unchanged.
    Old(Var),
    /// The rebuilt value times an original operand.
    Mul(Box<Expr>, Var),
    /// The rebuilt value divided by an original operand.
    Div(Box<Expr>, Var),
    /// The rebuilt value cast to a float dtype.
    Cast(Box<Expr>, DType),
    /// The rebuilt value negated.
    Neg(Box<Expr>),
}

/// A factor: the value holding it (its operand's kept axes, or the operand's rank with 1s on the others),
/// and whether the operand was divided by it.
type Factor = (Var, bool);

/// An operand rebuilt without its factor, and the factor.
type Stripped = (Expr, Factor);

/// The value each node index produces, and the nodes each value is read by.
struct Uses {
    producer: Vec<Option<usize>>,
    readers: Vec<Vec<usize>>,
}

/// Whether `node` is a matmul this pass can reorder: an unbatched float matmul of a rank-2+ lhs contracted
/// on its last axis with a 2-D rhs contracted on one axis.
fn plain(graph: &Graph, node: &Node) -> bool {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        accum_dtype,
        ..
    } = &node.primitive
    else {
        return false;
    };

    let (l, r) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));

    node.inputs[0] != node.inputs[1]
        && lhs_batch.is_empty()
        && l.shape.len() >= 2
        && *lhs_contracting == [l.shape.len() - 1]
        && r.shape.len() == 2
        && rhs_contracting.len() == 1
        && accum_dtype.is_float()
}

/// The axes of `node`'s operand `side` (0 the lhs, 1 the rhs) the matmul keeps: the lhs's rows, or the
/// rhs's columns.
fn kept(graph: &Graph, node: &Node, side: usize) -> Vec<usize> {
    let Primitive::DotGeneral {
        rhs_contracting, ..
    } = &node.primitive
    else {
        unreachable!("a matmul")
    };

    match side {
        0 => (0..graph.type_of(node.inputs[0]).shape.len() - 1).collect(),
        _ => vec![1 - rhs_contracting[0]],
    }
}

/// The value `b` broadcasts along every axis of `like` but `kept`, if it does: of the kept axes, or of
/// `like`'s rank with 1s on the others.
fn factor(graph: &Graph, uses: &Uses, b: Var, like: Var, kept: &[usize]) -> Option<Var> {
    let node = &graph.nodes()[uses.producer[b]?];

    let Primitive::BroadcastInDim {
        shape,
        broadcast_dimensions,
    } = &node.primitive
    else {
        return None;
    };

    let full = &graph.type_of(like).shape;
    let source = graph.type_of(node.inputs[0]);
    let sizes: Vec<usize> = kept.iter().map(|&a| full[a]).collect();
    let every: Vec<usize> = (0..full.len()).collect();

    let ones = (0..full.len())
        .map(|a| if kept.contains(&a) { full[a] } else { 1 })
        .collect::<Vec<_>>();

    let along = (source.shape == sizes && *broadcast_dimensions == kept)
        || (source.shape == ones && *broadcast_dimensions == every);

    (shape == full && source.dtype.is_float() && along).then_some(node.inputs[0])
}

/// `v` without one factor constant along all but its `kept` axes, found through multiplications,
/// divisions by it, float casts and negations: the rebuilt value and the factor.
fn strip(graph: &Graph, uses: &Uses, v: Var, kept: &[usize]) -> Option<Stripped> {
    let node = &graph.nodes()[uses.producer[v]?];
    let float = |u: Var| graph.type_of(u).dtype.is_float();

    match node.primitive {
        Primitive::Mul => {
            let [a, b] = node.inputs[..] else { return None };

            if let Some(f) = factor(graph, uses, b, v, kept) {
                return Some((Expr::Old(a), (f, false)));
            }

            if let Some(f) = factor(graph, uses, a, v, kept) {
                return Some((Expr::Old(b), (f, false)));
            }

            if let Some((e, f)) = strip(graph, uses, a, kept) {
                return Some((Expr::Mul(Box::new(e), b), f));
            }

            let (e, f) = strip(graph, uses, b, kept)?;
            Some((Expr::Mul(Box::new(e), a), f))
        }
        Primitive::Div => {
            let [a, b] = node.inputs[..] else { return None };

            if let Some(f) = factor(graph, uses, b, v, kept) {
                return Some((Expr::Old(a), (f, true)));
            }

            let (e, f) = strip(graph, uses, a, kept)?;
            Some((Expr::Div(Box::new(e), b), f))
        }
        Primitive::Cast { new_dtype } if float(node.inputs[0]) && new_dtype.is_float() => {
            let (e, f) = strip(graph, uses, node.inputs[0], kept)?;
            Some((Expr::Cast(Box::new(e), new_dtype), f))
        }
        Primitive::Neg => {
            let (e, f) = strip(graph, uses, node.inputs[0], kept)?;
            Some((Expr::Neg(Box::new(e)), f))
        }
        _ => None,
    }
}

/// Emits `e` into `out`, original values through `map`.
fn emit(out: &mut Graph, map: &[Var], e: &Expr) -> Var {
    let apply = |out: &mut Graph, p: Primitive, ins: &[Var]| out.apply(p, ins).expect("well typed");

    match e {
        Expr::Old(v) => map[*v],
        Expr::Mul(e, b) => {
            let a = emit(out, map, e);
            apply(out, Primitive::Mul, &[a, map[*b]])
        }
        Expr::Div(e, b) => {
            let a = emit(out, map, e);
            apply(out, Primitive::Div, &[a, map[*b]])
        }
        Expr::Cast(e, dtype) => {
            let a = emit(out, map, e);
            apply(out, Primitive::Cast { new_dtype: *dtype }, &[a])
        }
        Expr::Neg(e) => {
            let a = emit(out, map, e);
            apply(out, Primitive::Neg, &[a])
        }
    }
}

/// The operand `side` of matmul node `k` without its factor, when every reader of the operand is a matmul
/// reading it on that side with the same kept axes (so the scaled operand is no longer computed).
fn stripped(graph: &Graph, uses: &Uses, k: usize, side: usize) -> Option<Stripped> {
    let nodes = graph.nodes();
    let (node, v) = (&nodes[k], nodes[k].inputs[side]);
    let axes = kept(graph, node, side);

    let alike = |&r: &usize| {
        let reader = &nodes[r];
        plain(graph, reader) && reader.inputs[side] == v && kept(graph, reader, side) == axes
    };

    let only_matmuls = !graph.outputs().contains(&v) && uses.readers[v].iter().all(alike);

    only_matmuls.then(|| strip(graph, uses, v, &axes)).flatten()
}

/// `factor` (of operand `side`'s kept axes, or its rank with 1s) broadcast over a matmul output of
/// `shape`, in `dtype`: a row factor along the output's leading axes, a column factor along its last.
fn spread(
    out: &mut Graph,
    graph: &Graph,
    map: &[Var],
    (factor, side): (Var, usize),
    shape: &[usize],
    dtype: DType,
) -> Var {
    let apply = |out: &mut Graph, p: Primitive, ins: &[Var]| out.apply(p, ins).expect("well typed");
    let source = graph.type_of(factor);
    let rows = shape.len() - 1;
    let mut f = map[factor];

    if source.dtype != dtype {
        f = apply(out, Primitive::Cast { new_dtype: dtype }, &[f]);
    }

    // A row factor kept `[rows..., 1]` (a keepdim sum) broadcasts as it is, as a program writing `x W / s`
    // would; a column factor kept with the rhs's rank becomes its column vector first.
    let dims: Vec<usize> = match (side, source.shape.len()) {
        (0, n) if n == rows => (0..rows).collect(),
        (0, _) => (0..=rows).collect(),
        (_, 1) => vec![rows],
        _ => {
            let new_sizes = vec![shape[rows]];
            f = apply(out, Primitive::Reshape { new_sizes }, &[f]);
            vec![rows]
        }
    };

    let broadcast = Primitive::BroadcastInDim {
        shape: shape.to_vec(),
        broadcast_dimensions: dims,
    };

    apply(out, broadcast, &[f])
}

/// Rewrites each matmul whose operand, read only by such matmuls, carries a factor constant along the
/// contracted axis: the matmul of the operands without their factors, in its accumulation dtype, times
/// (or divided by) each factor per output row or column, rounded once to its output dtype.
pub(crate) fn gemm_linear_reordering(graph: &Graph) -> Graph {
    let nodes = graph.nodes();

    let mut uses = Uses {
        producer: vec![None; graph.types.len()],
        readers: vec![Vec::new(); graph.types.len()],
    };

    for (i, node) in nodes.iter().enumerate() {
        uses.producer[node.output] = Some(i);

        for &v in &node.inputs {
            uses.readers[v].push(i);
        }
    }

    // Each reorderable matmul's operands without their factors, by node index.
    let reordered: Vec<[Option<Stripped>; 2]> = (0..nodes.len())
        .map(|k| match plain(graph, &nodes[k]) {
            true => [0, 1].map(|side| stripped(graph, &uses, k, side)),
            false => [None, None],
        })
        .collect();

    if reordered.iter().flatten().all(Option::is_none) {
        return graph.clone();
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    for (k, node) in nodes.iter().enumerate() {
        out.set_origin(node);

        if reordered[k].iter().all(Option::is_none) {
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("well typed");

            continue;
        }

        let Primitive::DotGeneral {
            accum_dtype,
            output_dtype,
            ..
        } = &node.primitive
        else {
            unreachable!("a matmul")
        };

        let mut dot = node.primitive.clone();

        if let Primitive::DotGeneral { output_dtype, .. } = &mut dot {
            *output_dtype = *accum_dtype;
        }

        let operands = [0, 1].map(|side| match &reordered[k][side] {
            Some((e, _)) => emit(&mut out, &map, e),
            None => map[node.inputs[side]],
        });

        let shape = graph.type_of(node.output).shape.clone();
        let mut v = out.apply(dot, &operands).expect("well typed");

        for (side, s) in reordered[k].iter().enumerate() {
            let Some((_, (f, divide))) = s else { continue };
            let f = spread(&mut out, graph, &map, (*f, side), &shape, *accum_dtype);
            let op = if *divide {
                Primitive::Div
            } else {
                Primitive::Mul
            };
            v = out.apply(op, &[v, f]).expect("well typed");
        }

        if output_dtype != accum_dtype {
            let cast = Primitive::Cast {
                new_dtype: *output_dtype,
            };

            v = out.apply(cast, &[v]).expect("well typed");
        }

        map[node.output] = v;
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out.prune().0
}

/// Compiler pass running [`gemm_linear_reordering`].
pub(crate) struct GemmLinearReordering;

impl<M: Module> CompilerPass<M> for GemmLinearReordering {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "gemm_linear_reordering"
    }

    /// Replaces the module's graph with its reordered version.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = gemm_linear_reordering(m.graph());
        Ok(())
    }
}
