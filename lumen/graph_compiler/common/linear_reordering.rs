//! Pass moving a factor of a matmul's operand to its output (CODA's reordering, `f(x) W = f(x W)` for a
//! linear `f` acting on the axes the matmul doesn't contract): a factor constant along the contracted
//! axis, per lhs row (a norm's `1 / rms`, a softmax's `1 / sum`, an activation's per-token scale) or per
//! rhs column (a weight's per-channel dequantization scale), scales the matmul's output rows or columns
//! instead, in its epilogue, so the scaled operand is never needed whole. The rounding moves with it: a
//! cast between the factor and the matmul then rounds the unscaled value, so results differ from the
//! traced program's in the last bits; `linear_reordering` (off by default) turns it on.

use crate::DType;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};

/// How an operand is rebuilt without its factor: the original value, or an op of the rebuilt value and an
/// original operand; each op carries the index of the node it rebuilds, whose scope and source lines it
/// keeps (a norm's rows stay in the norm's profiler range).
enum Expr {
    /// An original value, unchanged.
    Old(Var),
    /// The rebuilt value times an original operand.
    Mul(Box<Expr>, Var, usize),
    /// The rebuilt value divided by an original operand.
    Div(Box<Expr>, Var, usize),
    /// The rebuilt value cast to a float dtype.
    Cast(Box<Expr>, DType, usize),
    /// The rebuilt value negated.
    Neg(Box<Expr>, usize),
}

/// A factor: the value holding it (its operand's kept axes, or the operand's rank with 1s on the others),
/// whether the operand was divided by it, and the index of the node that applied it (whose scope and
/// source lines its move past the matmul keeps).
type Factor = (Var, bool, usize);

/// An operand rebuilt without its factor, and the factor.
type Stripped = (Expr, Factor);

/// The value each node index produces, and the nodes each value is read by.
struct Uses {
    producer: Vec<Option<usize>>,
    readers: Vec<Vec<usize>>,
}

/// Whether `node` is a matmul this pass can reorder: an unbatched float matmul of a rank-2+ lhs contracted
/// on its last axis (or a 2-D lhs on either, `x` or `x.T`) with a 2-D rhs contracted on one axis (`W` or
/// `W.T`).
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

    let lhs = match l.shape.len() {
        2 => lhs_contracting.len() == 1,
        n => n > 2 && *lhs_contracting == [n - 1],
    };

    node.inputs[0] != node.inputs[1]
        && lhs_batch.is_empty()
        && lhs
        && r.shape.len() == 2
        && rhs_contracting.len() == 1
        && accum_dtype.is_float()
}

/// The axes of `node`'s operand `side` (0 the lhs, 1 the rhs) the matmul keeps: the lhs's uncontracted
/// axes (the output's rows), or the rhs's uncontracted one (its columns).
fn kept(graph: &Graph, node: &Node, side: usize) -> Vec<usize> {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        ..
    } = &node.primitive
    else {
        unreachable!("a matmul")
    };

    match side {
        0 => (0..graph.type_of(node.inputs[0]).shape.len())
            .filter(|a| !lhs_contracting.contains(a))
            .collect(),
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
    let k = uses.producer[v]?;
    let node = &graph.nodes()[k];
    let float = |u: Var| graph.type_of(u).dtype.is_float();

    match node.primitive {
        Primitive::Mul => {
            let [a, b] = node.inputs[..] else { return None };

            if let Some(f) = factor(graph, uses, b, v, kept) {
                return Some((Expr::Old(a), (f, false, k)));
            }

            if let Some(f) = factor(graph, uses, a, v, kept) {
                return Some((Expr::Old(b), (f, false, k)));
            }

            if let Some((e, f)) = strip(graph, uses, a, kept) {
                return Some((Expr::Mul(Box::new(e), b, k), f));
            }

            let (e, f) = strip(graph, uses, b, kept)?;
            Some((Expr::Mul(Box::new(e), a, k), f))
        }
        Primitive::Div => {
            let [a, b] = node.inputs[..] else { return None };

            if let Some(f) = factor(graph, uses, b, v, kept) {
                return Some((Expr::Old(a), (f, true, k)));
            }

            let (e, f) = strip(graph, uses, a, kept)?;
            Some((Expr::Div(Box::new(e), b, k), f))
        }
        Primitive::Cast { new_dtype } if float(node.inputs[0]) && new_dtype.is_float() => {
            let (e, f) = strip(graph, uses, node.inputs[0], kept)?;
            Some((Expr::Cast(Box::new(e), new_dtype, k), f))
        }
        Primitive::Neg => {
            let (e, f) = strip(graph, uses, node.inputs[0], kept)?;
            Some((Expr::Neg(Box::new(e), k), f))
        }
        _ => None,
    }
}

/// Emits `e` into `out`, original values through `map`, each op with its original node's origin in
/// `graph`.
fn emit(out: &mut Graph, graph: &Graph, map: &[Var], e: &Expr) -> Var {
    let apply = |out: &mut Graph, k: usize, p: Primitive, ins: &[Var]| {
        out.set_origin(&graph.nodes()[k]);
        out.apply(p, ins).expect("well typed")
    };

    match e {
        Expr::Old(v) => map[*v],
        Expr::Mul(e, b, k) => {
            let a = emit(out, graph, map, e);
            apply(out, *k, Primitive::Mul, &[a, map[*b]])
        }
        Expr::Div(e, b, k) => {
            let a = emit(out, graph, map, e);
            apply(out, *k, Primitive::Div, &[a, map[*b]])
        }
        Expr::Cast(e, dtype, k) => {
            let a = emit(out, graph, map, e);
            apply(out, *k, Primitive::Cast { new_dtype: *dtype }, &[a])
        }
        Expr::Neg(e, k) => {
            let a = emit(out, graph, map, e);
            apply(out, *k, Primitive::Neg, &[a])
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

/// `factor` (of operand `side`'s `kept` axes, or the operand's rank with 1s) broadcast over a matmul output
/// of `shape`, in `dtype`: a row factor along the output's leading axes, a column factor along its last.
fn spread(
    out: &mut Graph,
    graph: &Graph,
    map: &[Var],
    (factor, side, kept): (Var, usize, &[usize]),
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

    let at: Vec<usize> = match side {
        0 => (0..rows).collect(),
        _ => vec![rows],
    };

    // A row factor kept `[rows..., 1]` (a keepdim sum of `x`'s rows) broadcasts as it is, as a program
    // writing `x W / s` would; one kept with the operand's rank otherwise (`[1, m]` of `x.T`, `[1, n]` or
    // `[n, 1]` of a weight) becomes the vector of its kept axes first.
    let leading = side == 0 && kept.iter().copied().eq(0..rows);

    let dims = match source.shape.len() == kept.len() {
        true => at,
        false if leading => (0..=rows).collect(),
        false => {
            let new_sizes = at.iter().map(|&a| shape[a]).collect();
            f = apply(out, Primitive::Reshape { new_sizes }, &[f]);
            at
        }
    };

    let broadcast = Primitive::BroadcastInDim {
        shape: shape.to_vec(),
        broadcast_dimensions: dims,
    };

    apply(out, broadcast, &[f])
}

/// Whether `f` depends on a reduction over the last axis of rows wider than `tile`: a reduction a GEMM's
/// tiles don't hold whole, so applying `f` to its rows would take a pass over them of its own.
fn crosses_tiles(graph: &Graph, uses: &Uses, f: Var, tile: usize) -> bool {
    let nodes = graph.nodes();
    let (mut stack, mut seen) = (vec![f], vec![false; graph.types.len()]);

    while let Some(v) = stack.pop() {
        let Some(p) = uses.producer[v].filter(|_| !std::mem::replace(&mut seen[v], true)) else {
            continue;
        };

        let node = &nodes[p];

        if let Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } = &node.primitive {
            let x = graph.type_of(node.inputs[0]);
            let last = x.shape.len().saturating_sub(1);

            if *axes == [last] && x.shape.get(last).is_some_and(|&n| n > tile) {
                return true;
            }
        }

        stack.extend(&node.inputs);
    }

    false
}

/// Rewrites each matmul whose operand, read only by such matmuls, carries a factor constant along the
/// contracted axis: the matmul of the operands without their factors, in its accumulation dtype, times
/// (or divided by) each factor per output row or column, rounded once to its output dtype. With `tile`,
/// only a factor of a reduction over rows wider than `tile` moves ([`crosses_tiles`]): it is what would
/// otherwise take a pass over the rows after their GEMM's tiles; every other factor stays as traced.
pub(crate) fn linear_reordering(graph: &Graph, tile: Option<usize>) -> Graph {
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
            true => [0, 1].map(|side| {
                stripped(graph, &uses, k, side).filter(|(_, (f, _, _))| {
                    tile.is_none_or(|tile| crosses_tiles(graph, &uses, *f, tile))
                })
            }),
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
            Some((e, _)) => emit(&mut out, graph, &map, e),
            None => map[node.inputs[side]],
        });

        let shape = graph.type_of(node.output).shape.clone();
        out.set_origin(node);
        let mut v = out.apply(dot, &operands).expect("well typed");

        for (side, s) in reordered[k].iter().enumerate() {
            let Some((_, (f, divide, at))) = s else {
                continue;
            };

            // The factor applied after the matmul keeps the origin of the op that applied it before.
            out.set_origin(&nodes[*at]);
            let axes = kept(graph, node, side);
            let f = spread(
                &mut out,
                graph,
                &map,
                (*f, side, &axes),
                &shape,
                *accum_dtype,
            );
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

            out.set_origin(node);
            v = out.apply(cast, &[v]).expect("well typed");
        }

        map[node.output] = v;
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out.prune().0
}

/// Compiler pass running [`linear_reordering`].
pub(crate) struct LinearReordering {
    /// The widest row the backend's GEMM tiles hold whole; only factors of wider rows' reductions move.
    pub(crate) tile: usize,
}

impl<M: Module> CompilerPass<M> for LinearReordering {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "linear_reordering"
    }

    /// Replaces the module's graph with its reordered version.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = linear_reordering(m.graph(), Some(self.tile));
        Ok(())
    }
}
