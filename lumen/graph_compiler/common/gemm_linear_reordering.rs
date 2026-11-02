//! Pass moving a per-row factor of a matmul's lhs to its output (CODA's reordering, `f(x) W = f(x W)` for
//! a linear `f`): a matmul contracts each lhs row alone, so a row scaled by `r` gives that output row
//! scaled by `r`. A norm's `1 / rms` then multiplies the next matmul's result in its epilogue instead of
//! needing the whole normalized row before the matmul. The rounding moves with it: a cast between the
//! factor and the matmul rounds the unscaled value, so results differ from the traced program's in the
//! last bits (on by default, as the user chose; `gemm_linear_reordering` turns it off).

use crate::DType;
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};

/// How a matmul's lhs is built from its per-row factor's removal: the original value, or an op of the
/// rebuilt value and an original operand.
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

/// A per-row factor: the value holding it (`[rows...]` or `[rows..., 1]`), and whether the row was divided
/// by it.
type Factor = (Var, bool);

/// The value each node index produces, and the nodes each value is read by.
struct Uses {
    producer: Vec<Option<usize>>,
    readers: Vec<Vec<usize>>,
}

/// Whether node `k` is a matmul this pass can reorder: an unbatched float matmul of a rank-2+ lhs
/// contracted on its last axis with a 2-D rhs, reading `lhs` as its lhs.
fn plain(graph: &Graph, k: usize, lhs: Var) -> bool {
    let node = &graph.nodes()[k];

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

    node.inputs[0] == lhs
        && node.inputs[1] != lhs
        && lhs_batch.is_empty()
        && l.shape.len() >= 2
        && *lhs_contracting == [l.shape.len() - 1]
        && r.shape.len() == 2
        && rhs_contracting.len() == 1
        && accum_dtype.is_float()
}

/// The per-row value `b` broadcasts across the last axis of `like`'s shape, if it does.
fn row_factor(graph: &Graph, uses: &Uses, b: Var, like: Var) -> Option<Var> {
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
    let rows = &full[..full.len() - 1];
    let leading: Vec<usize> = (0..rows.len()).collect();
    let every: Vec<usize> = (0..full.len()).collect();

    let per_row = (source.shape == rows && *broadcast_dimensions == leading)
        || (source.shape.split_last() == Some((&1, rows)) && *broadcast_dimensions == every);

    (shape == full && full.last() > Some(&1) && source.dtype.is_float() && per_row)
        .then_some(node.inputs[0])
}

/// `v` without one per-row factor, found through multiplications, divisions by it, float casts and
/// negations: the rebuilt value and the factor.
fn strip(graph: &Graph, uses: &Uses, v: Var) -> Option<(Expr, Factor)> {
    let node = &graph.nodes()[uses.producer[v]?];
    let float = |u: Var| graph.type_of(u).dtype.is_float();

    match node.primitive {
        Primitive::Mul => {
            let [a, b] = node.inputs[..] else { return None };

            if let Some(f) = row_factor(graph, uses, b, v) {
                return Some((Expr::Old(a), (f, false)));
            }

            if let Some(f) = row_factor(graph, uses, a, v) {
                return Some((Expr::Old(b), (f, false)));
            }

            if let Some((e, f)) = strip(graph, uses, a) {
                return Some((Expr::Mul(Box::new(e), b), f));
            }

            let (e, f) = strip(graph, uses, b)?;
            Some((Expr::Mul(Box::new(e), a), f))
        }
        Primitive::Div => {
            let [a, b] = node.inputs[..] else { return None };

            if let Some(f) = row_factor(graph, uses, b, v) {
                return Some((Expr::Old(a), (f, true)));
            }

            let (e, f) = strip(graph, uses, a)?;
            Some((Expr::Div(Box::new(e), b), f))
        }
        Primitive::Cast { new_dtype } if float(node.inputs[0]) && new_dtype.is_float() => {
            let (e, f) = strip(graph, uses, node.inputs[0])?;
            Some((Expr::Cast(Box::new(e), new_dtype), f))
        }
        Primitive::Neg => {
            let (e, f) = strip(graph, uses, node.inputs[0])?;
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

/// Rewrites each matmul whose lhs, read only by such matmuls, carries a per-row factor: the matmul of the
/// lhs without it, in its accumulation dtype, times (or divided by) the factor per output row, rounded
/// once to its output dtype.
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

    // The rebuilt lhs of each reorderable matmul, by node index.
    let mut reordered: Vec<Option<(Expr, Factor)>> = (0..nodes.len()).map(|_| None).collect();

    for (k, node) in nodes.iter().enumerate() {
        let Some(&lhs) = node.inputs.first() else {
            continue;
        };

        let only_matmuls = !graph.outputs().contains(&lhs)
            && uses.readers[lhs].iter().all(|&r| plain(graph, r, lhs));

        if plain(graph, k, lhs) && only_matmuls {
            reordered[k] = strip(graph, &uses, lhs);
        }
    }

    if reordered.iter().all(Option::is_none) {
        return graph.clone();
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    for (k, node) in nodes.iter().enumerate() {
        out.set_origin(node);

        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

        let Some((expr, (factor, divide))) = &reordered[k] else {
            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("well typed");
            continue;
        };

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

        let lhs = emit(&mut out, &map, expr);
        let product = out.apply(dot, &[lhs, inputs[1]]).expect("well typed");
        let shape = graph.type_of(node.output).shape.clone();
        let rows = shape.len() - 1;
        let mut f = map[*factor];

        if graph.type_of(*factor).shape.len() != rows {
            let new_sizes = shape[..rows].to_vec();
            f = out
                .apply(Primitive::Reshape { new_sizes }, &[f])
                .expect("well typed");
        }

        if graph.type_of(*factor).dtype != *accum_dtype {
            let cast = Primitive::Cast {
                new_dtype: *accum_dtype,
            };

            f = out.apply(cast, &[f]).expect("well typed");
        }

        let spread = Primitive::BroadcastInDim {
            shape,
            broadcast_dimensions: (0..rows).collect(),
        };

        let f = out.apply(spread, &[f]).expect("well typed");
        let op = if *divide {
            Primitive::Div
        } else {
            Primitive::Mul
        };
        let mut v = out.apply(op, &[product, f]).expect("well typed");

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
