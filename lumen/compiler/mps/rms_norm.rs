//! RMS norm rewriting: the primitives an RMS norm traces to, however it is
//! written (`lumen.rms_norm`, or by hand as `x / (x * x).mean(-1,
//! keepdim=True).add(eps).sqrt()` or `x * (1 / (..).sqrt())`, each
//! multiply's operands in either order), times a weight or not,
//!
//! ```text
//! s = sqrt(sum(x * x) / n [+ eps])  over the last dimension, of size n
//! y = x * broadcast(1 / s)  or  x / broadcast(s)  [* broadcast(weight)]
//! ```
//!
//! become one [`Primitive::RmsNorm`], which
//! runs as one kernel (`ops/rms_norm/mps.metal`), when nothing else reads
//! the values in between ([`Matcher::exclusive`]).

use crate::DType;
use crate::compiler::pattern::{Match, Matcher, Pattern, bind, either, one_of, op};
use crate::graph::{Graph, Primitive, Var};

// Captures.
const X: usize = 0;
const N: usize = 1;
const EPS: usize = 2;
const W: usize = 3;
const SUM: usize = 4;
const R: usize = 5;
const WB: usize = 6;
const ONE: usize = 7;
const CAPTURES: usize = 8;

/// The RMS norm of `x` without its weight: `x * broadcast(1 / sqrt(..))`
/// or `x / broadcast(sqrt(..))`, the reciprocal or square root captured as
/// `R`.
fn normalized() -> Pattern {
    use Primitive::*;
    let mean = || {
        op(
            |p| matches!(p, Div),
            [
                op(
                    |p| matches!(p, Reshape { .. }),
                    [op(
                        |p| matches!(p, ReduceSum { .. }),
                        [either(|p| matches!(p, Mul), [bind(X), bind(X)])],
                    )
                    .bind(SUM)],
                ),
                bind(N),
            ],
        )
    };
    // The square root of the mean, plus epsilon or not (epsilon 0).
    let sqrt = || {
        op(
            |p| matches!(p, Sqrt),
            [one_of([
                either(|p| matches!(p, Add), [mean(), bind(EPS)]),
                mean(),
            ])],
        )
    };
    let broadcast = |p: Pattern| op(|p| matches!(p, BroadcastInDim { .. }), [p]);
    one_of([
        either(
            |p| matches!(p, Mul),
            [
                bind(X),
                broadcast(op(|p| matches!(p, Div), [bind(ONE), sqrt()]).bind(R)),
            ],
        ),
        op(|p| matches!(p, Div), [bind(X), broadcast(sqrt().bind(R))]),
    ])
}

/// `graph` with each RMS norm over the last dimension one primitive (the
/// primitives it replaces are left dead).
pub(crate) fn rewrite_rms_norm(graph: &Graph) -> Graph {
    let matcher = Matcher::new(graph);
    let weighted = either(
        |p| matches!(p, Primitive::Mul),
        [
            normalized(),
            op(|p| matches!(p, Primitive::BroadcastInDim { .. }), [bind(W)]).bind(WB),
        ],
    );
    let plain = normalized();
    let nodes = graph.nodes();
    // Weighted norms first, so their weight's multiply is part of them.
    let mut found: Vec<Option<(Match, f64)>> = vec![None; nodes.len()];
    let mut taken = vec![false; nodes.len()];
    for pattern in [&weighted, &plain] {
        for (i, node) in nodes.iter().enumerate().rev() {
            if taken[i] {
                continue;
            }
            let Some(m) = matcher.find(pattern, node.output, CAPTURES) else {
                continue;
            };
            if let Some(eps) = rms_norm(graph, &matcher, &m) {
                m.nodes.iter().for_each(|&j| taken[j] = true);
                found[i] = Some((m, eps));
            }
        }
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for (i, node) in nodes.iter().enumerate() {
        map[node.output] = match &found[i] {
            Some((m, epsilon)) => {
                let operands: Vec<Var> = [Some(m.get(X)), m.captures[W]]
                    .into_iter()
                    .flatten()
                    .map(|v| map[v])
                    .collect();
                out.apply(Primitive::RmsNorm { epsilon: *epsilon }, &operands)
            }
            None => {
                let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
                out.apply(node.primitive.clone(), &inputs)
            }
        }
        .expect("a rewrite keeps the node's type");
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// The epsilon of match `m`, if it is an RMS norm over the last dimension:
/// the sum over that dimension, divided by its size, the broadcasts back
/// along it (the weight's, of its size), a float dtype the kernel takes,
/// and nothing else reading what it replaces.
fn rms_norm(graph: &Graph, matcher: &Matcher, m: &Match) -> Option<f64> {
    let x = graph.type_of(m.get(X));
    let last = x.shape.len().checked_sub(1)?;
    let n = x.shape[last];
    let sum = matcher.node(m.get(SUM))?;
    let r = graph.type_of(m.get(R));
    let mut kept = x.shape.clone();
    kept[last] = 1;
    let weight_ok = match m.captures[W] {
        None => true,
        Some(w) => {
            let Some(Primitive::BroadcastInDim {
                broadcast_dimensions,
                ..
            }) = matcher.node(m.get(WB)).map(|b| &b.primitive)
            else {
                return None;
            };
            graph.type_of(w).shape.as_slice() == [n] && broadcast_dimensions.as_slice() == [last]
        }
    };
    let ok = matches!(x.dtype, DType::F16 | DType::BF16 | DType::F32)
        && sum.primitive == (Primitive::ReduceSum { axes: vec![last] })
        && r.shape == kept
        && matcher.scalar(m.get(N)) == Some(n as f64)
        && m.captures[ONE].is_none_or(|one| matcher.scalar(one) == Some(1.0))
        && weight_ok
        && matcher.exclusive(m);
    let epsilon = match m.captures[EPS] {
        Some(eps) => matcher.scalar(eps),
        None => Some(0.0),
    };
    ok.then_some(epsilon).flatten()
}
