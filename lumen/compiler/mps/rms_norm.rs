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
//! run as one kernel, when nothing else reads the values in between
//! ([`Matcher::exclusive`]): a fusion of these primitives with its
//! reduction inside (no primitive of its own), which the codegen makes a
//! kernel a threadgroup a row (`codegen.rs`: a row kernel).

use crate::DType;
use crate::compiler::pattern::{Match, Matcher, Pattern, bind, either, one_of, op};
use crate::graph::{Graph, Primitive};

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

/// Each RMS norm over the last dimension in `graph`: its root node (the
/// normalized, or weighted, value) and its reduction's, by index. The
/// fusion pass fuses each into one row kernel, its reduction inside.
pub(crate) fn rms_norms(graph: &Graph) -> Vec<(usize, usize)> {
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
    let mut found = Vec::new();
    let mut taken = vec![false; nodes.len()];
    for pattern in [&weighted, &plain] {
        for (i, node) in nodes.iter().enumerate().rev() {
            if taken[i] {
                continue;
            }
            let Some(m) = matcher.find(pattern, node.output, CAPTURES) else {
                continue;
            };
            if rms_norm(graph, &matcher, &m).is_some() {
                m.nodes.iter().for_each(|&j| taken[j] = true);
                found.push((i, matcher.index(m.get(SUM)).expect("a reduction")));
            }
        }
    }
    found
}

/// Whether match `m` is an RMS norm over the last dimension: the sum over
/// that dimension, divided by its size, plus a scalar epsilon (or not),
/// the broadcasts back along it (the weight's, of its size), a float dtype,
/// and nothing else reading its values in between.
fn rms_norm(graph: &Graph, matcher: &Matcher, m: &Match) -> Option<()> {
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
    (ok && m.captures[EPS].is_none_or(|eps| matcher.scalar(eps).is_some())).then_some(())
}
