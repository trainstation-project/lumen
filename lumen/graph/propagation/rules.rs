//! Strategy families many primitives share: elementwise ops by how far they are linear in their partial operands,
//! reductions, and the dimension groups a reshape maps whole.

use super::{Op, Placement, Strategy};

/// How far an elementwise op is linear in its partial operands, so keeps them partial.
#[derive(Clone, Copy)]
pub(super) enum Linear {
    /// Not linear: no partial operand stays partial.
    No,
    /// Linear in all operands at once (`add`): partial sums alike stay partial.
    Sum,
    /// Linear in each operand alone (`mul`): one partial sum, the others replicated.
    Scale,
    /// Linear in its first operand (`div`): a partial sum there, the others replicated.
    Numerator,
    /// The reduction itself (`max`, `min`): partial values of that reduction alike stay partial.
    Same(Op),
    /// An identity (a transfer or wait): any partial value stays partial.
    Any,
}

/// `placements` without repeats, in order.
pub(super) fn unique(placements: &[Placement]) -> Vec<Placement> {
    let mut out = Vec::new();

    for &p in placements {
        if !out.contains(&p) {
            out.push(p);
        }
    }

    out
}

/// An elementwise op's strategies: every operand sharded as one of them is, and partial ones as `linear` allows.
pub(super) fn elementwise(placements: &[Placement], linear: Linear) -> Vec<Strategy> {
    use Placement::*;

    let n = placements.len();
    let distinct = unique(placements);

    let mut out: Vec<Strategy> = distinct
        .iter()
        .filter(|p| matches!(p, Shard(_)))
        .map(|&p| (vec![p; n], p))
        .collect();

    for &p in &distinct {
        let Partial(op) = p else {
            continue;
        };

        match linear {
            Linear::Any => out.push((vec![p; n], p)),
            Linear::Sum if p.summed() => out.push((vec![p; n], p)),
            Linear::Same(q) if q == op => out.push((vec![p; n], p)),
            Linear::Scale if p.summed() => {
                out.extend((0..n).map(|k| {
                    let ins = (0..n).map(|j| if j == k { p } else { Replicate }).collect();
                    (ins, p)
                }));
            }
            Linear::Numerator if p.summed() => {
                let mut ins = vec![Replicate; n];
                ins[0] = p;
                out.push((ins, p));
            }
            _ => {}
        }
    }

    out
}

/// A reduction by `op` over `axes` of an operand placed `p`: a shard along a reduced axis made partial, along a
/// kept one kept (renumbered); a value partial by the same reduction (or an average, for a sum) kept.
pub(super) fn reduce(p: Placement, axes: &[usize], op: Op) -> Vec<Strategy> {
    use Placement::*;

    match p {
        Shard(d) if axes.contains(&d) => vec![(vec![p], Partial(op))],
        Shard(d) => vec![(vec![p], Shard(d - axes.iter().filter(|&&a| a < d).count()))],
        Partial(q) if q == op || (op == Op::Sum && p.summed()) => vec![(vec![p], p)],
        _ => Vec::new(),
    }
}

/// The dims of shapes `a` and `b` (of one element count, none zero) in runs of equal products, in order: a
/// reshape from `a` to `b` merges or splits each run's dims of `a` into its dims of `b`.
pub(super) fn groups(a: &[usize], b: &[usize]) -> Vec<(Vec<usize>, Vec<usize>)> {
    let (mut i, mut j, mut out) = (0, 0, Vec::new());

    while i < a.len() || j < b.len() {
        let (mut mine, mut theirs, mut x, mut y) = (Vec::new(), Vec::new(), 1, 1);

        if i < a.len() {
            mine.push(i);
            x = a[i];
            i += 1;
        }

        if j < b.len() {
            theirs.push(j);
            y = b[j];
            j += 1;
        }

        while x != y {
            if x < y {
                mine.push(i);
                x *= a[i];
                i += 1;
            } else {
                theirs.push(j);
                y *= b[j];
                j += 1;
            }
        }

        out.push((mine, theirs));
    }

    out
}
