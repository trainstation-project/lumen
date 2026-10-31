//! Grows normalization rows forward over the elementwise ops that consume them.

use super::analysis::Analysis;
use super::elementwise;
use crate::graph_compiler::common::diamonds::Row;

/// Extends each row's root through its first elementwise reader while that keeps the row a single kernel,
/// adding the old root to the row's outputs when others still read it and it can't be recomputed.
pub(super) fn extended(a: &Analysis, rows: &[Row]) -> Vec<Row> {
    let (nodes, users, is_output, fusible) = (a.nodes, &a.users, &a.is_output, &a.fusible);
    let depends = |v, k| a.depends(v, k);
    let recomputed = |i| a.recomputed(i);

    let owned = |u: usize, other: &Row| {
        rows.iter().any(|r| {
            !std::ptr::eq(r, other)
                && (r.root == u || r.inner.contains(&u) || r.reductions.contains(&u))
        })
    };

    rows.iter()
        .map(|original| {
            let mut row = original.clone();

            loop {
                let v = nodes[row.root].output;
                let Some(&u) = users[v].first() else { break };
                let inside = |r: usize| r == row.root || row.inner.contains(&r);
                let after = |k: usize| users[nodes[k].output].iter().all(|&r| inside(r) || r >= u);

                let extends = !is_output[v]
                    && !owned(u, original)
                    && fusible[u]
                    && elementwise(&nodes[u].primitive)
                    && users[v][1..].iter().all(|&r| r > u)
                    && row.outputs.iter().all(|&k| after(k))
                    && row.partials.iter().all(|&p| {
                        let v = nodes[u].output;
                        !nodes[u].inputs.iter().any(|&w| w != v && depends(w, p))
                    });

                if !extends {
                    break;
                }

                if users[v].len() > 1 && !recomputed(row.root) {
                    row.outputs.push(row.root);
                }

                row.inner.push(row.root);
                row.root = u;
            }

            row
        })
        .collect()
}
