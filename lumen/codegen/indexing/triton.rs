//! Triton sources for indexed writes: dynamic-update-slice and scatter-add.

use std::fmt::Write;

use indoc::formatdoc;

use super::IndexingCodegen;
use crate::codegen::triton::*;
use crate::graph::{Graph, TensorType};
use crate::tensor::contiguous_strides;

/// Triton indexing: a generated kernel per op.
impl IndexingCodegen for Triton {
    /// The kernel's name and source.
    type Code = (String, String);

    /// The kernel of `updates` fusion `body`: every update (inputs `1..`, one shape) written into `out`
    /// (already holding operand input `0`) at its constant start indices, clamped so it stays in bounds;
    /// each thread writes element `j` of every update, except where a later update overlapping it writes
    /// over it.
    fn updates(body: &Graph) -> Result<(String, String), String> {
        let starts: Vec<Vec<i64>> = crate::graph_compiler::common::updates::starts(body)
            .into_iter()
            .map(|(_, s)| s)
            .collect();

        let ty = |k: usize| body.type_of(body.inputs()[k]);
        let (x, update) = (ty(0), ty(1));

        let (spans, strides) = (
            contiguous_strides(&update.shape),
            contiguous_strides(&x.shape),
        );

        let rank = x.shape.len();

        // Element `j`'s coordinates in the update, then its offset in `x` before a start is added.
        let mut code: String = (0..rank)
            .filter(|&d| update.shape[d] != 1)
            .map(|d| format!("    c{d} = (j // {}) % {}\n", spans[d], update.shape[d]))
            .collect();

        let coordinate = |d: usize| match update.shape[d] {
            1 => "0".to_string(),
            _ => format!("c{d}"),
        };

        let terms: Vec<String> = (0..rank)
            .filter(|&d| update.shape[d] != 1)
            .map(|d| format!("c{d} * {}", strides[d]))
            .collect();

        let at = match terms.is_empty() {
            true => "j".to_string(),
            false => terms.join(" + "),
        };

        writeln!(code, "    at = {at}").unwrap();

        let chain =
            crate::graph_compiler::common::updates::overlaps(&x.shape, &update.shape, &starts);

        for (k, (start, later)) in chain.iter().enumerate() {
            let offset: usize = (0..rank).map(|d| start[d] * strides[d]).sum();

            // Elements a later overlapping update writes over are left to it.
            let covered: Vec<String> = later
                .iter()
                .map(|&l| {
                    let inside: Vec<String> = (0..rank)
                        .filter(|&d| update.shape[d] != 1 || start[d] != chain[l].0[d])
                        .map(|d| {
                            let (c, s, t) = (coordinate(d), start[d], chain[l].0[d]);
                            format!("({c} + {s} >= {t}) & ({c} + {s} < {})", t + update.shape[d])
                        })
                        .collect();

                    match inside.is_empty() {
                        true => "(j >= 0)".to_string(),
                        false => format!("({})", inside.join(" & ")),
                    }
                })
                .collect();

            let mask = match covered.is_empty() {
                true => "mask".to_string(),
                false => format!("mask & ~({})", covered.join(" | ")),
            };

            let k = k + 1;

            writeln!(
                code,
                "    STORE(out, at + {offset}, LOAD{k}(in{k}, j, {mask}), {mask})"
            )
            .unwrap();
        }

        let params = params(&vec![false; starts.len() + 1], None, 1);

        let source = formatdoc!(
            "
            @triton.jit
            def NAME(
            {params}):
                j = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = j < n
            {code}"
        );

        Ok(named("updates", source))
    }

    /// The dynamic-update-slice kernel writing `update` into `out` (already holding `x`) at runtime start
    /// indices (inputs `2..`), clamped so the update stays in bounds.
    fn update(
        x: &TensorType,
        update: &TensorType,
        by_value: &[bool],
    ) -> Result<(String, String), String> {
        // Input `k` read as an int64 scalar, from a by-value argument or a 1-element tensor.
        let scalar = |k: usize| match by_value[k] {
            true => format!("tl.full([], in{k}, tl.int64)"),
            false => format!("LOAD{k}(in{k}, 0, None).to(tl.int64)"),
        };

        let (spans, strides) = (
            contiguous_strides(&update.shape),
            contiguous_strides(&x.shape),
        );

        // Per dimension: the update coordinate of flat index `j` plus the clamped start, times x's stride.
        let terms: Vec<String> = (0..x.shape.len())
            .filter(|&d| x.shape[d] != 1)
            .map(|d| {
                let start = scalar(2 + d);
                let most = x.shape[d] - update.shape[d];

                format!(
                    "((j // {}) % {} + tl.minimum(tl.maximum({start}, 0), {most})) * {}",
                    spans[d], update.shape[d], strides[d]
                )
            })
            .collect();

        let at = match terms.is_empty() {
            true => "j".to_string(),
            false => terms.join(" + "),
        };

        let params = params(by_value, None, 1);

        let source = formatdoc!(
            "
            @triton.jit
            def NAME(
            {params}):
                j = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = j < n
                STORE(out, {at}, LOAD1(in1, j, mask), mask)
            "
        );

        Ok(named("update", source))
    }

    /// The kernel adding `picks` slices of input 2 into `out` at the `axis` positions listed in input 1;
    /// each program owns distinct `(outer, inner)` elements so the loop over picks is race-free.
    fn scatter_add(x: &TensorType, axis: usize, picks: usize) -> Result<(String, String), String> {
        let after: usize = x.shape[axis + 1..].iter().product();
        let length = x.shape[axis];

        let params = params(&[false; 3], None, 1).replace(
            "    BLOCK_SIZE: tl.constexpr,",
            "    OUTLOAD: tl.constexpr,\n    BLOCK_SIZE: tl.constexpr,",
        );

        let source = formatdoc!(
            "
            @triton.jit
            def NAME(
            {params}):
                j = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = j < n
                o = j // {after}
                r = j % {after}
                for t in range(0, {picks}):
                    k = LOAD1(in1, t, None).to(tl.int64)
                    at = (o * {length} + k) * {after} + r
                    value = OUTLOAD(out, at, mask) + LOAD2(in2, (o * {picks} + t) * {after} + r, mask)
                    STORE(out, at, value, mask)
            "
        );

        Ok(named("scatter_add", source))
    }
}
