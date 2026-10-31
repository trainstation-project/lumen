//! Triton dynamic-update-slices, one at runtime starts or a fused chain at constant starts: the `BODY` of
//! the fixed `dynamic_update_slice_kernel` (`__init__.py`).

use std::fmt::Write;

use super::DynamicSliceCodegen;
use crate::codegen::triton::*;
use crate::graph::{Graph, TensorType};
use crate::tensor::contiguous_strides;

/// Triton dynamic slices: the fixed `dynamic_update_slice_kernel` runs a generated `BODY` over the update's
/// elements, loading and storing through helpers for its tensors' layouts.
impl DynamicSliceCodegen for Triton {
    /// The `BODY` function's name and Python source.
    type Code = (String, String);
    type Options = Tensors;

    /// The `BODY(EXTRA, BLOCK, MASK)` of `updates` fusion `body`: every update (inputs `1..`, one shape)
    /// written into `out` (already holding operand input `0`) at its constant start indices, clamped so it
    /// stays in bounds; element `BLOCK` of every update is written, except where a later update
    /// overlapping it writes over it.
    fn updates(body: &Graph, layouts: &Tensors) -> Result<(String, String), String> {
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
            .map(|d| format!("    c{d} = (BLOCK // {}) % {}\n", spans[d], update.shape[d]))
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
            true => "BLOCK".to_string(),
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
                        true => "(BLOCK >= 0)".to_string(),
                        false => format!("({})", inside.join(" & ")),
                    }
                })
                .collect();

            let mask = match covered.is_empty() {
                true => "MASK".to_string(),
                false => format!("MASK & ~({})", covered.join(" | ")),
            };

            let k = k + 1;

            writeln!(
                code,
                "    STORE(out, at + {offset}, LOAD{k}(in{k}, BLOCK, {mask}), {mask})"
            )
            .unwrap();
        }

        Ok(body_function(
            "dynamic_update_slice",
            layouts,
            "BLOCK, MASK",
            &code,
        ))
    }

    /// The `BODY(EXTRA, BLOCK, MASK)` writing elements `BLOCK` of `update` into `out` (already holding
    /// `x`) at runtime start indices (inputs `2..`), clamped so the update stays in bounds.
    fn update(
        x: &TensorType,
        update: &TensorType,
        by_value: &[bool],
        layouts: &Tensors,
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
                    "((BLOCK // {}) % {} + tl.minimum(tl.maximum({start}, 0), {most})) * {}",
                    spans[d], update.shape[d], strides[d]
                )
            })
            .collect();

        let at = match terms.is_empty() {
            true => "BLOCK".to_string(),
            false => terms.join(" + "),
        };

        let code = format!("    STORE(out, {at}, LOAD1(in1, BLOCK, MASK), MASK)\n");

        Ok(body_function(
            "dynamic_update_slice",
            layouts,
            "BLOCK, MASK",
            &code,
        ))
    }
}
