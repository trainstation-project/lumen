//! Triton row kernels for normalization diamonds (softmax, RMSNorm, LayerNorm): each program holds
//! whole rows in registers, reduces them along the last axis and computes the rest of the row from
//! those per-row values, so the row is read once and its reductions never reach memory.

use indoc::writedoc;

use super::config::MOST_ROW_COLUMNS;
use super::reduction::{Reduction, combine, lowest, reduce_along};
use super::*;
use crate::graph_compiler::elementwise;

/// The body's last-axis reductions, in order.
fn reductions(body: &Graph) -> impl Iterator<Item = &Node> {
    body.nodes().iter().filter(|n| {
        matches!(
            n.primitive,
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
        )
    })
}

/// Whether a row fusion's `body` runs as one Triton row kernel: rows of at most `MOST_ROW_COLUMNS`
/// reduced along their last axis, and per-row values (the reductions and what is computed from them
/// alone) only reshaped, combined with each other, or broadcast back across their rows.
pub(crate) fn row_fits(body: &Graph) -> bool {
    let root = body.type_of(body.outputs()[0]);
    let Some(&n) = root.shape.last() else {
        return false;
    };
    let rows = root.numel() / n.max(1);
    let last = root.shape.len() - 1;
    let mut per_row = vec![false; body.types.len()];
    for node in body.nodes() {
        let read = node.inputs.iter().any(|&v| per_row[v]);
        let out = body.type_of(node.output);
        match &node.primitive {
            Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => {
                if read || body.type_of(node.inputs[0]).shape != root.shape || axes != &[last] {
                    return false;
                }
                per_row[node.output] = true;
            }
            _ if !read => {}
            Primitive::BroadcastInDim {
                broadcast_dimensions,
                ..
            } => {
                let x = body.type_of(node.inputs[0]);
                let along = broadcast_dimensions
                    .iter()
                    .zip(&x.shape)
                    .all(|(&d, &size)| d != last || size == 1);
                if out.shape != root.shape || !along {
                    return false;
                }
            }
            p if (elementwise(p) || matches!(p, Primitive::Reshape { .. }))
                && out.numel() == rows =>
            {
                per_row[node.output] = true
            }
            _ => return false,
        }
    }
    n > 1 && n <= MOST_ROW_COLUMNS
}

/// Generates the row kernel for a row fusion's `body`: `BLOCK_SIZE // lanes` rows per program, each
/// row's last-axis reductions over the whole row, then the outputs, per element or per row. The first
/// output is `out` (`STORE`), output `k` after it `out{k}` (`STORE{k}`); `n` is the row count.
pub(crate) fn row(body: &Graph, by_value: &[bool]) -> Reduction {
    let root = body.type_of(body.outputs()[0]);
    let n = *root.shape.last().expect("a row");
    let rows = root.numel() / n;
    let lanes = n.next_power_of_two();
    let mut emitter = Emitter::new(body, by_value);
    let mut combines = String::new();
    for (k, node) in reductions(body).enumerate() {
        let a = body.type_of(node.output).dtype;
        let acc = triton_type(a).expect("a Triton dtype");
        let sum = matches!(node.primitive, Primitive::ReduceSum { .. });
        let x = emitter.value(node.inputs[0], "j".into());
        let x = match body.type_of(node.inputs[0]).dtype == a {
            true => x,
            false => format!("{x}.to({acc})"),
        };
        let init = match sum {
            true => "0".to_string(),
            false => lowest(a),
        };
        let masked = format!("tl.where(mask, {x}, tl.full([], {init}, {acc}))");
        let call = reduce_along(&masked, 1, sum, a, &format!("NAME_combine{k}"));
        let value = emitter.local(format!("{call}[:, None]"));
        emitter.reduced.insert(node.output, value);
        if !call.starts_with("tl.reduce") {
            continue;
        }
        writedoc!(
            combines,
            "
            @triton.jit
            def NAME_combine{k}(a, b):
                return ({}).to(a.dtype)


            ",
            combine(sum, a)
        )
        .unwrap();
    }
    // Each output is stored per element of the rows, or once per row.
    let mut stores = String::new();
    for (k, &v) in body.outputs().iter().enumerate() {
        let (index, mask) = match body.type_of(v).numel() == rows {
            true => ("o", "row_mask"),
            false => ("j", "mask"),
        };
        emitter.mask = mask.into();
        let value = emitter.value(v, index.into());
        let store = match k {
            0 => "STORE(out".to_string(),
            k => format!("STORE{k}(out{k}"),
        };
        writeln!(stores, "    {store}, {index}, {value}, {mask})").unwrap();
    }
    let extra = (1..body.outputs().len())
        .map(|k| format!("out{k}"))
        .collect::<Vec<_>>()
        .join(",\n    ");
    let params = params(
        by_value,
        (!extra.is_empty()).then_some(extra.as_str()),
        body.outputs().len(),
    );
    let lines = &emitter.lines;
    let source = formatdoc!(
        "
        {combines}@triton.jit
        def NAME(
        {params}):
            o = (tl.program_id(0).to(tl.int64) * (BLOCK_SIZE // {lanes}) + tl.arange(0, BLOCK_SIZE // {lanes}))[:, None]
            c = tl.arange(0, {lanes})[None, :]
            j = o * {n} + c
            row_mask = o < n
            mask = row_mask & (c < {n})
        {lines}{stores}"
    );
    let (name, source) = named("row", source);
    Reduction {
        name,
        source,
        lanes,
        splits: 1,
    }
}
