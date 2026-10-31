//! Triton row kernels for normalization diamonds (softmax, RMSNorm, LayerNorm): each row's last-axis
//! reductions, then the rest of the row computed from those per-row values, in one kernel. Rows up to
//! `MOST_WHOLE_ROW` are held whole in registers and read once; wider rows are walked in chunks of
//! `BLOCK_SIZE`, a pass a reduction (a max and the sum of `exp(x - max)` sharing one, online) and a last
//! pass writing the outputs.

use std::fmt::Write;

use indoc::{formatdoc, writedoc};

use super::RowCodegen;
use crate::DType;
use crate::codegen::reduction::triton::Reduction;
use crate::codegen::reduction::triton::{combine, lowest, reduce_along};
use crate::codegen::triton::*;
use crate::codegen::utils::triton::indented;
use crate::graph::TensorType;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::common::diamonds::Partials;
use crate::graph_compiler::common::diamonds::softmax_pair;
use crate::graph_compiler::elementwise;

/// Widest row a normalization row kernel holds whole in registers (XMA's `MAX_TRITON_BLOCK_SIZE`);
/// wider rows are walked in chunks of `BLOCK_SIZE`, a pass a reduction.
const MOST_WHOLE_ROW: usize = 65536;

/// How column sums (a norm weight's gradient) ride along row kernels: rows up to 8192 wide, each
/// program holding a partial per column, split into 32 to 256 row groups.
pub(crate) const ROW_PARTIALS: Partials = Partials {
    columns: 8192,
    groups: 256,
    fewest_groups: 32,
};

/// The rows a row fusion's `body` reduces: the operand of its first last-axis reduction, or its root
/// when it has none. A row's results may be its rows' shape or one a row (a cross entropy's loss).
pub(crate) fn rows_type(body: &Graph) -> &TensorType {
    body.nodes()
        .iter()
        .find_map(|n| match &n.primitive {
            Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => {
                let x = body.type_of(n.inputs[0]);
                (axes == &[x.shape.len().saturating_sub(1)]).then_some(x)
            }
            _ => None,
        })
        .unwrap_or_else(|| body.type_of(body.outputs()[0]))
}

/// The row count of a row fusion's `body`.
pub(crate) fn row_count(body: &Graph) -> usize {
    rows_type(body).numel() / width(body).max(1)
}

/// The body's last-axis reductions of its rows, in order.
pub(crate) fn reductions(body: &Graph) -> Vec<&Node> {
    let root = &rows_type(body).shape;

    body.nodes()
        .iter()
        .filter(|n| {
            matches!(
                n.primitive,
                Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
            ) && &body.type_of(n.inputs[0]).shape == root
        })
        .collect()
}

/// The body's column partials: sums over each group of rows (a `[groups, rows / groups, n]` reshape of
/// the rows reduced along its middle axis), written per group.
pub(crate) fn column_partials(body: &Graph) -> Vec<&Node> {
    let root = &rows_type(body).shape;

    body.nodes()
        .iter()
        .filter(|n| {
            matches!(n.primitive, Primitive::ReduceSum { .. })
                && &body.type_of(n.inputs[0]).shape != root
        })
        .collect()
}

/// The row groups of a row fusion's `body` whose column partials each program sums, one group a
/// program; `None` without partials.
pub(crate) fn row_groups(body: &Graph) -> Option<usize> {
    let partial = column_partials(body).into_iter().next()?;
    Some(body.type_of(partial.output).shape[0])
}

/// Row width of a row fusion's `body`: its rows' last dimension.
pub(crate) fn width(body: &Graph) -> usize {
    rows_type(body).shape.last().copied().unwrap_or(1)
}

/// Whether a row fusion's `body` runs as one Triton row kernel: rows reduced along their last axis,
/// and per-row values (the reductions and what is computed from them alone) only reshaped, combined
/// with each other, or broadcast back across their rows; its outputs of its rows' shape or one a row.
pub(crate) fn row_fits(body: &Graph) -> bool {
    let root = rows_type(body);

    let Some(&n) = root.shape.last() else {
        return false;
    };

    let rows = root.numel() / n.max(1);
    let last = root.shape.len() - 1;
    let mut per_row = vec![false; body.types.len()];
    let mut partial = vec![false; body.types.len()];

    for node in body.nodes() {
        let read = node.inputs.iter().any(|&v| per_row[v]);
        let out = body.type_of(node.output);
        let x = body.type_of(node.inputs.first().copied().unwrap_or(node.output));

        if node.inputs.iter().any(|&v| partial[v]) {
            return false;
        }

        match &node.primitive {
            // A column partial: whole rows, each program summing its own group of them.
            Primitive::ReduceSum { axes, .. } if x.shape != root.shape => {
                let grouped = x.shape.len() == 3
                    && x.shape[2] == n
                    && x.numel() == root.numel()
                    && axes == &[1];

                if read || !grouped || n > MOST_WHOLE_ROW {
                    return false;
                }

                partial[node.output] = true;
            }
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

    let written = |v: &Var| {
        let numel = body.type_of(*v).numel();
        numel == rows || numel == root.numel() || partial[*v]
    };

    n > 1 && body.outputs().iter().all(written)
}

/// Whether a row fusion's `body` is too wide to hold a row whole, so its kernel walks rows in chunks.
pub(crate) fn row_looped(body: &Graph) -> bool {
    width(body) > MOST_WHOLE_ROW
}

/// Value of reduction `node`'s operand at `j`, in its accumulation dtype; the dtype and its Triton type.
pub(crate) fn operand(emitter: &mut Emitter, node: &Node) -> (String, DType, &'static str) {
    let a = emitter.body.type_of(node.output).dtype;
    let acc = triton_type(a).expect("a Triton dtype");
    let x = emitter.value(node.inputs[0], "j".into());

    match emitter.body.type_of(node.inputs[0]).dtype == a {
        true => (x, a, acc),
        false => (format!("{x}.to({acc})"), a, acc),
    }
}

/// Identity of reduction `node` (0 or the lowest value) as a Triton literal.
pub(crate) fn identity(node: &Node, dtype: DType) -> String {
    match node.primitive {
        Primitive::ReduceSum { .. } => "0".into(),
        _ => lowest(dtype),
    }
}

/// Appends `NAME_combine{k}`, the elementwise combine of reduction `node` accumulating in `dtype`.
pub(crate) fn combined(combines: &mut String, k: usize, node: &Node, dtype: DType) {
    let sum = matches!(node.primitive, Primitive::ReduceSum { .. });

    writedoc!(
        combines,
        "
        @triton.jit
        def NAME_combine{k}(a, b):
            return ({}).to(a.dtype)


        ",
        combine(sum, dtype)
    )
    .unwrap();
}

/// Reduces reduction `k`, `node`, over whole rows of the tile (`mask`) and binds its per-row value.
pub(crate) fn whole(emitter: &mut Emitter, k: usize, node: &Node, combines: &mut String) {
    let sum = matches!(node.primitive, Primitive::ReduceSum { .. });
    let (x, a, acc) = operand(emitter, node);

    let masked = format!(
        "tl.where(mask, {x}, tl.full([], {}, {acc}))",
        identity(node, a)
    );

    let call = reduce_along(&masked, 1, sum, a, &format!("NAME_combine{k}"));
    let value = emitter.local(format!("{call}[:, None]"));
    emitter.reduced.insert(node.output, value);

    if call.starts_with("tl.reduce") {
        combined(combines, k, node, a);
    }
}

/// A loop over a row's chunks whose body is `emit`'s statements; values computed inside stay there.
fn chunks(emitter: &mut Emitter, n: usize, emit: impl FnOnce(&mut Emitter)) -> String {
    emitter.values.clear();
    emitter.indices.clear();
    emitter.mask = "mask".into();
    emit(emitter);

    let lines: String = std::mem::take(&mut emitter.lines)
        .lines()
        .map(|l| format!("    {l}\n"))
        .collect();

    emitter.values.clear();
    emitter.indices.clear();

    formatdoc!(
        "
            for start in range(0, {n}, BLOCK_SIZE):
                c = start + tl.arange(0, BLOCK_SIZE)[None, :]
                j = o * {n} + c
                mask = row_mask & (c < {n})
        {lines}"
    )
}

/// Generates the row kernel for a row fusion's `body`: whole rows, `BLOCK_SIZE // lanes` a program, or
/// (`row_looped`) one row a program walked in chunks of `BLOCK_SIZE`, or, with column partials, one
/// group of rows a program, `BLOCK_SIZE // lanes` at a time, each partial summed per column across them;
/// each row's last-axis reductions, then its outputs, per element or per row. The first output is `out`
/// (`STORE`), output `k` after it `out{k}` (`STORE{k}`); `n` is the row count (the group count with
/// partials).
pub(crate) fn row(body: &Graph, by_value: &[bool]) -> Reduction {
    let root = rows_type(body);
    let n = width(body);
    let rows = root.numel() / n;
    let looped = row_looped(body);
    let lanes = if looped { 1 } else { n.next_power_of_two() };
    let mut emitter = Emitter::new(body, by_value);
    let mut combines = String::new();
    let mut code = String::new();
    let reductions = reductions(body);
    let mut k = 0;

    while k < reductions.len() {
        let node = reductions[k];
        let sum = matches!(node.primitive, Primitive::ReduceSum { .. });

        if !looped {
            whole(&mut emitter, k, node, &mut combines);
            k += 1;
            continue;
        }

        let next = reductions.get(k + 1).copied();

        if let Some(total) = next.filter(|s| softmax_pair(body, node, s)) {
            // Lane-wise running max and sum of exp(x - max), rescaled as the max grows.
            let (m, s) = (format!("m{k}"), format!("s{k}"));
            let a = body.type_of(node.output).dtype;

            let (ma, sa) = (
                triton_type(a).expect("a Triton dtype"),
                body.type_of(total.output).dtype,
            );

            let st = triton_type(sa).expect("a Triton dtype");

            writeln!(
                code,
                "    {m} = tl.full([1, BLOCK_SIZE], float(\"-inf\"), {ma})"
            )
            .unwrap();

            writeln!(code, "    {s} = tl.zeros([1, BLOCK_SIZE], {st})").unwrap();

            code += &chunks(&mut emitter, n, |e| {
                let (x, _, _) = operand(e, node);

                writeln!(
                    e.lines,
                    "    {m}_new = NAME_combine{k}({m}, tl.where(mask, {x}, float(\"-inf\")))"
                )
                .unwrap();

                e.reduced.insert(node.output, format!("{m}_new"));
                let (x, _, _) = operand(e, total);

                writeln!(
                    e.lines,
                    "    {s} = {s} * tl.where({m}_new == float(\"-inf\"), 1.0, libdevice.exp({m} - {m}_new)) + tl.where(mask & ({m}_new != float(\"-inf\")), {x}, 0.0)\n    {m} = {m}_new"
                )
                .unwrap();
            });

            combined(&mut combines, k, node, a);

            let peak = emitter.local(format!(
                "{}[:, None]",
                reduce_along(&m, 1, false, a, &format!("NAME_combine{k}"))
            ));

            let scaled =
                format!("tl.where({m} == float(\"-inf\"), 0.0, {s} * libdevice.exp({m} - {peak}))");

            let total_value = emitter.local(format!(
                "{}[:, None]",
                reduce_along(&scaled, 1, true, sa, "")
            ));

            code += &std::mem::take(&mut emitter.lines);
            emitter.reduced.insert(node.output, peak);
            emitter.reduced.insert(total.output, total_value);
            k += 2;
            continue;
        }

        // A pass reducing lane-wise partials, then the lanes.
        let a = body.type_of(node.output).dtype;
        let (acc, init) = (triton_type(a).expect("a Triton dtype"), identity(node, a));
        writeln!(code, "    acc{k} = tl.full([1, BLOCK_SIZE], {init}, {acc})").unwrap();

        code += &chunks(&mut emitter, n, |e| {
            let (x, _, _) = operand(e, node);

            writeln!(
                e.lines,
                "    acc{k} = NAME_combine{k}(acc{k}, tl.where(mask, {x}, tl.full([], {init}, {acc})))"
            )
            .unwrap();
        });

        combined(&mut combines, k, node, a);
        let call = reduce_along(&format!("acc{k}"), 1, sum, a, &format!("NAME_combine{k}"));
        let value = emitter.local(format!("{call}[:, None]"));
        code += &std::mem::take(&mut emitter.lines);
        emitter.reduced.insert(node.output, value);
        k += 1;
    }

    // Each output is stored per element of the rows, or once per row; partials once per group.
    let partials = column_partials(body);
    let partial = |v: Var| partials.iter().any(|p| p.output == v);
    let per_row = |v: Var| body.type_of(v).numel() == rows && !partial(v);

    let store = |k: usize| match k {
        0 => "STORE(out".to_string(),
        k => format!("STORE{k}(out{k}"),
    };

    let elements = |emitter: &mut Emitter| {
        for (k, &v) in body.outputs().iter().enumerate() {
            if !per_row(v) && !partial(v) {
                let value = emitter.value(v, "j".into());
                writeln!(emitter.lines, "    {}, j, {value}, mask)", store(k)).unwrap();
            }
        }
    };

    match looped {
        true => code += &chunks(&mut emitter, n, elements),
        false => {
            elements(&mut emitter);
            code += &std::mem::take(&mut emitter.lines);
        }
    }

    emitter.mask = "row_mask".into();

    for (k, &v) in body.outputs().iter().enumerate() {
        if per_row(v) {
            let value = emitter.value(v, "o".into());
            writeln!(emitter.lines, "    {}, o, {value}, row_mask)", store(k)).unwrap();
        }
    }

    // Each partial adds up its operand over the tile's rows, kept per column across the group.
    emitter.mask = "mask".into();
    let mut sums = String::new();
    let mut stores = String::new();

    for (p, node) in partials.iter().enumerate() {
        let k = body.outputs().iter().position(|&v| v == node.output);

        let reshape = body
            .nodes()
            .iter()
            .find(|r| r.output == node.inputs[0])
            .expect("a partial reads its rows reshaped into groups");

        let a = body.type_of(node.output).dtype;
        let acc = triton_type(a).expect("a Triton dtype");
        let x = emitter.value(reshape.inputs[0], "j".into());

        let x = match body.type_of(reshape.inputs[0]).dtype == a {
            true => x,
            false => format!("{x}.to({acc})"),
        };

        let total = reduce_along(
            &format!("tl.where(mask, {x}, tl.full([], 0, {acc}))"),
            0,
            true,
            a,
            "",
        );

        writeln!(sums, "    part{p} = tl.zeros([1, {lanes}], {acc})").unwrap();
        writeln!(emitter.lines, "    part{p} += {total}[None, :]").unwrap();

        writeln!(
            stores,
            "    {}, g * {n} + c, part{p}, c < {n})",
            store(k.expect("a partial is an output"))
        )
        .unwrap();
    }

    code += &emitter.lines;

    let extra = (1..body.outputs().len())
        .map(|k| format!("out{k}"))
        .collect::<Vec<_>>()
        .join(",\n    ");

    let params = params(
        by_value,
        (!extra.is_empty()).then_some(extra.as_str()),
        body.outputs().len(),
    );

    let each = rows / row_groups(body).unwrap_or(1);

    let tile = match looped {
        // One group of rows a program, `BLOCK_SIZE // lanes` of them at a time.
        false if !partials.is_empty() => {
            let rows: String = [
                format!("i = r + tl.arange(0, BLOCK_SIZE // {lanes})[:, None]"),
                format!("o = g * {each} + i"),
                format!("j = o * {n} + c"),
                format!("row_mask = i < {each}"),
                format!("mask = row_mask & (c < {n})"),
            ]
            .iter()
            .map(|l| format!("    {l}\n"))
            .collect();

            code = format!(
                "{sums}    for r in range(0, {each}, BLOCK_SIZE // {lanes}):\n{}{stores}",
                indented(&(rows + &code))
            );

            vec![
                "g = tl.program_id(0).to(tl.int64)".to_string(),
                format!("c = tl.arange(0, {lanes})[None, :]"),
            ]
        }
        true => vec![
            "o = tl.program_id(0).to(tl.int64) + tl.zeros([1, 1], tl.int64)".to_string(),
            "row_mask = o < n".into(),
        ],
        false => vec![
            format!(
                "o = (tl.program_id(0).to(tl.int64) * (BLOCK_SIZE // {lanes}) + tl.arange(0, BLOCK_SIZE // {lanes}))[:, None]"
            ),
            format!("c = tl.arange(0, {lanes})[None, :]"),
            format!("j = o * {n} + c"),
            "row_mask = o < n".into(),
            format!("mask = row_mask & (c < {n})"),
        ],
    };

    let tile: String = tile.iter().map(|l| format!("    {l}\n")).collect();

    let source = formatdoc!(
        "
        {combines}@triton.jit
        def NAME(
        {params}):
        {tile}{code}"
    );

    let (name, source) = named("row", source);

    Reduction {
        name,
        source,
        lanes,
        splits: 1,
    }
}

/// Triton row fusions: one generated kernel per body, a row (or a group of rows) a program.
impl RowCodegen for Triton {
    /// The kernel.
    type Code = Reduction;
    type Options = ();

    /// The row kernel of `body`.
    fn row(body: &Graph, by_value: &[bool], _: &()) -> Result<Reduction, String> {
        Ok(row(body, by_value))
    }
}
