//! Triton row kernels for normalization diamonds (softmax, RMSNorm, LayerNorm): each row's last-axis
//! reductions, then the rest of the row computed from those per-row values, in one kernel. Rows up to
//! `MOST_WHOLE_ROW` are held whole in registers and read once; wider rows are walked in chunks of
//! `BLOCK_SIZE`, a pass a reduction (a max and the sum of `exp(x - max)` sharing one, online) and a last
//! pass writing the outputs.

use indoc::writedoc;

use super::config::MOST_WHOLE_ROW;
use super::reduction::{Reduction, combine, lowest, reduce_along};
use super::*;
use crate::graph_compiler::elementwise;

/// The body's last-axis reductions, in order.
fn reductions(body: &Graph) -> Vec<&Node> {
    body.nodes()
        .iter()
        .filter(|n| {
            matches!(
                n.primitive,
                Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
            )
        })
        .collect()
}

/// Row width of a row fusion's `body`: its root's last dimension.
fn width(body: &Graph) -> usize {
    body.type_of(body.outputs()[0])
        .shape
        .last()
        .copied()
        .unwrap_or(1)
}

/// Whether a row fusion's `body` runs as one Triton row kernel: rows reduced along their last axis,
/// and per-row values (the reductions and what is computed from them alone) only reshaped, combined
/// with each other, or broadcast back across their rows.
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
    n > 1
}

/// Whether a row fusion's `body` is too wide to hold a row whole, so its kernel walks rows in chunks.
pub(crate) fn row_looped(body: &Graph) -> bool {
    width(body) > MOST_WHOLE_ROW
}

/// Whether `v` depends on value `on` in `body`.
fn depends(body: &Graph, v: Var, on: Var) -> bool {
    let producer: HashMap<Var, &Node> = body.nodes().iter().map(|n| (n.output, n)).collect();
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        if v == on {
            return true;
        }
        if let Some(node) = producer.get(&v) {
            stack.extend(&node.inputs);
        }
    }
    false
}

/// Whether reductions `max` then `sum` can share one online pass: `sum` reduces `exp(g - m)`, `m`
/// `max`'s value reshaped or broadcast, `g` independent of it (softmax's denominator).
fn online(body: &Graph, max: &Node, sum: &Node) -> bool {
    let producer: HashMap<Var, &Node> = body.nodes().iter().map(|n| (n.output, n)).collect();
    let (Primitive::ReduceMax { .. }, Primitive::ReduceSum { .. }) =
        (&max.primitive, &sum.primitive)
    else {
        return false;
    };
    let Some(exp) = producer
        .get(&sum.inputs[0])
        .filter(|n| n.primitive == Primitive::Exp)
    else {
        return false;
    };
    let Some(sub) = producer
        .get(&exp.inputs[0])
        .filter(|n| n.primitive == Primitive::Sub)
    else {
        return false;
    };
    let mut m = sub.inputs[1];
    while let Some(node) = producer.get(&m).filter(|n| {
        matches!(
            n.primitive,
            Primitive::Reshape { .. } | Primitive::BroadcastInDim { .. }
        )
    }) {
        m = node.inputs[0];
    }
    m == max.output && !depends(body, sub.inputs[0], max.output)
}

/// Value of reduction `node`'s operand at `j`, in its accumulation dtype; the dtype and its Triton type.
fn operand(emitter: &mut Emitter, node: &Node) -> (String, DType, &'static str) {
    let a = emitter.body.type_of(node.output).dtype;
    let acc = triton_type(a).expect("a Triton dtype");
    let x = emitter.value(node.inputs[0], "j".into());
    match emitter.body.type_of(node.inputs[0]).dtype == a {
        true => (x, a, acc),
        false => (format!("{x}.to({acc})"), a, acc),
    }
}

/// Identity of reduction `node` (0 or the lowest value) as a Triton literal.
fn identity(node: &Node, dtype: DType) -> String {
    match node.primitive {
        Primitive::ReduceSum { .. } => "0".into(),
        _ => lowest(dtype),
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
/// (`row_looped`) one row a program walked in chunks of `BLOCK_SIZE`; each row's last-axis reductions,
/// then its outputs, per element or per row. The first output is `out` (`STORE`), output `k` after it
/// `out{k}` (`STORE{k}`); `n` is the row count.
pub(crate) fn row(body: &Graph, by_value: &[bool]) -> Reduction {
    let root = body.type_of(body.outputs()[0]);
    let n = width(body);
    let rows = root.numel() / n;
    let looped = row_looped(body);
    let lanes = if looped { 1 } else { n.next_power_of_two() };
    let mut emitter = Emitter::new(body, by_value);
    let mut combines = String::new();
    let mut code = String::new();
    let combined = |combines: &mut String, k: usize, node: &Node, dtype: DType| {
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
    };
    let reductions = reductions(body);
    let mut k = 0;
    while k < reductions.len() {
        let node = reductions[k];
        let sum = matches!(node.primitive, Primitive::ReduceSum { .. });
        if !looped {
            let (x, a, acc) = operand(&mut emitter, node);
            let masked = format!(
                "tl.where(mask, {x}, tl.full([], {}, {acc}))",
                identity(node, a)
            );
            let call = reduce_along(&masked, 1, sum, a, &format!("NAME_combine{k}"));
            let value = emitter.local(format!("{call}[:, None]"));
            emitter.reduced.insert(node.output, value);
            if call.starts_with("tl.reduce") {
                combined(&mut combines, k, node, a);
            }
            k += 1;
            continue;
        }
        let next = reductions.get(k + 1).copied();
        if let Some(total) = next.filter(|s| online(body, node, s)) {
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
    // Each output is stored per element of the rows, or once per row.
    let per_row = |v: Var| body.type_of(v).numel() == rows;
    let store = |k: usize| match k {
        0 => "STORE(out".to_string(),
        k => format!("STORE{k}(out{k}"),
    };
    let elements = |emitter: &mut Emitter| {
        for (k, &v) in body.outputs().iter().enumerate() {
            if !per_row(v) {
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
    let tile = match looped {
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
