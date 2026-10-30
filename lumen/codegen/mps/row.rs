//! Metal source for row kernels: bodies whose reductions run over the last axis (softmax, norms),
//! computed one row per threadgroup or SIMD group with values kept in registers between passes.
use super::*;
pub(crate) use crate::codegen::kinds::{row_partials, row_reductions};
use crate::graph_compiler::common::diamonds::{depends_on, softmax_pair};
use crate::kernels::dot_general::mps::GemmConfig;

/// Rows and row length of the first row reduction's operand.
pub(super) fn row_shape(body: &Graph) -> (usize, usize) {
    let reductions = row_reductions(body);
    let x = body.type_of(reductions.first().expect("a row reduction").inputs[0]);
    let n = x.shape.last().copied().unwrap_or(1).max(1);
    (x.numel() / n, n)
}

/// Threadgroups the row kernel launches: the partials' leading dim, one per `SIMD_ROWS` rows, or
/// one per row.
pub(crate) fn row_groups(body: &Graph) -> usize {
    let (rows, _) = row_shape(body);

    match row_partials(body).first() {
        Some(p) => body.type_of(p.output).shape[0],
        None if simd_rows(body) => rows.div_ceil(SIMD_ROWS),
        None => rows,
    }
}

/// Rows one threadgroup handles in the SIMD form, one per SIMD group.
pub(super) const SIMD_ROWS: usize = 8;

/// Longest row the SIMD form handles with one SIMD group per row.
pub(super) const SIMD_ROW: usize = 256;

/// Whether rows are short enough and reductions SIMD-reducible to give each row one SIMD group.
pub(crate) fn simd_rows(body: &Graph) -> bool {
    let (_, n) = row_shape(body);

    let simd = |r: &&Node| {
        matches!(
            body.type_of(r.output).dtype,
            DType::F32 | DType::F16 | DType::I32
        )
    };

    let reductions = row_reductions(body);
    n <= SIMD_ROW && reductions.iter().all(simd)
}

/// Generates the row kernel for `body`: one pass over each row per group of independent
/// reductions, then a final pass writing the outputs. `tile` carries the original body, dot and
/// GEMM config when the rows come from a fused matmul.
pub(super) fn row_kernel(
    body: &Graph,
    by_value: &[bool],
    reductions: &[&Node],
    config: &MpsConfig,
    tile: Option<(&Graph, &[bool], &Node, Option<GemmConfig>)>,
) -> (String, String) {
    let out = body.outputs()[0];
    let out_type = body.type_of(out);
    let (rows, n) = row_shape(body);
    let mut e = Emitter::new(body, by_value);

    // A value is per-row when it has one element per row and is never broadcast along the last axis.
    let per_row = |v: Var| {
        body.type_of(v).numel() == rows
            && body
                .nodes()
                .iter()
                .filter(|node| node.inputs.contains(&v))
                .all(|node| match &node.primitive {
                    Primitive::BroadcastInDim {
                        shape,
                        broadcast_dimensions,
                    } => !broadcast_dimensions
                        .iter()
                        .zip(&body.type_of(v).shape)
                        .any(|(&d, &size)| d + 1 == shape.len() && size > 1),
                    p => {
                        (fusion::elementwise(p) || matches!(p, Primitive::Reshape { .. }))
                            && body.type_of(node.output).numel() == rows
                    }
                })
    };

    // Mark values that are the same across a row (constants, per-row values and reduction results),
    // so the emitter hoists them out of the per-element loop.
    let mut constant = vec![false; body.types.len()];

    for &v in body.inputs() {
        constant[v] = body.type_of(v).numel() == 1;
        e.invariant[v] = constant[v] || per_row(v);
    }

    for node in body.nodes() {
        let v = node.output;
        let reduced = reductions.iter().any(|r| r.output == v);

        constant[v] = !reduced
            && !matches!(node.primitive, Primitive::Iota { .. })
            && node.inputs.iter().all(|&u| constant[u]);

        let reads = match node.primitive {
            Primitive::Gather { .. } => &node.inputs[1..],
            _ => &node.inputs[..],
        };

        let iota = matches!(node.primitive, Primitive::Iota { .. });

        let of_rows = body.type_of(v).numel() == rows
            && reads.iter().all(|&u| e.invariant[u])
            && (!iota || per_row(v));

        e.invariant[v] = reduced || constant[v] || of_rows;
    }

    let hoisted = |code: &str| match code.is_empty() {
        true => String::new(),
        false => formatdoc!(
            "
            // the same across the row: once a row
            {code}"
        ),
    };

    let reads = |v: Var, of: &[Var]| depends_on(body, v, of);
    // Group reductions into passes: a reduction joins the current pass unless it reads one of its
    // results; a max followed by its softmax sum becomes one online-softmax step.
    let mut passes: Vec<Vec<(usize, bool)>> = Vec::new();
    let mut k = 0;

    while k < reductions.len() {
        let online = config.online_softmax
            && k + 1 < reductions.len()
            && softmax_pair(body, reductions[k], reductions[k + 1]);

        let inside: Vec<Var> = passes.last().map_or(Vec::new(), |pass| {
            pass.iter()
                .flat_map(|&(m, pair)| (m..=m + usize::from(pair)).map(|m| reductions[m].output))
                .collect()
        });

        match passes.last_mut() {
            Some(pass) if !reads(reductions[k].inputs[0], &inside) => pass.push((k, online)),
            _ => passes.push(vec![(k, online)]),
        }

        k += 1 + usize::from(online);
    }

    let partials = row_partials(body);

    let summed: Vec<Var> = partials
        .iter()
        .map(|p| {
            let reshape = body.nodes().iter().find(|n| n.output == p.inputs[0]);
            reshape.expect("a reshape of the rows").inputs[0]
        })
        .collect();

    let roots: Vec<Vec<Var>> = passes
        .iter()
        .map(|pass| pass.iter().map(|&(k, _)| reductions[k].inputs[0]).collect())
        .chain([std::iter::once(out).chain(summed.iter().copied()).collect()])
        .collect();

    let simd = simd_rows(body);

    let lanes = match simd {
        true => 32,
        false => reduce::REDUCE_THREADS,
    };

    let per_thread = n.div_ceil(lanes);

    // Keep element values in registers across passes when each thread's share of the row fits.
    let mut kept = match per_thread <= config.row_cache {
        true => kept_values(body, &e.invariant, &roots),
        false => Vec::new(),
    };

    let wide = tile.is_some() && n > ROW_TILE;
    let tiles = n.div_ceil(ROW_TILE);

    // Wide matmul rows get their first pass from per-tile partials, so nothing is kept from it.
    if wide {
        kept.retain(|&(_, p)| p > 0);
    }

    let (mut source, mut decls) = (String::new(), String::new());

    for (m, &(v, _)) in kept.iter().enumerate() {
        let t = metal_type(body.type_of(v).dtype);
        writeln!(source, "    {t} kept{m}[{per_thread}];").unwrap();
    }

    let header = match kept.is_empty() && partials.is_empty() {
        true => formatdoc!(
            "
            for (uint c = t; c < {n}u; c += {lanes}u) {{
                uint j = row * {n}u + c;
            "
        ),
        false => {
            let bound = match n % lanes {
                0 => String::new(),
                _ => formatdoc!(
                    "
                    if (c >= {n}u) {{
                        break;
                    }}
                    "
                ),
            };

            formatdoc!(
                "
                _Pragma(\"clang loop unroll(full)\") for (uint e = 0; e < {per_thread}u; ++e) {{
                    uint c = t + e * {lanes}u;
                    {bound}uint j = row * {n}u + c;
                "
            )
        }
    };

    // Emits the values of pass `k`'s roots, reading values kept by earlier passes and storing those
    // that later passes reuse.
    let pass = |e: &mut Emitter, k: usize| {
        (e.lines, e.hoisted) = (String::new(), String::new());
        (e.values, e.indices) = (HashMap::new(), HashMap::new());

        for (m, &(v, _)) in kept.iter().enumerate().filter(|(_, (_, p))| *p < k) {
            e.values.insert((v, "j".into()), format!("kept{m}[e]"));
        }

        let values: Vec<String> = roots[k].iter().map(|&v| e.value(v, "j".into())).collect();
        let mut stores = String::new();

        for (m, &(v, _)) in kept.iter().enumerate().filter(|(_, (_, p))| *p == k) {
            let local = e.value(v, "j".into());
            writeln!(stores, "        kept{m}[e] = {local};").unwrap();
        }

        (values, stores)
    };

    for (p, members) in passes.iter().enumerate() {
        let from_tiles = wide && p == 0;

        let (values, stores) = match from_tiles {
            true => {
                (e.lines, e.hoisted) = (String::new(), String::new());
                let partial = |&(k, _): &(usize, bool)| format!("tiles{k}[row * {tiles}u + c]");
                (members.iter().map(partial).collect(), String::new())
            }
            false => pass(&mut e, p),
        };

        let (mut init, mut step, mut combine) = (String::new(), String::new(), String::new());
        let mut locals: Vec<(Var, String)> = Vec::new();

        for (&(k, online), value) in members.iter().zip(&values) {
            let r = reductions[k];
            let x = body.type_of(r.inputs[0]);

            assert_eq!(
                (x.shape.last(), body.type_of(r.output).numel()),
                (Some(&n), rows),
                "a reduction of rows"
            );

            let a = metal_type(body.type_of(r.output).dtype);

            // Online softmax: track the running max and the sum rescaled to it in one sweep.
            if online {
                let sum = reductions[k + 1];
                writeln!(init, "    {a} m{k} = -INFINITY, s{k} = 0;").unwrap();

                writedoc!(
                    step,
                    "
                    {a} x{k} = {value};
                    {a} n{k} = Max::apply(m{k}, x{k});
                    {a} e{k} = select(Exp::apply(min(m{k}, x{k}) - n{k}), {a}(0), n{k} == -INFINITY);
                    s{k} = select(s{k} + e{k}, s{k} * e{k} + {a}(1), x{k} > m{k});
                    m{k} = n{k};
                    "
                )
                .unwrap();

                let k1 = k + 1;

                match simd {
                    true => writedoc!(
                        combine,
                        "
                        {a} r{k} = simd_max(m{k});
                        // A lane with no elements has no sum to rescale.
                        {a} r{k1} = simd_sum(m{k} == -INFINITY ? {a}(0) : s{k} * Exp::apply(m{k} - r{k}));
                        "
                    ),
                    false => {
                        writeln!(
                            decls,
                            "    threadgroup {a} maxima{k}[REDUCE_THREADS], sums{k}[REDUCE_THREADS];"
                        )
                        .unwrap();

                        writedoc!(
                            combine,
                            "
                            maxima{k}[t] = m{k};
                            sums{k}[t] = s{k};
                            for (uint q = REDUCE_THREADS / 2; q > 0; q /= 2) {{
                                threadgroup_barrier(mem_flags::mem_threadgroup);
                                if (t < q) {{
                                    {a} m1 = maxima{k}[t], m2 = maxima{k}[t + q], mm = Max::apply(m1, m2);
                                    // A thread with no elements has no sum to rescale.
                                    {a} s1 = m1 == -INFINITY ? {a}(0) : sums{k}[t] * Exp::apply(m1 - mm);
                                    {a} s2 = m2 == -INFINITY ? {a}(0) : sums{k}[t + q] * Exp::apply(m2 - mm);
                                    maxima{k}[t] = mm;
                                    sums{k}[t] = s1 + s2;
                                }}
                            }}
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                            {a} r{k} = maxima{k}[0];
                            {a} r{k1} = sums{k}[0];
                            "
                        )
                    }
                }
                .unwrap();

                locals.push((r.output, format!("r{k}")));
                locals.push((sum.output, format!("r{}", k + 1)));
                continue;
            }

            let op = functor_of_reduction(&r.primitive);

            let value = match body.type_of(r.output).dtype == x.dtype {
                true => value.clone(),
                false => format!("{a}({value})"),
            };

            writeln!(init, "    {a} acc{k} = {op}::template identity<{a}>();").unwrap();
            writeln!(step, "        acc{k} = {op}::apply(acc{k}, {value});").unwrap();

            match simd {
                true => {
                    let lanes = match op {
                        "Add" => "simd_sum",
                        _ => "simd_max",
                    };

                    writeln!(combine, "    {a} r{k} = {lanes}(acc{k});")
                }
                false => {
                    writeln!(decls, "    threadgroup {a} shared{k}[REDUCE_THREADS];").unwrap();

                    writedoc!(
                        combine,
                        "
                        shared{k}[t] = acc{k};
                        for (uint s = REDUCE_THREADS / 2; s > 0; s /= 2) {{
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                            if (t < s) {{
                                shared{k}[t] = {op}::apply(shared{k}[t], shared{k}[t + s]);
                            }}
                        }}
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        {a} r{k} = shared{k}[0];
                        "
                    )
                }
            }
            .unwrap();

            locals.push((r.output, format!("r{k}")));
        }

        let (hoisted, lines) = (hoisted(&e.hoisted), &e.lines);

        let header = match from_tiles {
            true => format!("for (uint c = t; c < {tiles}u; c += {lanes}u) {{\n"),
            false => header.clone(),
        };

        writedoc!(
            source,
            "
            {hoisted}{init}{header}{lines}{step}{stores}}}
            {combine}"
        )
        .unwrap();

        e.row_locals.extend(locals);
    }

    let (values, _) = pass(&mut e, passes.len());
    let value = &values[0];
    let (mut once, mut each) = (String::new(), String::new());
    let (mut init, mut written) = (String::new(), String::new());

    // Column partial outputs accumulate per thread across the group's rows, then sum across SIMD
    // groups before the store.
    for (k, &v) in body.outputs()[1..].iter().enumerate() {
        if let Some(p) = partials.iter().position(|p| p.output == v) {
            let a = metal_type(body.type_of(v).dtype);
            let value = e.value(summed[p], "j".into());

            writedoc!(
                init,
                "
                {a} part{k}[{per_thread}];
                for (uint e = 0; e < {per_thread}u; ++e) {{
                    part{k}[e] = 0;
                }}
                "
            )
            .unwrap();

            writeln!(each, "        part{k}[e] += {a}({value});").unwrap();
            let k1 = k + 1;

            match simd {
                true => {
                    writeln!(decls, "    threadgroup {a} sums{k}[{SIMD_ROWS} * {n}];").unwrap();

                    writedoc!(
                        written,
                        "
                        for (uint e = 0; e < {per_thread}u; ++e) {{
                            uint c = t + e * 32u;
                            if (c < {n}u) {{
                                sums{k}[sg * {n}u + c] = part{k}[e];
                            }}
                        }}
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        for (uint c = tid.y * 16 + tid.x; c < {n}u; c += REDUCE_THREADS) {{
                            {a} sum = sums{k}[c];
                            for (uint s = 1; s < {SIMD_ROWS}u; ++s) {{
                                sum += sums{k}[s * {n}u + c];
                            }}
                            out{k1}[group.x * {n}u + c] = sum;
                        }}
                        "
                    )
                }
                false => writedoc!(
                    written,
                    "
                    for (uint e = 0; e < {per_thread}u; ++e) {{
                        uint c = t + e * REDUCE_THREADS;
                        if (c < {n}u) {{
                            out{k1}[group.x * {n}u + c] = part{k}[e];
                        }}
                    }}
                    "
                ),
            }
            .unwrap();

            continue;
        }

        let per_row = body.type_of(v).numel() == rows;
        let value = e.value(v, if per_row { "row" } else { "j" }.into());

        match (per_row, e.invariant[v]) {
            (true, true) => writedoc!(
                once,
                "
                if (t == 0) {{
                    out{}[row] = {value};
                }}
                ",
                k + 1
            ),
            (true, false) => writedoc!(
                each,
                "
                if (c == 0u) {{
                    out{}[row] = {value};
                }}
                ",
                k + 1
            ),
            (false, _) => writeln!(each, "        out{}[j] = {value};", k + 1),
        }
        .unwrap();
    }

    match out_type.numel() == rows && e.invariant[out] {
        true => {
            writedoc!(
                source,
                "
                {}if (t == 0) {{
                    out[row] = {value};
                }}
                {once}",
                hoisted(&e.hoisted)
            )
            .unwrap();

            if !each.is_empty() {
                writeln!(source, "{header}{}{each}}}", e.lines).unwrap();
            }
        }
        false => {
            let write = match out_type.numel() == rows {
                true => formatdoc!(
                    "
                    if (c == 0u) {{
                        out[row] = {};
                    }}",
                    e.value(out, "row".into())
                ),
                false => format!("out[j] = {value};"),
            };

            let (hoisted, lines) = (hoisted(&e.hoisted), &e.lines);

            writedoc!(
                source,
                "
                {hoisted}{once}{header}{lines}{write}
                {each}}}
                "
            )
            .unwrap()
        }
    }

    // Within a row `j % n` is just the column `c`.
    let source = source.replace(&format!("(j) % {n}u"), "c");

    let partials_of = |&(k, _): &(usize, bool)| {
        let a = metal_type(body.type_of(reductions[k].output).dtype);
        format!("device const {a} *tiles{k}")
    };

    let extra: Vec<String> = match wide {
        true => passes[0].iter().map(partials_of).collect(),
        false => Vec::new(),
    };

    let (params, _) = io_params(body, by_value, metal_type(out_type.dtype), &extra);
    let signature = format!("kernel void NAME({params}{THREAD_ARGS})");
    let staged = body.inputs().len() - 1;

    // A fused matmul either stages the whole row block in one kernel or, for wide rows, adds a
    // tiles kernel that the row kernel reads partials from.
    let prelude = match tile {
        Some((real, by_value, dot, _)) if !wide => {
            let source = read_staged(&source, staged, |x| format!("staged[({x}) - m0n]"));
            return gemm_rows(real, by_value, dot, &source, rows, n);
        }
        Some((real, by_value, dot, gemm)) => gemm_tiles(
            real, by_value, dot, body, reductions, &passes[0], rows, n, gemm,
        ),
        None => String::new(),
    };

    if simd && partials.is_empty() {
        return named(
            prelude.clone()
                + &formatdoc!(
                    "
            {signature} {{
                uint row = group.x * {SIMD_ROWS}u + sg, t = lane;
                if (row >= {rows}u) {{
                    return;
                }}
                {source}
            }}
            "
                ),
        );
    }

    if simd {
        let block = rows / row_groups(body);

        return named(prelude.clone() + &formatdoc!(
            "
            {signature} {{
                uint t = lane;
                {decls}{init}for (uint row = group.x * {block}u + sg; row < (group.x + 1) * {block}u; row += {SIMD_ROWS}u) {{
                    {source}
                }}
                {written}
            }}
            "
        ));
    }

    if partials.is_empty() {
        return named(
            prelude.clone()
                + &formatdoc!(
                    "
            {signature} {{
                uint row = group.x, t = tid.y * 16 + tid.x;
                {decls}{source}
            }}
            "
                ),
        );
    }

    let block = rows / row_groups(body);

    named(prelude + &formatdoc!(
        "
        {signature} {{
            uint t = tid.y * 16 + tid.x;
            {decls}{init}for (uint row = group.x * {block}u; row < (group.x + 1) * {block}u; ++row) {{
                {source}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            {written}
        }}
        "
    ))
}

/// Values that later passes recompute from an earlier pass's inputs, each with the first pass
/// that computes it, so they can be kept in registers instead.
pub(super) fn kept_values(
    body: &Graph,
    invariant: &[bool],
    roots: &[Vec<Var>],
) -> Vec<(Var, usize)> {
    let producer: HashMap<Var, &Node> = body.nodes().iter().map(|n| (n.output, n)).collect();

    let walk = |roots: &[Var], stop: &dyn Fn(Var) -> bool| {
        let (mut seen, mut stack) = (Vec::new(), roots.to_vec());

        while let Some(v) = stack.pop() {
            if invariant[v] || seen.contains(&v) {
                continue;
            }

            seen.push(v);

            if stop(v) {
                continue;
            }

            if let Some(node) = producer.get(&v)
                && fusion::elementwise(&node.primitive)
            {
                stack.extend(&node.inputs);
            }
        }

        seen
    };

    let mut first: HashMap<Var, usize> = HashMap::new();
    let mut kept: Vec<(Var, usize)> = Vec::new();

    for (k, root) in roots.iter().enumerate() {
        if k > 0 {
            for v in walk(root, &|v| first.contains_key(&v)) {
                if let Some(&p) = first.get(&v)
                    && !kept.iter().any(|&(u, _)| u == v)
                {
                    kept.push((v, p));
                }
            }
        }

        for v in walk(root, &|_| false) {
            first.entry(v).or_insert(k);
        }
    }

    kept
}

/// Metal functor that combines a reduction's values.
pub(super) fn functor_of_reduction(p: &Primitive) -> &'static str {
    match p {
        Primitive::ReduceSum { .. } => "Add",
        _ => "Max",
    }
}
