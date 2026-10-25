use super::*;
use crate::kernels::dot_general::mps::{Algorithm, GemmConfig, mpp_transposes};

pub(crate) const ROW_TILE: usize = 64;

pub(crate) fn row_dot(body: &Graph) -> Option<&Node> {
    let dot = gemm_dot(body)?;
    let Primitive::DotGeneral { lhs_batch, .. } = &dot.primitive else {
        return None;
    };
    let dots = body
        .nodes()
        .iter()
        .filter(|n| matches!(n.primitive, Primitive::DotGeneral { .. }))
        .count();
    if dots != 1 || !lhs_batch.is_empty() || row_reductions(body).is_empty() {
        return None;
    }
    let float = |v: Var| matches!(body.type_of(v).dtype, DType::F16 | DType::BF16 | DType::F32);
    let (rows, n) = row_shape(body);
    let rows_fit = match n <= ROW_TILE {
        true => simd_rows(body),
        false => tile_reductions(body).is_some(),
    };
    let fits = body.type_of(dot.output).shape == [rows, n]
        && rows_fit
        && row_partials(body).is_empty()
        && dot
            .inputs
            .iter()
            .all(|v| body.inputs().contains(v) && float(*v))
        && float(dot.output);
    fits.then_some(dot)
}

pub(super) fn tile_reductions(body: &Graph) -> Option<Vec<usize>> {
    let reductions = row_reductions(body);
    if reductions
        .windows(2)
        .any(|w| softmax_pair(body, w[0], w[1]))
    {
        return None;
    }
    let mut first: Vec<usize> = Vec::new();
    for (k, r) in reductions.iter().enumerate() {
        let earlier: Vec<Var> = first.iter().map(|&i| reductions[i].output).collect();
        if depends_on(body, r.inputs[0], &earlier) {
            break;
        }
        first.push(k);
    }
    let simd = |&k: &usize| {
        matches!(
            body.type_of(reductions[k].output).dtype,
            DType::F32 | DType::F16 | DType::I32
        )
    };
    first.iter().all(simd).then_some(first)
}

pub(crate) fn tile_scratch(body: &Graph) -> Option<(Vec<usize>, usize)> {
    let dot = row_dot(body)?;
    let (rows, n) = row_shape(body);
    if n <= ROW_TILE {
        return None;
    }
    let reductions = row_reductions(body);
    let tiles = n.div_ceil(ROW_TILE);
    let aligned = |bytes: usize| bytes.next_multiple_of(crate::graph_compiler::plan::ALIGNMENT);
    let mut offsets = vec![0];
    let mut at = aligned(rows * n * body.type_of(dot.output).dtype.size_of());
    for k in tile_reductions(body)? {
        offsets.push(at);
        at += aligned(rows * tiles * body.type_of(reductions[k].output).dtype.size_of());
    }
    Some((offsets, at))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_tiles(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    rows_body: &Graph,
    reductions: &[&Node],
    first: &[(usize, bool)],
    rows: usize,
    n: usize,
    gemm: Option<GemmConfig>,
) -> String {
    let Primitive::DotGeneral { accum_dtype, .. } = dot.primitive else {
        unreachable!("a dot")
    };
    let accum = metal_type(accum_dtype);
    let t = metal_type(body.type_of(dot.inputs[0]).dtype);
    let o = metal_type(body.type_of(dot.output).dtype);
    let out = metal_type(body.type_of(body.outputs()[0]).dtype);
    let operand = |v: Var| {
        let k = body.inputs().iter().position(|&i| i == v);
        format!("in{}", k.expect("a dot's operands are the fusion's inputs"))
    };
    let (lhs, rhs) = (operand(dot.inputs[0]), operand(dot.inputs[1]));
    let (tile, tiles) = (ROW_TILE, n.div_ceil(ROW_TILE));
    let mpp = gemm.filter(|c| c.algorithm == Algorithm::Mpp);
    let at = match mpp {
        Some(_) => format!("(uint(n0) + c0) / {tile}u"),
        None => "group.x".to_owned(),
    };
    let mut extra = vec![format!("device {o} *dotv")];
    let mut read = by_value.to_vec();
    read.resize(body.inputs().len(), false);
    read.push(false);
    let mut e = Emitter::new(rows_body, &read);
    let (mut init, mut step, mut write) = (String::new(), String::new(), String::new());
    for &(k, _) in first {
        let r = reductions[k];
        let a = metal_type(rows_body.type_of(r.output).dtype);
        let op = functor_of_reduction(&r.primitive);
        let value = e.value(r.inputs[0], "j".into());
        let lanes = match op {
            "Add" => "simd_sum",
            _ => "simd_max",
        };
        extra.push(format!("device {a} *tiles{k}"));
        writeln!(init, "{a} acc{k} = {op}::template identity<{a}>();").unwrap();
        writeln!(step, "acc{k} = {op}::apply(acc{k}, {a}({value}));").unwrap();
        writedoc!(
            write,
            "
            acc{k} = {lanes}(acc{k});
            if (lane == 0) {{
                tiles{k}[row * {tiles}u + {at}] = acc{k};
            }}
            "
        )
        .unwrap();
    }
    let staged = rows_body.inputs().len() - 1;
    let (params, first_arg) = io_params(body, by_value, out, &extra);
    if let Some(c) = mpp {
        let lines = read_staged(&format!("{}{}", e.hoisted, e.lines), staged, |x| {
            format!("dotv[{x}]")
        });
        let ty = |v: Var| body.type_of(v);
        let (left, right) = mpp_transposes(
            &dot.primitive,
            ty(dot.inputs[0]),
            ty(dot.inputs[1]),
            ty(dot.output),
        )
        .expect("a layout Metal Performance Primitives read");
        let (bm, bn, bk, threads, simds) = (c.bm, c.bn, c.bk, c.threads, c.threads / 32);
        return formatdoc!(
            "
            [[max_total_threads_per_threadgroup({threads})]]
            kernel void NAME_tiles({params}constant ulong *p [[buffer({first_arg})]],
                                   {THREAD_ARGS}) {{
                matmul_mpp<{t}, {accum}, {o}, {bm}, {bn}, {bk}, {simds}, {left}, {right}, {o}, Same>(
                    {lhs}, {rhs}, dotv, p, group);
                threadgroup_barrier(mem_flags::mem_device);
                const ulong m0 = ulong(group.y) * {bm}, n0 = ulong(group.x) * {bn};
                for (uint rr = sg; rr < {bm}u; rr += {simds}u) {{
                    uint row = uint(m0) + rr;
                    if (row >= {rows}u) {{
                        break;
                    }}
                    for (uint c0 = 0; c0 < {bn}u && n0 + c0 < {n}u; c0 += {tile}u) {{
                        {init}for (uint cc = lane; cc < {tile}u && n0 + c0 + cc < {n}u; cc += 32u) {{
                            uint j = row * {n}u + uint(n0) + c0 + cc;
                            {lines}{step}
                        }}
                        {write}
                    }}
                }}
            }}
            "
        );
    }
    let lines = read_staged(&format!("{}{}", e.hoisted, e.lines), staged, |x| {
        format!("staged[(ulong({x}) / {n}ul - m0) * {tile}ul + ulong({x}) % {n}ul - n0]")
    });
    formatdoc!(
        "
        struct NAME_tiles_staging {{
            threadgroup {o} *staged;
            device {o} *dotv;
            ulong M, N;
            // Each lane's elements of the accumulators (two of each 8x8
            // matrix), rounded to the dot's dtype: staged, and written to
            // the dot's value inside it.
            template <uint FM, uint FN>
            void tile(thread simdgroup_matrix<{accum}, 8, 8> (&c)[FM][FN],
                      ulong m0,
                      ulong n0,
                      uint sy,
                      uint sx,
                      uint flat,
                      uint lane) {{
                const uint fr = frag_row(lane), fc = frag_col(lane);
                UNROLL for (uint i = 0; i < FM; ++i) {{
                    UNROLL for (uint j = 0; j < FN; ++j) {{
                        const vec<{accum}, 2> e = frag(c[i][j]);
                        const uint r = (sy * FM + i) * 8 + fr, col = (sx * FN + j) * 8 + fc;
                        for (uint h = 0; h < 2; ++h) {{
                            const {o} v = {o}(e[h]);
                            staged[r * {tile}u + col + h] = v;
                            if (m0 + r < M && n0 + col + h < N) {{
                                dotv[(m0 + r) * N + n0 + col + h] = v;
                            }}
                        }}
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
        }};
        kernel void NAME_tiles({params}constant ulong *p [[buffer({first_arg})]],
                               {THREAD_ARGS}) {{
            threadgroup {t} lt[SG_LT({tile}, SG_BK, {t}, {accum})], rt[SG_BK * {tile}];
            threadgroup {o} staged[{tile} * {tile}];
            matmul_sg_impl<{t}, {accum}, {o}, {tile}, {tile}, SG_BK, {o},
                           NAME_tiles_staging, false, false, false, true>(
                {lhs}, {rhs}, (device {o} *)nullptr, p,
                lt, rt,
                group, tid.y * 16 + tid.x, sg, lane,
                NAME_tiles_staging{{staged, dotv, p[0], p[1]}});
            const ulong m0 = ulong(group.y) * {tile}, n0 = ulong(group.x) * {tile};
            for (uint rr = sg; rr < {tile}u; rr += {SIMD_ROWS}u) {{
                uint row = uint(m0) + rr;
                if (row >= {rows}u) {{
                    break;
                }}
                {init}for (uint cc = lane; cc < {tile}u && n0 + cc < {n}u; cc += 32u) {{
                    uint j = row * {n}u + uint(n0) + cc;
                    {lines}{step}
                }}
                {write}
            }}
        }}
        "
    )
}

pub(super) fn gemm_row_kernel(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    config: &MpsConfig,
    gemm: Option<GemmConfig>,
) -> (String, String) {
    let mut rows = Graph::new();
    let mut var: HashMap<Var, Var> = HashMap::new();
    for &v in body.inputs() {
        var.insert(v, rows.input(body.type_of(v).clone()));
    }
    var.insert(dot.output, rows.input(body.type_of(dot.output).clone()));
    for node in body.nodes().iter().filter(|n| n.output != dot.output) {
        rows.set_origin(node);
        let inputs: Vec<Var> = node.inputs.iter().map(|v| var[v]).collect();
        let v = rows.apply(node.primitive.clone(), &inputs);
        var.insert(node.output, v.expect("a node of the body"));
    }
    let outputs: Vec<Var> = body.outputs().iter().map(|v| var[v]).collect();
    rows.set_outputs(&outputs).expect("the body's outputs");
    let mut read = by_value.to_vec();
    read.resize(body.inputs().len(), false);
    read.push(false);
    let reductions = row_reductions(&rows);
    row_kernel(
        &rows,
        &read,
        &reductions,
        config,
        Some((body, by_value, dot, gemm)),
    )
}

pub(super) fn read_staged(source: &str, k: usize, staged: impl Fn(&str) -> String) -> String {
    let name = load(k);
    let (mut out, mut rest) = (String::new(), source);
    while let Some(at) = rest.find(&name) {
        let ident = rest[..at]
            .chars()
            .last()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        let after = &rest[at + name.len()..];
        let mut depth = 1;
        let end = after
            .char_indices()
            .find(|&(_, c)| {
                depth += match c {
                    '(' => 1,
                    ')' => -1,
                    _ => 0,
                };
                depth == 0
            })
            .map(|(i, _)| i)
            .expect("a closing parenthesis");
        out.push_str(&rest[..at]);
        match ident {
            true => out.push_str(&rest[at..at + name.len() + end + 1]),
            false => out.push_str(&staged(&after[..end])),
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    out
}

pub(super) fn gemm_rows(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    source: &str,
    rows: usize,
    n: usize,
) -> (String, String) {
    let Primitive::DotGeneral { accum_dtype, .. } = dot.primitive else {
        unreachable!("a dot")
    };
    let accum = metal_type(accum_dtype);
    let t = metal_type(body.type_of(dot.inputs[0]).dtype);
    let o = metal_type(body.type_of(dot.output).dtype);
    let out = metal_type(body.type_of(body.outputs()[0]).dtype);
    let operand = |v: Var| {
        let k = body.inputs().iter().position(|&i| i == v);
        format!("in{}", k.expect("a dot's operands are the fusion's inputs"))
    };
    let (lhs, rhs) = (operand(dot.inputs[0]), operand(dot.inputs[1]));
    let (params, first) = io_params(body, by_value, out, &[]);
    let tile = ROW_TILE;
    named(formatdoc!(
        "
        struct NAME_staging {{
            threadgroup {o} *staged;
            // Each lane's elements of the accumulators (two of each 8x8
            // matrix), rounded to the dot's dtype, at their place in the
            // tile; the columns past the rows' none.
            template <uint FM, uint FN>
            void tile(thread simdgroup_matrix<{accum}, 8, 8> (&c)[FM][FN],
                      ulong m0,
                      ulong n0,
                      uint sy,
                      uint sx,
                      uint flat,
                      uint lane) {{
                const uint fr = frag_row(lane), fc = frag_col(lane);
                UNROLL for (uint i = 0; i < FM; ++i) {{
                    UNROLL for (uint j = 0; j < FN; ++j) {{
                        const vec<{accum}, 2> e = frag(c[i][j]);
                        const uint r = (sy * FM + i) * 8 + fr, col = (sx * FN + j) * 8 + fc;
                        if (col < {n}u) {{
                            staged[r * {n}u + col] = {o}(e[0]);
                        }}
                        if (col + 1 < {n}u) {{
                            staged[r * {n}u + col + 1] = {o}(e[1]);
                        }}
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
        }};
        kernel void NAME({params}constant ulong *p [[buffer({first})]],
                         {THREAD_ARGS}) {{
            threadgroup {t} lt[SG_LT({tile}, SG_BK, {t}, {accum})], rt[SG_BK * {tile}];
            threadgroup {o} staged[{tile} * {n}];
            matmul_sg_impl<{t}, {accum}, {o}, {tile}, {tile}, SG_BK, {o},
                           NAME_staging, false, false, false, true>(
                {lhs}, {rhs}, (device {o} *)nullptr, p,
                lt, rt,
                group, tid.y * 16 + tid.x, sg, lane,
                NAME_staging{{staged}});
            const ulong m0n = ulong(group.y) * {tile} * {n};
            for (uint rr = sg; rr < {tile}u; rr += {SIMD_ROWS}u) {{
                uint row = group.y * {tile}u + rr, t = lane;
                if (row >= {rows}u) {{
                    break;
                }}
                {source}
            }}
        }}
        "
    ))
}
