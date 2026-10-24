use super::*;

pub(crate) fn attention_kernel(body: &Graph, a: &Attention) -> (String, String) {
    let n = body.inputs().len() - if a.dropout.is_some() { 2 } else { 0 };
    let input = |base: Var| {
        let k = body.inputs().iter().position(|&v| v == base);
        format!("in{}", k.expect("an input of the body"))
    };
    let index = |name: &str, acc: &Access| ix_method(name, acc, &a.batch);
    let ix = formatdoc!(
        "
        struct NAME_ix {{
            {}
            {}
            {}
            {}
        }};
        ",
        index("q", &a.q),
        index("k", &a.k),
        index("v", &a.v),
        index("o", &a.out)
    );
    let (t, o) = (metal_type(a.dtype), metal_type(a.out_dtype));
    let out = body.outputs()[0];
    let root = body.nodes()[a.root].output;
    let w = metal_type(body.type_of(out).dtype);
    let (epilogue, epi) = match out == root {
        true => (String::new(), String::new()),
        false => {
            let mut e = Emitter::new(body, &[]);
            e.invariant[root] = true;
            e.row_locals.insert(root, "r".into());
            let value = e.value(out, "j".into());
            let mut fields = String::new();
            for (k, &v) in body.inputs()[..n].iter().enumerate() {
                let vt = metal_type(body.type_of(v).dtype);
                writeln!(fields, "    device const {vt} *in{k};").unwrap();
            }
            let members: Vec<String> = (0..n).map(|k| format!("in{k}")).collect();
            let (hoisted, lines) = (&e.hoisted, &e.lines);
            (
                formatdoc!(
                    "
                    struct NAME_epi {{
                        {fields}inline {w} operator()({o} r, ulong i) const {{
                            uint j = uint(i);
                            {hoisted}{lines}return {value};
                        }}
                    }};
                    "
                ),
                format!(", NAME_epi{{{}}}", members.join(", ")),
            )
        }
    };
    let mut params: Vec<String> = body.inputs()[..n]
        .iter()
        .enumerate()
        .map(|(k, &v)| {
            let vt = metal_type(body.type_of(v).dtype);
            format!("device const {vt} *in{k} [[buffer({k})]]")
        })
        .collect();
    params.push(format!("device {w} *out [[buffer({n})]]"));
    let mut buffer = n;
    let raw = body.outputs()[1..].contains(&root);
    if raw {
        buffer += 1;
        params.push(format!("device {o} *raw [[buffer({buffer})]]"));
    }
    let lse = a.lse.is_some();
    if lse {
        buffer += 1;
        params.push(format!("device float *lse [[buffer({buffer})]]"));
    }
    let raw_arg = if raw { "raw" } else { "nullptr" };
    let lse_arg = if lse { "lse" } else { "nullptr" };
    let (drop, epi) = match &a.dropout {
        Some(d) => {
            params.extend(random_state(buffer + 1));
            let epi = if epi.is_empty() {
                ", Same()".to_owned()
            } else {
                epi
            };
            let drop = dropout_functor(d, a.dtype, a.sq, a.sk);
            (drop, format!("{epi}, NAME_drop{{seed, start}}"))
        }
        None => (String::new(), epi),
    };
    let (q, k, v) = (input(a.q.base), input(a.k.base), input(a.v.base));
    let score = score_functor(&a.scores);
    let (causal, offset) = (a.causal.is_some(), a.causal.unwrap_or(0));
    let (h, hv, sq, sk) = (a.h, a.hv, a.sq, a.sk);
    let call = match decodes(a) {
        true => formatdoc!(
            "
            threadgroup float maxima[ATTN_GROUPS], sums[ATTN_GROUPS], os[ATTN_GROUPS * {hv}];
            attention_decode<
                {t}, {o}, {h}u, {hv}u, {causal}, {raw}, {lse},
                NAME_ix, NAME_score, {w}
            >(
                {q}, {k}, {v}, out, {raw_arg}, {lse_arg},
                NAME_ix(), {sq}u, {sk}u, NAME_score(), {offset},
                maxima, sums, os,
                group.x, group.y, sg, lane, tid.y * 16 + tid.x{epi}
            );"
        ),
        false => {
            let bk = key_block(a);
            formatdoc!(
                "
                threadgroup {t} ks[{bk} * {h}], vs[{bk} * {hv}];
                flash_attention<
                    {t}, {o}, {h}u, {hv}u, {bk}u, {causal}, {raw}, {lse},
                    NAME_ix, NAME_score, {w}
                >(
                    {q}, {k}, {v}, out, {raw_arg}, {lse_arg},
                    NAME_ix(), {sq}u, {sk}u, NAME_score(), {offset},
                    ks, vs,
                    group.x, group.y, sg, lane, tid.y * 16 + tid.x{epi}
                );"
            )
        }
    };
    let params = params.join(",\n");
    named(formatdoc!(
        "
        {ix}{score}{epilogue}{drop}kernel void NAME(
            {params},
            {THREAD_ARGS}
        ) {{
            {call}
        }}
        "
    ))
}

pub(super) fn random_state(first: usize) -> [String; 2] {
    [
        format!("constant ulong &seed [[buffer({first})]]"),
        format!("constant ulong &start [[buffer({})]]", first + 1),
    ]
}

pub(super) fn dropout_functor(d: &Dropout, dtype: DType, sq: usize, sk: usize) -> String {
    let t = metal_type(dtype);
    let scale = constant(dtype, Scalar::Float(d.scale));
    let (offset, threshold) = (d.offset, d.threshold);
    formatdoc!(
        "
        struct NAME_drop {{
            ulong seed, start;
            inline {t} operator()({t} x, uint b, uint i, uint j) const {{
                ulong at = (ulong(b) * {sq}ul + i) * {sk}ul + j;
                bool dropped = philox_bits(seed, start + {offset}ul + at) < {threshold}u;
                return dropped ? {t}(0) : {t}(Mul::apply(x, {scale}));
            }}
        }};
        "
    )
}

pub(super) fn ix_method(name: &str, acc: &Access, batch: &[usize]) -> String {
    let up = name.to_uppercase();
    let mut code = formatdoc!(
        "
        static constant constexpr uint {up}_ROW = {}u, {up}_COL = {}u;
        inline ulong {name}(uint b) const {{
            ulong o = {}ul;
        ",
        acc.row,
        acc.col,
        acc.offset
    );
    if !batch.is_empty() {
        writeln!(code, "uint rest = b;").unwrap();
    }
    for (k, (&size, &(stride, div))) in batch.iter().zip(&acc.batch).enumerate().rev() {
        let last = k == 0;
        let idx = match last {
            true => "rest".to_string(),
            false => format!("(rest % {size}u)"),
        };
        if stride != 0 {
            let idx = match div {
                1 => idx,
                _ => format!("({idx} / {div}u)"),
            };
            writeln!(code, "        o += ulong({idx}) * {stride}ul;").unwrap();
        }
        if !last {
            writeln!(code, "        rest /= {size}u;").unwrap();
        }
    }
    writedoc!(
        code,
        "
            return o;
        }}"
    )
    .unwrap();
    code
}

pub(super) fn score_functor(scores: &[Score]) -> String {
    let mut score = "s".to_string();
    for step in scores {
        score = match *step {
            Score::Round(d) => format!("convert_value<{}>({score})", metal_type(d)),
            Score::Mul(c, d) => format!("Mul::apply({score}, {})", constant(d, Scalar::Float(c))),
        };
    }
    formatdoc!(
        "
        struct NAME_score {{
            inline float operator()(float s) const {{
                return {score};
            }}
        }};
        "
    )
}

pub(crate) fn attention_backward_kernel(body: &Graph, b: &Backward) -> (String, String) {
    let input = |base: Var| {
        let k = body.inputs().iter().position(|&v| v == base);
        format!("in{}", k.expect("an input of the body"))
    };
    let operands = [
        ("q", &b.q),
        ("k", &b.k),
        ("v", &b.v),
        ("g", &b.d_o),
        ("lse", &b.lse),
        ("delta", &b.delta),
    ];
    let outputs = [("dv", &b.dv_out), ("dk", &b.dk_out), ("dq", &b.dq_out)];
    let outputs = outputs
        .iter()
        .filter_map(|(name, out)| out.as_ref().map(|(_, acc)| (*name, acc)));
    let mut methods = String::new();
    for (name, acc) in operands.into_iter().chain(outputs) {
        writeln!(methods, "{}", ix_method(name, acc, &b.batch)).unwrap();
    }
    let ix = formatdoc!(
        "
        struct NAME_ix {{
            {methods}
        }};
        "
    );
    let score = score_functor(&b.scores);
    let t = metal_type(b.dtype);
    let n = body.inputs().len() - if b.dropout.is_some() { 2 } else { 0 };
    let mut params: Vec<String> = body.inputs()[..n]
        .iter()
        .enumerate()
        .map(|(k, &v)| {
            format!(
                "device const {} *in{k} [[buffer({k})]]",
                metal_type(body.type_of(v).dtype)
            )
        })
        .collect();
    let dkdv = b.dv_out.is_some();
    let dq = b.dq_out.is_some();
    let outs: &[&str] = match (dkdv, dq) {
        (true, true) => &["dv", "dk", "dq"],
        (true, false) => &["dv", "dk"],
        _ => &["dq"],
    };
    let stored = |out: &Option<(usize, Access)>| {
        let (node, _) = out.as_ref().expect("a computed gradient");
        metal_type(body.type_of(body.nodes()[*node].output).dtype)
    };
    for (k, out) in outs.iter().enumerate() {
        let ty = match *out {
            "dq" if dkdv => "atomic_float",
            "dq" => stored(&b.dq_out),
            "dk" => stored(&b.dk_out),
            _ => stored(&b.dv_out),
        };
        params.push(format!("device {ty} *{out} [[buffer({})]]", n + k));
    }
    let (drop, drop_arg) = match &b.dropout {
        Some(d) => {
            params.extend(random_state(n + outs.len()));
            let drop = dropout_functor(d, DType::F32, b.sq, b.sk);
            (drop, ", NAME_drop{seed, start}")
        }
        None => (String::new(), ""),
    };
    let (q, k, v, g) = (
        input(b.q.base),
        input(b.k.base),
        input(b.v.base),
        input(b.d_o.base),
    );
    let (lse, delta) = (input(b.lse.base), input(b.delta.base));
    let (causal, offset) = (b.causal.is_some(), b.causal.unwrap_or(0));
    let (h, hv, sq, sk) = (b.h, b.hv, b.sq, b.sk);
    let ds_scale = constant(DType::F32, Scalar::Float(b.ds_scale));
    let rows = match dkdv && dq {
        true => backward_block_with_dq(b).expect("a fused backward fits"),
        false => backward_block(b),
    };
    let (dk_t, dv_t, dq_t) = match (dkdv, dq) {
        (true, _) => (stored(&b.dk_out), stored(&b.dv_out), ""),
        (false, _) => ("", "", stored(&b.dq_out)),
    };
    let call = match (dkdv, dq) {
        (true, true) => formatdoc!(
            "
            threadgroup {t} qs[{rows} * {h}], gs[{rows} * {hv}];
            threadgroup {t} ks[ATTN_BQ * {h}], dss[ATTN_BQ * {rows}];
            threadgroup float lses[{rows}], deltas[{rows}];
            flash_attention_dkdv<
                {t}, {h}u, {hv}u, {rows}u, {causal}, true,
                NAME_ix, NAME_score, {dk_t}, {dv_t}
            >(
                {q}, {k}, {v}, {g}, {lse}, {delta},
                dk, dv, dq,
                NAME_ix(), {sq}u, {sk}u, {offset}, {ds_scale},
                qs, gs, lses, deltas, ks, dss,
                group.x, group.y, sg, lane, tid.y * 16 + tid.x,
                NAME_score(){drop_arg}
            );"
        ),
        (true, false) => formatdoc!(
            "
            threadgroup {t} qs[{rows} * {h}], gs[{rows} * {hv}];
            threadgroup float lses[{rows}], deltas[{rows}];
            flash_attention_dkdv<
                {t}, {h}u, {hv}u, {rows}u, {causal}, false,
                NAME_ix, NAME_score, {dk_t}, {dv_t}
            >(
                {q}, {k}, {v}, {g}, {lse}, {delta},
                dk, dv, nullptr,
                NAME_ix(), {sq}u, {sk}u, {offset}, {ds_scale},
                qs, gs, lses, deltas, qs, qs,
                group.x, group.y, sg, lane, tid.y * 16 + tid.x,
                NAME_score(){drop_arg}
            );"
        ),
        (false, _) => formatdoc!(
            "
            threadgroup {t} ks[{rows} * {h}], vs[{rows} * {hv}];
            flash_attention_dq<
                {t}, {h}u, {hv}u, {rows}u, {causal},
                NAME_ix, NAME_score, {dq_t}
            >(
                {q}, {k}, {v}, {g}, {lse}, {delta},
                dq,
                NAME_ix(), {sq}u, {sk}u, {offset}, {ds_scale},
                ks, vs,
                group.x, group.y, sg, lane, tid.y * 16 + tid.x,
                NAME_score(){drop_arg}
            );"
        ),
    };
    let params = params.join(",\n");
    named(formatdoc!(
        "
        {ix}{score}{drop}kernel void NAME(
            {params},
            {THREAD_ARGS}
        ) {{
            {call}
        }}
        "
    ))
}

pub(super) fn backward_block(b: &Backward) -> usize {
    let bytes = |n: usize| n * (b.h + b.hv) * b.dtype.size_of() + 2 * n * 4;
    [32, 16, 8]
        .into_iter()
        .find(|&n| bytes(n) <= 28 << 10)
        .unwrap_or(8)
}

pub(crate) fn backward_block_with_dq(b: &Backward) -> Option<usize> {
    const KEYS: usize = 64;
    let t = b.dtype.size_of();
    let bytes = |n: usize| n * (b.h + b.hv) * t + 2 * n * 4 + KEYS * b.h * t + KEYS * n * t;
    [32, 16].into_iter().find(|&n| bytes(n) <= 28 << 10)
}

pub(crate) fn decodes(a: &Attention) -> bool {
    a.sq <= 8
}

pub(super) fn key_block(a: &Attention) -> usize {
    let t = a.dtype.size_of();
    let bytes = |bk: usize| bk * (a.h + a.hv) * t;
    [32, 16, 8]
        .into_iter()
        .find(|&bk| bytes(bk) <= 28 << 10)
        .unwrap_or(8)
}
