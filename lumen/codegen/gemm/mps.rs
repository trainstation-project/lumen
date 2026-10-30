//! Metal source for matmuls fused with elementwise epilogues, column regions and paired outputs.

use std::collections::HashMap;
use std::fmt::Write;

use indoc::{formatdoc, writedoc};

use super::GemmCodegen;
use crate::codegen::mps::*;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::MpsConfig;
use crate::kernels::dot_general::mps::{
    Algorithm, DEFAULT_CONFIGS, GemmConfig, gemv, gemv_default, gemv_rows, gemv_tiles_default,
    mpp_transposes, plan_matmul,
};
use crate::tensor::contiguous_strides;

pub(crate) use crate::codegen::kinds::gemm_dot;

pub(crate) use crate::codegen::regions::{
    Columns, Expansion, ROW_COLUMNS, Region, expansion, gemm_expanded, gemm_pair, gemm_regions,
};

/// Number of dot columns one epilogue call consumes, and whether the regions run along rows
/// instead of the last axis; `None` when the body has no column regions.
pub(crate) fn gemm_group(body: &Graph, dot: &Node) -> Option<(usize, bool)> {
    let rank = body.type_of(dot.output).shape.len();

    match gemm_pair(body, dot) {
        Some(_) => Some((2, false)),
        None => gemm_regions(body, dot).map(|r| (ROW_COLUMNS, r[0].axis + 1 != rank)),
    }
}

/// Whether the paired epilogue can run inside the MPP kernel, given the operand layouts it reads.
pub(crate) fn gemm_mpp_pair(body: &Graph, dot: &Node) -> bool {
    let ty = |v: Var| body.type_of(v);

    let layout = mpp_transposes(
        &dot.primitive,
        ty(dot.inputs[0]),
        ty(dot.inputs[1]),
        ty(dot.output),
    );

    match gemm_pair(body, dot) {
        Some((first, _)) if first.inputs[0] == dot.output => layout.is_some(),
        Some(_) => matches!(layout, Some((_, true))),
        None => false,
    }
}

/// Template flags describing how the GEMM epilogue consumes the accumulator tile.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct EpilogueFlags {
    /// The epilogue returns a `Pair` of values per element (the body ends in a two-way concat).
    pub(crate) expanded: bool,
    /// Paired values are interleaved in memory rather than stored as separate halves.
    pub(crate) interleaved: bool,
    /// Adjacent accumulator columns passed to one epilogue call.
    pub(crate) group: usize,
    /// The grouping or expansion runs along rows rather than columns.
    pub(crate) rows: bool,
}

impl EpilogueFlags {
    /// Whether the epilogue takes one value per element and writes one result.
    pub(crate) fn plain(&self) -> bool {
        self.group == 1 && !self.expanded
    }

    /// Flags as Metal template arguments `expanded, interleaved, group, rows`.
    fn template(&self) -> String {
        let EpilogueFlags {
            expanded,
            interleaved,
            group,
            rows,
        } = self;

        format!("{expanded}, {interleaved}, {group}, {rows}")
    }
}

/// Whether the first region's members don't read the dot output directly, so columns arrive
/// interleaved.
fn regions_interleaved(regions: &[Columns], dot: &Node) -> bool {
    !matches!(&regions[0].region, Region::Group(m) if m[0].inputs[0] == dot.output)
}

/// Epilogue flags for the GEMM over `dot` in `body`.
pub(crate) fn gemm_flags(body: &Graph, dot: &Node) -> EpilogueFlags {
    if let (Some(regions), Some((group, rows))) = (gemm_regions(body, dot), gemm_group(body, dot)) {
        return EpilogueFlags {
            interleaved: regions_interleaved(&regions, dot),
            group,
            rows,
            ..EpilogueFlags::default()
        };
    }

    let expansion = gemm_expanded(body, dot).map(|(_, e)| e);

    EpilogueFlags {
        expanded: expansion.is_some(),
        interleaved: expansion == Some(Expansion::Pairs),
        group: 1,
        rows: expansion == Some(Expansion::Rows),
    }
}

/// Metal call into the staged MPP, skinny or GEMV matmul for config `c`, passing the epilogue
/// functor built from `members`; `dual` runs a paired epilogue inside the MPP kernel.
#[allow(clippy::too_many_arguments)]
fn staged_call(
    body: &Graph,
    dot: &Node,
    c: GemmConfig,
    flags: EpilogueFlags,
    dual: bool,
    [t, a, o, w]: [&str; 4],
    [lhs, rhs]: [&str; 2],
    members: &str,
) -> String {
    let (bm, bn, bk, threads, simds) = (c.bm, c.bn, c.bk, c.threads, c.threads / 32);
    let template = flags.template();

    match c.algorithm {
        Algorithm::Mpp => {
            let ty = |v: Var| body.type_of(v);

            let (left, right) = mpp_transposes(
                &dot.primitive,
                ty(dot.inputs[0]),
                ty(dot.inputs[1]),
                ty(dot.output),
            )
            .expect("a layout Metal Performance Primitives read");

            match flags.plain() || dual {
                true => {
                    let (members_per_group, interleaved) = match dual {
                        true => (2, flags.interleaved),
                        false => (1, false),
                    };

                    formatdoc!(
                        "
                        matmul_mpp<{t}, {a}, {o}, {bm}, {bn}, {bk}, {simds}, {left}, {right}, {w}, NAME_epi, {members_per_group}, {interleaved}>(
                        {lhs}, {rhs}, out, p, group,
                        NAME_epi{{{members}}});
                        "
                    )
                }
                false => formatdoc!(
                    "
                    threadgroup float tile[{bm} * {bn}], stage[{simds} * 64];
                    matmul_mpp_staged<{t}, {a}, {o}, {bm}, {bn}, {bk}, {simds}, {left}, {right}, {w}, NAME_epi, {template}>(
                    {lhs}, {rhs}, out, p, group, sg, lane, tile, stage,
                    NAME_epi{{{members}}});
                    "
                ),
            }
        }
        Algorithm::Skinny => match flags.plain() {
            true => formatdoc!(
                "
                threadgroup {a} partials[{threads} / ({bn} / {bk}) * ({bn} + 1)];
                matmul_skinny<{t}, {a}, {o}, {bm}, {threads}, {bn}, {bk}, {w}, NAME_epi>(
                {lhs}, {rhs}, out, p, partials, group, tid.x,
                NAME_epi{{{members}}});
                "
            ),
            false => formatdoc!(
                "
                threadgroup {a} partials[{threads} / ({bn} / {bk}) * ({bn} + 1)];
                threadgroup float tile[8 * {bn}], stage[{simds} * 64];
                matmul_skinny<{t}, {a}, {o}, {bm}, {threads}, {bn}, {bk}, {w}, NAME_epi, {template}>(
                {lhs}, {rhs}, out, p, partials, group, tid.x,
                NAME_epi{{{members}}}, tile, stage, sg, lane);
                "
            ),
        },
        Algorithm::Gemv => {
            let (vector_lhs, vectors) = gemv_of(body, dot).expect("a matrix-vector dot");
            let rows = gemv_rows(&c, vector_lhs);

            match flags.plain() {
                true => formatdoc!(
                    "
                    matmul_gemv<{t}, {a}, {o}, {rows}, {simds}, {bk}, {vectors}, {vector_lhs}, {w}, NAME_epi>(
                    {lhs}, {rhs}, out, p, group, sg, lane,
                    NAME_epi{{{members}}});
                    "
                ),
                false => formatdoc!(
                    "
                    threadgroup float tile[8 * {rows}], stage[{simds} * 64];
                    matmul_gemv<{t}, {a}, {o}, {rows}, {simds}, {bk}, {vectors}, {vector_lhs}, {w}, NAME_epi, {template}>(
                    {lhs}, {rhs}, out, p, group, sg, lane,
                    NAME_epi{{{members}}}, tile, stage);
                    "
                ),
            }
        }
        Algorithm::Tiled => unreachable!("the tiled kernel stores from its fragments"),
    }
}

/// For a matrix-vector dot, whether the lhs is the vector and how many vectors there are.
pub(crate) fn gemv_of(body: &Graph, dot: &Node) -> Option<(bool, usize)> {
    let ty = |v: Var| body.type_of(v);
    let (lhs, rhs, out) = (ty(dot.inputs[0]), ty(dot.inputs[1]), ty(dot.output));
    let vector_lhs = gemv(&dot.primitive, lhs, rhs, out)?;

    let (ls, rs) = (
        contiguous_strides(&lhs.shape),
        contiguous_strides(&rhs.shape),
    );

    let launch = plan_matmul(&dot.primitive, lhs, rhs, out, &ls, &rs, "", None).ok()?;
    Some((vector_lhs, launch.p[usize::from(!vector_lhs)]))
}

/// For each column region, the body outputs that depend on any of its members.
pub(crate) fn region_ends(body: &Graph, regions: &[Columns]) -> Vec<Vec<usize>> {
    let producer: HashMap<Var, &Node> = body.nodes().iter().map(|n| (n.output, n)).collect();

    let reaches = |from: Var, to: &[Var]| {
        let mut stack = vec![from];
        let mut seen = Vec::new();

        while let Some(v) = stack.pop() {
            if to.contains(&v) {
                return true;
            }

            if seen.contains(&v) {
                continue;
            }

            seen.push(v);

            if let Some(n) = producer.get(&v) {
                stack.extend(&n.inputs);
            }
        }

        false
    };

    regions
        .iter()
        .map(|c| {
            let members = c.members();

            (0..body.outputs().len())
                .filter(|&k| reaches(body.outputs()[k], &members))
                .collect()
        })
        .collect()
}

/// Generates a GEMM whose epilogue splits the output columns (or rows) into regions, each
/// writing its own outputs; one kernel per entry of `variants`.
pub(crate) fn gemm_regions_kernel(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    regions: &[Columns],
    variants: &[(&str, GemmConfig)],
) -> (String, String) {
    let Primitive::DotGeneral { accum_dtype, .. } = dot.primitive else {
        unreachable!("a dot")
    };

    let outputs = body.outputs();

    let (t, a, o) = (
        metal_type(body.type_of(dot.inputs[0]).dtype),
        metal_type(accum_dtype),
        metal_type(body.type_of(dot.output).dtype),
    );

    let w = metal_type(body.type_of(outputs[0]).dtype);
    let ends = region_ends(body, regions);
    let shape = &body.type_of(dot.output).shape;
    let rows = regions[0].axis + 1 != shape.len();
    let columns: usize = shape[1..].iter().product();

    let (window, bound, flag) = match rows {
        true => ("window", "m", "true"),
        false => ("row", "n", "false"),
    };

    let interleaved = regions_interleaved(regions, dot);
    let mut store = String::new();

    for (c, ends) in regions.iter().zip(&ends) {
        let (offset, size, limit, k) = (c.offset, c.size(), c.offset + c.width, c.k());

        // Builds one epilogue element: binds the region's accumulator reads to locals and stores every
        // output the region reaches at index `index`.
        let element = |locals: &[(Var, String)], index: &str| {
            let mut e = Emitter::new(body, by_value);
            let mut declared = String::new();

            for (t, (v, read)) in locals.iter().enumerate() {
                e.invariant[*v] = true;
                e.row_locals.insert(*v, format!("r{t}"));
                writeln!(declared, "{o} r{t} = {read};").unwrap();
            }

            let mut stores = String::new();

            for &end in ends {
                let value = e.value(outputs[end], "j".into());
                writeln!(stores, "out{end}[j] = {value};").unwrap();
            }

            let (hoisted, lines) = (&e.hoisted, &e.lines);

            formatdoc!(
                "
                {{
                    uint j = uint({index});
                    {declared}{hoisted}{lines}{stores}}}
                "
            )
        };

        let branch = match rows {
            false => {
                // The epilogue receives two adjacent columns per call as `row[c / 2][c % 2]`.
                let read = |column: &str| format!("row[({column}) / 2][({column}) % 2]");

                match &c.region {
                    Region::Group(members) => {
                        let locals: Vec<(Var, String)> = members
                            .iter()
                            .enumerate()
                            .map(|(t, m)| (m.output, read(&format!("c + {t}"))))
                            .collect();

                        let one = element(&locals, &format!("m * {size} + (n - {offset}) / {k}"));
                        format!("if (c % {k} == 0) {one}")
                    }
                    Region::Plain(slice) => {
                        let first = element(
                            &[(slice.output, read("c"))],
                            &format!("m * {size} + (n - {offset})"),
                        );

                        let second = element(
                            &[(slice.output, read("c + 1"))],
                            &format!("m * {size} + (n - {offset}) + 1"),
                        );

                        format!("{first}{second}")
                    }
                }
            }
            true => {
                let members = c.members();

                let at = match k {
                    1 => format!("(m - {offset}) * {columns} + n"),
                    _ => format!("(m - {offset}) / {k} * {columns} + n"),
                };

                let column = |q: usize| {
                    let locals: Vec<(Var, String)> = members
                        .iter()
                        .enumerate()
                        .map(|(t, &v)| (v, format!("window[c + {t}][{q}]")))
                        .collect();

                    element(&locals, &format!("{at} + {q}"))
                };

                let (first, second) = (column(0), column(1));

                let both = formatdoc!(
                    "
                    {first}if (n + 1 < {columns}) {second}"
                );

                match k {
                    1 => both,
                    _ => format!("if (c % {k} == 0) {{\n{both}}}\n"),
                }
            }
        };

        writedoc!(
            store,
            "
            if ({bound} < {limit}) {{
                {branch}return;
            }}
            "
        )
        .unwrap();
    }

    let (mut fields, mut members) = (String::new(), Vec::new());

    for (k, &v) in body.inputs().iter().enumerate() {
        let vt = metal_type(body.type_of(v).dtype);

        match by_value.get(k) == Some(&true) {
            true => writeln!(fields, "    {vt} in{k};").unwrap(),
            false => writeln!(fields, "    device const {vt} *in{k};").unwrap(),
        }

        members.push(format!("in{k}"));
    }

    for (k, &v) in outputs[1..].iter().enumerate() {
        let vt = metal_type(body.type_of(v).dtype);
        writeln!(fields, "    device {vt} *out{};", k + 1).unwrap();
        members.push(format!("out{}", k + 1));
    }

    writeln!(fields, "    device {w} *out0;").unwrap();
    members.push("out".into());

    let operand = |v: Var| {
        let k = body.inputs().iter().position(|&i| i == v);
        format!("in{}", k.expect("a dot's operands are the fusion's inputs"))
    };

    let (lhs, rhs) = (operand(dot.inputs[0]), operand(dot.inputs[1]));
    let members = members.join(", ");

    let mut source = formatdoc!(
        "
        struct NAME_epi {{
            {fields}inline void store(ulong m, ulong n, thread const vec<{o}, 2> *{window}) const {{
                uint c = uint({bound} % {ROW_COLUMNS});
        {store}    }}
        }};
        "
    );

    let flags = gemm_flags(body, dot);

    for &(suffix, c) in variants {
        let (bm, bn, bk, threads, cols, simds) =
            (c.bm, c.bn, c.bk, c.threads, c.cols, c.threads / 32);

        let prefetch = c.prefetch;
        let (params, first) = io_params(body, by_value, w, &[]);

        let call = match c.algorithm {
            Algorithm::Tiled => formatdoc!(
                "
                threadgroup {t} lt[SG_LT_OF({bm}, {bk}, {t}, {a}, {simds})], rt[{bk} * {bn}];
                matmul_sg_impl<{t}, {a}, {o}, {bm}, {bn}, {bk}, {w},
                               NAME_epi, false, false, false, {interleaved}, {ROW_COLUMNS}, {flag},
                               {threads}, {cols}, {prefetch}>(
                    {lhs}, {rhs}, out, p,
                    lt, rt,
                    group, tid.y * 16 + tid.x, sg, lane,
                    NAME_epi{{{members}}});
                "
            ),
            _ => staged_call(
                body,
                dot,
                c,
                flags,
                false,
                [t, a, o, w],
                [&lhs, &rhs],
                &members,
            ),
        };

        writedoc!(
            source,
            "
            [[max_total_threads_per_threadgroup({threads})]]
            kernel void NAME{suffix}({params}constant ulong *p [[buffer({first})]],
                                     {THREAD_ARGS}) {{
            {call}}}
            "
        )
        .unwrap();
    }

    named(source)
}

/// Generates the fused GEMM for `body`, specialized to the default GEMV config when it applies,
/// else one variant per default config.
pub(crate) fn gemm_kernel(body: &Graph, by_value: &[bool], dot: &Node) -> (String, String) {
    let code = match gemv_config(body) {
        Some(c) => Mps::gemm(body, dot, by_value, &[("", c)]),
        None => Mps::gemm(body, dot, by_value, &DEFAULT_CONFIGS),
    };

    code.expect("a Metal GEMM of every GEMM body")
}

/// GEMV config for a matrix-vector body, or `None` when it isn't one (or its row-reduction form
/// needs a non-vector lhs).
pub(crate) fn gemv_config(body: &Graph) -> Option<GemmConfig> {
    if row_dot(body).is_some() && tile_scratch(body).is_none() {
        return None;
    }

    let (vector_lhs, _) = gemv_of(body, gemm_dot(body)?)?;

    match tile_scratch(body) {
        Some(_) => vector_lhs.then(gemv_tiles_default),
        None => Some(gemv_default(vector_lhs)),
    }
}

/// Generates the fused GEMM (or GEMM with row reductions) for one autotuning `config`.
pub(crate) fn gemm_variant(
    body: &Graph,
    by_value: &[bool],
    config: GemmConfig,
    mps: &MpsConfig,
) -> (String, String) {
    if tile_scratch(body).is_some() {
        let dot = row_dot(body).expect("a matmul with row reductions");
        return gemm_row_kernel(body, by_value, dot, mps, Some(config));
    }

    let dot = gemm_dot(body).expect("a matmul");
    Mps::gemm(body, dot, by_value, &[("", config)]).expect("a Metal GEMM of every GEMM body")
}

/// Metal GEMMs: one kernel a fusion, a variant a tile config, its epilogue applied as each tile is written
/// (in column regions or pairs when it reads the product's parts).
impl GemmCodegen for Mps {
    /// The kernel's name and Metal source.
    type Code = (String, String);
    /// The tile configs to generate a variant of the kernel for, by name suffix.
    type Options = [(&'static str, GemmConfig)];

    /// Generates the fused GEMM for `dot` with one kernel per `variants` entry, dispatching to the
    /// region kernel when the epilogue writes column regions.
    fn gemm(
        body: &Graph,
        dot: &Node,
        by_value: &[bool],
        variants: &[(&'static str, GemmConfig)],
    ) -> Result<(String, String), String> {
        let pair = gemm_pair(body, dot);

        if let Some(regions) = gemm_regions(body, dot).filter(|_| pair.is_none()) {
            return Ok(gemm_regions_kernel(body, by_value, dot, &regions, variants));
        }

        let Primitive::DotGeneral { accum_dtype, .. } = dot.primitive else {
            unreachable!("a dot")
        };

        // Epilogue outputs must have the dot's element count, or half or double it for paired outputs.
        let elements = body.type_of(dot.output).numel();

        let indexed =
            |v: &Var| [elements, elements / 2, 2 * elements].contains(&body.type_of(*v).numel());

        assert!(
            body.outputs().iter().all(indexed),
            "a contraction's epilogue writes values of its index space: {body}"
        );

        let out = body.outputs()[0];

        let (t, a) = (
            metal_type(body.type_of(dot.inputs[0]).dtype),
            metal_type(accum_dtype),
        );

        let (o, w) = (
            metal_type(body.type_of(dot.output).dtype),
            metal_type(body.type_of(out).dtype),
        );

        let mut e = Emitter::new(body, by_value);

        match pair {
            Some((first, second)) => {
                for (v, name) in [(first.output, "r1"), (second.output, "r2")] {
                    e.invariant[v] = true;
                    e.row_locals.insert(v, name.into());
                }
            }
            None => {
                e.invariant[dot.output] = true;
                e.row_locals.insert(dot.output, "r".into());
            }
        }

        let mut stores = String::new();

        for (k, &v) in body.outputs()[1..].iter().enumerate() {
            let value = e.value(v, "j".into());
            writeln!(stores, "        out{}[i] = {value};", k + 1).unwrap();
        }

        let expanded = gemm_expanded(body, dot);

        let value = match expanded {
            Some(([a, b], _)) => {
                let (va, vb) = (e.value(a, "j".into()), e.value(b, "j".into()));
                format!("Pair<{w}>{{{va}, {vb}}}")
            }
            None => e.value(out, "j".into()),
        };

        let (mut fields, mut members) = (String::new(), Vec::new());

        for (k, &v) in body.inputs().iter().enumerate() {
            let vt = metal_type(body.type_of(v).dtype);

            match by_value.get(k) == Some(&true) {
                true => writeln!(fields, "    {vt} in{k};").unwrap(),
                false => writeln!(fields, "    device const {vt} *in{k};").unwrap(),
            }

            members.push(format!("in{k}"));
        }

        for (k, &v) in body.outputs()[1..].iter().enumerate() {
            writeln!(
                fields,
                "    device {} *out{};",
                metal_type(body.type_of(v).dtype),
                k + 1
            )
            .unwrap();

            members.push(format!("out{}", k + 1));
        }

        let operand = |v: Var| {
            let k = body.inputs().iter().position(|&i| i == v);
            format!("in{}", k.expect("a dot's operands are the fusion's inputs"))
        };

        let (lhs, rhs) = (operand(dot.inputs[0]), operand(dot.inputs[1]));
        let members = members.join(", ");
        let layout = expanded.map(|(_, layout)| layout);

        let (operands, group, interleaved) = match pair {
            Some((first, _)) => (format!("{o} r1, {o} r2"), 2, first.inputs[0] != dot.output),
            None => (format!("{o} r"), 1, layout == Some(Expansion::Pairs)),
        };

        let rows = layout == Some(Expansion::Rows);
        let expanded_flag = expanded.is_some();

        let flags =
            format!("NAME_epi, false, {expanded_flag}, false, {interleaved}, {group}, {rows}");

        let returns = match expanded {
            Some(_) => format!("Pair<{w}>"),
            None => w.to_string(),
        };

        let (hoisted, lines) = (&e.hoisted, &e.lines);

        let mut source = formatdoc!(
            "
            struct NAME_epi {{
                {fields}inline {returns} operator()({operands}, ulong i) const {{
                    uint j = uint(i);
                    {hoisted}{lines}{stores}return {value};
                }}
            }};
            "
        );

        let mpp_pair = gemm_mpp_pair(body, dot);
        let epilogue = gemm_flags(body, dot);

        for &(suffix, c) in variants {
            let (bm, bn, bk, threads, cols, simds) =
                (c.bm, c.bn, c.bk, c.threads, c.cols, c.threads / 32);

            let prefetch = c.prefetch;
            let (params, first) = io_params(body, by_value, w, &[]);

            let call = match c.algorithm {
                Algorithm::Tiled => formatdoc!(
                    "
                    threadgroup {t} lt[SG_LT_OF({bm}, {bk}, {t}, {a}, {simds})], rt[{bk} * {bn}];
                    matmul_sg_impl<{t}, {a}, {o}, {bm}, {bn}, {bk}, {w},
                    {flags},
                    {threads}, {cols}, {prefetch}>(
                    {lhs}, {rhs}, out, p,
                    lt, rt,
                    group, tid.y * 16 + tid.x, sg, lane,
                    NAME_epi{{{members}}});
                    "
                ),
                _ => staged_call(
                    body,
                    dot,
                    c,
                    epilogue,
                    mpp_pair,
                    [t, a, o, w],
                    [&lhs, &rhs],
                    &members,
                ),
            };

            writedoc!(
                source,
                "
                [[max_total_threads_per_threadgroup({threads})]]
                kernel void NAME{suffix}({params}constant ulong *p [[buffer({first})]],
                                         {THREAD_ARGS}) {{
                {call}}}
                "
            )
            .unwrap();
        }

        Ok(named(source))
    }
}
