//! Metal source for a matmul fused with row reductions over its output (e.g. a projection
//! followed by softmax or a norm), either in one tile or across column tiles.
use std::collections::HashMap;
use std::fmt::Write;

use indoc::{formatdoc, writedoc};

use super::codegen::gemv_of;
use crate::DType;
use crate::codegen::dot_general::mps::{Algorithm, GemmConfig, mpp_transposes};
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::mps::Emitter;
use crate::codegen::kinds::gemm_dot;
use crate::codegen::row::mps::*;
use crate::codegen::store::mps::load;
use crate::codegen::utils::mps::*;
use crate::graph::{Graph, Node, Primitive, TensorType, Var};
use crate::graph_compiler::CompilerConfig;

/// Output columns one threadgroup tile covers; wider rows are reduced across tiles.
pub(crate) const ROW_TILE: usize = 64;

/// Rows a threadgroup of the single-tile row matmul (`gemm_rows`) covers for `rows` rows, and its K
/// step: `ROW_TILE` rows with 16-wide K steps once that makes 512 threadgroups, else half as many with
/// 32-wide ones (room for which half the rows leave), twice the threadgroups each walking half the K
/// steps. On an M4 Pro, RMSNorm of [rows, 512] · [512, 64] takes 24 vs 15 µs at 8-512 rows with 64-row
/// blocks, and 395 vs 427 µs at 32768 rows (f32) with 32-row ones.
pub(crate) fn single_rows(rows: usize) -> (usize, &'static str) {
    match rows.div_ceil(ROW_TILE) >= 512 {
        true => (ROW_TILE, "SG_BK"),
        false => (ROW_TILE / 2, "SMALL_BK"),
    }
}

/// The single unbatched float matmul in a body whose row reductions can run over its output
/// rows, or `None` when the body doesn't fit that kernel.
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

/// Indices of the leading row reductions that can be partially reduced per tile (none reads an
/// earlier one's result), or `None` if a softmax pair or a non-SIMD-reducible dtype prevents it.
pub(crate) fn tile_reductions(body: &Graph) -> Option<Vec<usize>> {
    let reductions = row_reductions(body);
    let first = crate::graph_compiler::common::diamonds::tile_reductions(body, &reductions)?;

    let simd = |&k: &usize| {
        matches!(
            body.type_of(reductions[k].output).dtype,
            DType::F32 | DType::F16 | DType::I32
        )
    };

    first.iter().all(simd).then_some(first)
}

/// Outputs of a multi-tile row matmul's `body` (by output index) its partials pass writes: whole rows
/// that read no row reduction (a residual stream; CODA's rows scaled by a norm's weight, its `1 / rms`
/// moved past the next matmul), from the GEMM's accumulators. The combine writes the rest.
pub(crate) fn early_outputs(body: &Graph) -> Vec<usize> {
    let (rows, _) = row_shape(body);
    let reduced: Vec<Var> = row_reductions(body).iter().map(|r| r.output).collect();
    let summed: Vec<Var> = row_partials(body).iter().map(|p| p.output).collect();

    (0..body.outputs().len())
        .filter(|&k| {
            let v = body.outputs()[k];

            body.type_of(v).numel() != rows
                && !summed.contains(&v)
                && !crate::graph_compiler::common::diamonds::depends_on(body, v, &reduced)
        })
        .collect()
}

/// Where the combine of a multi-tile row matmul reads its rows from: the value of the partials pass
/// that cuts it from the dot most cheaply ([`cut`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cut {
    /// The combine's outputs follow from the tiles' partials alone: it reads no rows.
    Nothing,
    /// The dot's value, staged in scratch (`dotv`).
    Dot,
    /// Output `.0` (by index), which the partials pass writes anyway.
    Output(usize),
    /// Value `.0`, which the partials pass writes to scratch (`cutv`).
    Scratch(Var),
}

/// The cheapest cut, in bytes moved, between a multi-tile row matmul's partials pass and its combine,
/// in `body` whose matmul's value is `dot` (its node's output, or the staged input of the rows body):
/// each value the partials pass computes from the dot (before any tile reduction) that leaves the
/// combine needing neither the dot nor more of the rows costs its read, its write unless it is an output
/// written anyway, and the inputs the combine still reads (row-sized ones as a full read); the dot itself
/// a write and a read. Values an earlier pass computes are computed there, never again.
pub(crate) fn cut(body: &Graph, dot: Var) -> Cut {
    let (rows, n) = row_shape(body);
    let reductions = row_reductions(body);
    let reduced: Vec<Var> = reductions.iter().map(|r| r.output).collect();
    let early = early_outputs(body);
    let producer: HashMap<Var, &Node> = body.nodes().iter().map(|n| (n.output, n)).collect();

    let tiled: Vec<Var> = tile_reductions(body)
        .unwrap_or_default()
        .iter()
        .map(|&k| reduced[k])
        .collect();

    // What the combine computes: its outputs and the operands of the reductions it runs itself.
    let roots: Vec<Var> = (0..body.outputs().len())
        .filter(|k| !early.contains(k))
        .map(|k| body.outputs()[k])
        .chain(
            reductions
                .iter()
                .filter(|r| !tiled.contains(&r.output))
                .map(|r| r.inputs[0]),
        )
        .collect();

    let whole = rows * n;
    let bytes = |v: Var| body.type_of(v).numel().min(whole) * body.type_of(v).dtype.size_of();

    // The values the combine reaches with `stop` read from memory, not past the tiles' partials.
    let reached = |stop: Option<Var>| {
        let (mut stack, mut seen) = (roots.clone(), vec![false; body.types.len()]);
        let mut reached = Vec::new();

        while let Some(v) = stack.pop() {
            if std::mem::replace(&mut seen[v], true) || tiled.contains(&v) {
                continue;
            }

            reached.push(v);

            if Some(v) != stop {
                stack.extend(producer.get(&v).map_or(&[][..], |n| &n.inputs[..]));
            }
        }

        reached
    };

    let rows_of = |v: Var| body.type_of(v).numel() == whole;

    if !reached(None).into_iter().any(rows_of) {
        return Cut::Nothing;
    }

    let output = |v: Var| body.outputs().iter().position(|&o| o == v);

    // A candidate's cost: its read, its write unless written anyway, and the inputs left to read.
    let cost = |u: Var| -> Option<usize> {
        let reached = reached(Some(u));

        if u != dot && reached.contains(&dot) {
            return None;
        }

        let inputs: usize = reached
            .iter()
            .filter(|&&v| v != u && body.inputs().contains(&v))
            .map(|&v| bytes(v))
            .sum();

        let write = match output(u) {
            Some(_) if u != dot => 0,
            _ => bytes(u),
        };

        Some(bytes(u) + write + inputs)
    };

    let candidates = body
        .nodes()
        .iter()
        .map(|node| node.output)
        .filter(|&v| {
            rows_of(v)
                && crate::graph_compiler::common::diamonds::depends_on(body, v, &[dot])
                && !crate::graph_compiler::common::diamonds::depends_on(body, v, &reduced)
        })
        .chain([dot]);

    let best = candidates
        .filter_map(|u| cost(u).map(|c| (c, output(u).is_none(), u)))
        .min()
        .map(|(_, _, u)| u);

    match best {
        Some(u) if u == dot => Cut::Dot,
        Some(u) => match output(u) {
            Some(k) => Cut::Output(k),
            None => Cut::Scratch(u),
        },
        None => Cut::Dot,
    }
}

/// For a multi-tile row matmul, byte offsets into scratch of the dot value (first), each
/// tile-reduction's partials, then the [`Cut`] when the partials pass writes it there, and the total
/// scratch size.
pub(crate) fn tile_scratch(body: &Graph) -> Option<(Vec<usize>, usize)> {
    let dot = row_dot(body)?;
    let (rows, n) = row_shape(body);

    if n <= ROW_TILE {
        return None;
    }

    let reductions = row_reductions(body);
    let tiles = n.div_ceil(ROW_TILE);
    let aligned = |bytes: usize| bytes.next_multiple_of(crate::plan::ALIGNMENT);
    let mut offsets = vec![0];
    let mut at = aligned(rows * n * body.type_of(dot.output).dtype.size_of());

    for k in tile_reductions(body)? {
        offsets.push(at);
        at += aligned(rows * tiles * body.type_of(reductions[k].output).dtype.size_of());
    }

    // A cut the partials pass writes for the combine, after the partials.
    if let Cut::Scratch(u) = cut(body, dot.output) {
        offsets.push(at);
        at += aligned(rows * n * body.type_of(u).dtype.size_of());
    }

    Some((offsets, at))
}

/// A multi-tile row matmul's two launches, as the profiler shows them: the body inputs each reads (by
/// position), the outputs the partials pass writes, and the scratch values between them.
pub(crate) struct TilePasses {
    /// Inputs the partials pass reads: the dot's operands and what its tile reductions and early
    /// outputs read.
    pub(crate) partials: Vec<usize>,
    /// Inputs the combine pass reads: what its outputs read past the dot's value and the partials.
    pub(crate) combine: Vec<usize>,
    /// Outputs (by index) the partials pass writes ([`early_outputs`]); the combine writes the rest.
    pub(crate) early: Vec<usize>,
    /// The output (by index) the combine reads its rows from, when the [`Cut`] is one.
    pub(crate) read: Option<usize>,
    /// What the partials pass writes to scratch and the combine reads: the dot's value when the cut is
    /// the dot, each tile reduction's partials (`[rows, tiles]`), then the cut when it is kept there.
    pub(crate) scratch: Vec<TensorType>,
}

/// [`TilePasses`] of `body`, a multi-tile row matmul, or `None` when its rows fit one tile.
pub(crate) fn tile_passes(body: &Graph) -> Option<TilePasses> {
    let dot = row_dot(body)?;
    let (rows, n) = row_shape(body);

    if n <= ROW_TILE {
        return None;
    }

    let reductions = row_reductions(body);
    let tiled: Vec<Var> = tile_reductions(body)?
        .iter()
        .map(|&k| reductions[k].output)
        .collect();
    let tiles = n.div_ceil(ROW_TILE);
    let mut producer = vec![None; body.types.len()];

    for node in body.nodes() {
        producer[node.output] = Some(node);
    }

    // The inputs reached walking back from `roots`, not past the values in `stops` (read from scratch).
    let reads = |roots: Vec<Var>, stops: &[Var]| {
        let mut seen = vec![false; body.types.len()];
        let mut stack = roots;

        while let Some(v) = stack.pop() {
            if std::mem::replace(&mut seen[v], true) || stops.contains(&v) {
                continue;
            }

            if let Some(node) = producer[v] {
                stack.extend(&node.inputs);
            }
        }

        (0..body.inputs().len())
            .filter(|&i| seen[body.inputs()[i]])
            .collect::<Vec<_>>()
    };

    let mut stops = vec![dot.output];
    stops.extend(&tiled);

    let early = early_outputs(body);
    let late: Vec<Var> = (0..body.outputs().len())
        .filter(|k| !early.contains(k))
        .map(|k| body.outputs()[k])
        .collect();

    // The combine reads its rows from the cut, not past it.
    let cut = cut(body, dot.output);

    let (value, kept) = match cut {
        Cut::Dot => (Some(dot.output), None),
        Cut::Scratch(u) => (None, Some(u)),
        Cut::Output(k) => {
            stops.push(body.outputs()[k]);
            (None, None)
        }
        Cut::Nothing => (None, None),
    };

    stops.extend(kept);

    let rows_of = |v: Var| TensorType::new(body.type_of(v).dtype, &[rows, n]);

    let scratch = value
        .map(rows_of)
        .into_iter()
        .chain(
            tiled
                .iter()
                .map(|&v| TensorType::new(body.type_of(v).dtype, &[rows, tiles])),
        )
        .chain(kept.map(rows_of))
        .collect();

    let written = early.iter().map(|&k| body.outputs()[k]);

    Some(TilePasses {
        partials: reads(
            std::iter::once(dot.output)
                .chain(tiled.iter().copied())
                .chain(written)
                .chain(kept)
                .collect(),
            &[],
        ),
        combine: reads(late, &stops),
        early,
        read: match cut {
            Cut::Output(k) => Some(k),
            _ => None,
        },
        scratch,
    })
}

/// Generates `NAME_partials`, the first pass of a multi-tile row matmul: computes the dot, stores each
/// per-tile partial of the `first` reductions to `tiles{k}` and the outputs that read no reduction
/// ([`early_outputs`]), and the combine's [`Cut`] when it is the dot's value (`dotv`) or a value kept
/// in scratch (`cutv`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_tiles(
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
    // MPP and GEMV kernels tile by config, so the partial's tile index comes from the column.
    let mpp = gemm.filter(|c| c.algorithm != Algorithm::Tiled);

    let at = match mpp {
        Some(_) => format!("(uint(n0) + c0) / {tile}u"),
        None => "group.x".to_owned(),
    };

    let mut extra = vec![format!("device {o} *dotv")];
    // The rows body reads the dot value as one extra input after the original ones.
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

    // Outputs that read no reduction are written here, from the accumulators, not by the combine.
    for k in early_outputs(rows_body) {
        let value = e.value(rows_body.outputs()[k], "j".into());

        let name = match k {
            0 => "out".to_owned(),
            k => format!("out{k}"),
        };

        writeln!(step, "{name}[j] = {value};").unwrap();
    }

    let staged = rows_body.inputs().len() - 1;
    let cut = cut(rows_body, rows_body.inputs()[staged]);

    // A cut that is neither the dot nor an output is written to scratch here, for the combine.
    if let Cut::Scratch(u) = cut {
        let value = e.value(u, "j".into());
        let ty = metal_type(rows_body.type_of(u).dtype);
        extra.push(format!("device {ty} *cutv"));
        writeln!(step, "cutv[j] = {value};").unwrap();
    }

    // The combine reads the dot's value only when it is the cut.
    let kept = match cut == Cut::Dot {
        true => formatdoc!(
            "
            if (m0 + r < M && n0 + col + h < N) {{
                                            dotv[(m0 + r) * N + n0 + col + h] = v;
                                        }}"
        ),
        false => String::new(),
    };

    let (params, first_arg) = io_params(body, by_value, out, &extra);

    if let Some(c) = mpp {
        let lines = read_staged(&format!("{}{}", e.hoisted, e.lines), staged, |x| {
            format!("dotv[{x}]")
        });

        let (bm, bn, bk, threads, simds) = (c.bm, c.bn, c.bk, c.threads, c.threads / 32);

        let product = match c.algorithm {
            Algorithm::Gemv => {
                let (_, vectors) = gemv_of(body, dot).expect("a matrix-vector dot");

                format!(
                    "matmul_gemv<{t}, {accum}, {o}, {bn}, {simds}, {bk}, {vectors}, true, {o}, Same>(\n\
                     {lhs}, {rhs}, dotv, p, group, sg, lane);"
                )
            }
            _ => {
                let ty = |v: Var| body.type_of(v);

                let (left, right) = mpp_transposes(
                    &dot.primitive,
                    ty(dot.inputs[0]),
                    ty(dot.inputs[1]),
                    ty(dot.output),
                )
                .expect("a layout Metal Performance Primitives read");

                format!(
                    "matmul_mpp<{t}, {accum}, {o}, {bm}, {bn}, {bk}, {simds}, {left}, {right}, {o}, Same>(\n\
                     {lhs}, {rhs}, dotv, p, group);"
                )
            }
        };

        return formatdoc!(
            "
            [[max_total_threads_per_threadgroup({threads})]]
            kernel void NAME_partials({params}constant ulong *p [[buffer({first_arg})]],
                                   {THREAD_ARGS}) {{
                {product}
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
        struct NAME_partials_staging {{
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
                            {kept}
                        }}
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
        }};
        kernel void NAME_partials({params}constant ulong *p [[buffer({first_arg})]],
                               {THREAD_ARGS}) {{
            threadgroup {t} lt[SG_LT({tile}, SG_BK, {t}, {accum})], rt[SG_RT(SG_BK, {tile}, {t})];
            threadgroup {o} staged[{tile} * {tile}];
            matmul_sg_impl<{t}, {accum}, {o}, {tile}, {tile}, SG_BK, {o},
                           NAME_partials_staging, false, false, true>(
                {lhs}, {rhs}, (device {o} *)nullptr, p,
                lt, rt,
                group, tid.y * 16 + tid.x, sg, lane,
                NAME_partials_staging{{staged, dotv, p[0], p[1]}});
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

/// Generates the row kernel for a matmul fused with row reductions, by rebuilding the body with
/// the dot's value as an extra input and handing the dot to the row kernel.
pub(crate) fn gemm_row_kernel(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    config: &CompilerConfig,
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

/// Replaces each call to input `k`'s load helper in `source` with `staged(index)`, leaving
/// occurrences inside longer identifiers alone.
pub(crate) fn read_staged(source: &str, k: usize, staged: impl Fn(&str) -> String) -> String {
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

/// Generates a single-tile row matmul: each threadgroup stages a block of `single_rows` rows of the
/// dot (`n` wide) in threadgroup memory, then runs the row `source` over its rows.
pub(crate) fn gemm_rows(
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
    let (tile, (block, bk)) = (ROW_TILE, single_rows(rows));

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
            threadgroup {t} lt[SG_LT({block}, {bk}, {t}, {accum})], rt[SG_RT({bk}, {tile}, {t})];
            threadgroup {o} staged[{block} * {n}];
            matmul_sg_impl<{t}, {accum}, {o}, {block}, {tile}, {bk}, {o},
                           NAME_staging, false, false, true>(
                {lhs}, {rhs}, (device {o} *)nullptr, p,
                lt, rt,
                group, tid.y * 16 + tid.x, sg, lane,
                NAME_staging{{staged}});
            const ulong m0n = ulong(group.y) * {block} * {n};
            for (uint rr = sg; rr < {block}u; rr += {SIMD_ROWS}u) {{
                uint row = group.y * {block}u + rr, t = lane;
                if (row >= {rows}u) {{
                    break;
                }}
                {source}
            }}
        }}
        "
    ))
}
