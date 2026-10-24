use super::*;
use crate::kernels::dot_general::mps::{Algorithm, DEFAULT_CONFIGS, GemmConfig, mpp_transposes};

pub(crate) fn gemm_dot(body: &Graph) -> Option<&Node> {
    body.nodes()
        .iter()
        .find(|n| matches!(n.primitive, Primitive::DotGeneral { .. }))
}

pub(crate) const ROW_COLUMNS: usize = 8;

pub(crate) enum Region<'a> {
    Group(Vec<&'a Node>),
    Plain(&'a Node),
}

pub(crate) struct Columns<'a> {
    pub(crate) axis: usize,
    pub(crate) offset: usize,
    pub(crate) width: usize,
    pub(crate) region: Region<'a>,
}

impl Columns<'_> {
    pub(crate) fn members(&self) -> Vec<Var> {
        match &self.region {
            Region::Group(members) => members.iter().map(|m| m.output).collect(),
            Region::Plain(s) => vec![s.output],
        }
    }

    pub(crate) fn k(&self) -> usize {
        match &self.region {
            Region::Group(members) => members.len(),
            Region::Plain(_) => 1,
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.width / self.k()
    }
}

pub(crate) fn gemm_regions<'a>(body: &'a Graph, dot: &Node) -> Option<Vec<Columns<'a>>> {
    let Primitive::DotGeneral {
        lhs_batch,
        lhs_contracting,
        rhs_contracting,
        ..
    } = &dot.primitive
    else {
        return None;
    };
    let rank = body.type_of(dot.output).shape.len();
    let one_free = |side: usize, contracting: &[usize]| {
        body.type_of(dot.inputs[side]).shape.len() == contracting.len() + 1
    };
    if !lhs_batch.is_empty() || body.outputs().contains(&dot.output) || rank == 0 {
        return None;
    }
    let columns = one_free(1, rhs_contracting)
        .then(|| regions_along(body, dot, rank - 1))
        .flatten();
    columns.or_else(|| {
        (one_free(0, lhs_contracting) && rank > 1)
            .then(|| regions_along(body, dot, 0).or_else(|| row_halves(body, dot)))
            .flatten()
    })
}

fn row_halves<'a>(body: &'a Graph, dot: &Node) -> Option<Vec<Columns<'a>>> {
    let shape = &body.type_of(dot.output).shape;
    let m = shape[0];
    let half = |node: &Node, start: usize| match &node.primitive {
        Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } => (0..shape.len()).all(|d| match d {
            0 => start_indices[d] == start && limit_indices[d] == start + m / 2,
            _ => start_indices[d] == 0 && limit_indices[d] == shape[d],
        }),
        _ => false,
    };
    let (a, b) = match readers_of(body, dot.output).as_slice() {
        [a, b] if half(a, 0) && half(b, m / 2) => (*a, *b),
        [a, b] if half(b, 0) && half(a, m / 2) => (*b, *a),
        _ => return None,
    };
    m.is_multiple_of(2).then(|| {
        vec![Columns {
            axis: 0,
            offset: 0,
            width: m,
            region: Region::Group(vec![a, b]),
        }]
    })
}

fn regions_along<'a>(body: &'a Graph, dot: &Node, axis: usize) -> Option<Vec<Columns<'a>>> {
    let shape = &body.type_of(dot.output).shape;
    let (rank, n) = (shape.len(), shape[axis]);
    let outputs = body.outputs();
    let along = |node: &Node, of: &[usize], axis: usize| match &node.primitive {
        Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } => (0..of.len())
            .all(|d| d == axis || (start_indices[d] == 0 && limit_indices[d] == of[d]))
            .then(|| (start_indices[axis], limit_indices[axis])),
        _ => None,
    };
    let group = |tupling: &'a Node, width: usize| -> Option<Region<'a>> {
        let Primitive::Reshape { new_sizes: tuples } = &tupling.primitive else {
            return None;
        };
        let k = *tuples.get(axis + 1)?;
        let fits = tuples.len() == rank + 1
            && tuples[..axis] == shape[..axis]
            && tuples[axis + 2..] == shape[axis + 1..]
            && tuples[axis] * k == width
            && matches!(k, 2 | 4 | 8)
            && !outputs.contains(&tupling.output);
        if !fits {
            return None;
        }
        let mut members: Vec<Option<&Node>> = vec![None; k];
        for member in readers_of(body, tupling.output) {
            let (first, last) = along(member, tuples, axis + 1)?;
            if last != first + 1 || members[first].is_some() {
                return None;
            }
            members[first] = Some(member);
        }
        Some(Region::Group(
            members.into_iter().collect::<Option<Vec<_>>>()?,
        ))
    };
    let mut regions: Vec<Columns> = Vec::new();
    for top in readers_of(body, dot.output) {
        let (start, width, region) = match &top.primitive {
            Primitive::Reshape { .. } => (0, n, group(top, n)?),
            _ => {
                let (start, limit) = along(top, shape, axis)?;
                let width = limit - start;
                let region = match readers_of(body, top.output).as_slice() {
                    [r] if matches!(r.primitive, Primitive::Reshape { .. }) => {
                        if outputs.contains(&top.output) {
                            return None;
                        }
                        group(r, width)?
                    }
                    _ => Region::Plain(top),
                };
                (start, width, region)
            }
        };
        if width % ROW_COLUMNS != 0 {
            return None;
        }
        regions.push(Columns {
            axis,
            offset: start,
            width,
            region,
        });
    }
    regions.sort_by_key(|c| c.offset);
    let covers = regions.first().is_some_and(|c| c.offset == 0)
        && regions
            .windows(2)
            .all(|w| w[0].offset + w[0].width == w[1].offset)
        && regions.last().is_some_and(|c| c.offset + c.width == n);
    let grouped = regions.iter().any(|c| c.k() > 1);
    let one_pair = axis + 1 == rank && regions.len() == 1 && regions[0].k() == 2;
    (covers && grouped && !one_pair).then_some(regions)
}

pub(crate) fn gemm_pair<'a>(body: &'a Graph, dot: &Node) -> Option<(&'a Node, &'a Node)> {
    let Primitive::DotGeneral {
        lhs_batch,
        rhs_contracting,
        ..
    } = &dot.primitive
    else {
        return None;
    };
    let shape = &body.type_of(dot.output).shape;
    let rhs_rank = body.type_of(dot.inputs[1]).shape.len();
    let (rank, n) = (shape.len(), *shape.last()?);
    if !lhs_batch.is_empty() || rhs_rank != rhs_contracting.len() + 1 || n % 2 != 0 {
        return None;
    }
    if let Some(pair) = gemm_interleaved(body, dot) {
        return Some(pair);
    }
    if gemm_regions(body, dot).is_some() {
        return None;
    }
    let readers = readers_of(body, dot.output);
    let half = |node: &&Node, start: usize| match &node.primitive {
        Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } => (0..rank).all(|d| match d == rank - 1 {
            true => start_indices[d] == start && limit_indices[d] == start + n / 2,
            false => start_indices[d] == 0 && limit_indices[d] == shape[d],
        }),
        _ => false,
    };
    match (readers.as_slice(), body.outputs().contains(&dot.output)) {
        ([a, b], false) => match (half(a, 0) && half(b, n / 2), half(b, 0) && half(a, n / 2)) {
            (true, _) => Some((a, b)),
            (_, true) => Some((b, a)),
            _ => None,
        },
        _ => None,
    }
}

pub(super) fn readers_of(body: &Graph, v: Var) -> Vec<&Node> {
    body.nodes()
        .iter()
        .filter(|node| node.inputs.contains(&v))
        .collect()
}

pub(crate) fn gemm_interleaved<'a>(body: &'a Graph, dot: &Node) -> Option<(&'a Node, &'a Node)> {
    let shape = &body.type_of(dot.output).shape;
    let (rank, n) = (shape.len(), *shape.last()?);
    let pairs: Vec<usize> = shape[..rank - 1]
        .iter()
        .copied()
        .chain([n / 2, 2])
        .collect();
    let outputs = body.outputs();
    let reshape = match (
        readers_of(body, dot.output).as_slice(),
        outputs.contains(&dot.output),
    ) {
        ([r], false) if matches!(&r.primitive, Primitive::Reshape { new_sizes } if *new_sizes == pairs) => {
            *r
        }
        _ => return None,
    };
    let one = |node: &&Node, start: usize| match &node.primitive {
        Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } => (0..=rank).all(|d| match d == rank {
            true => start_indices[d] == start && limit_indices[d] == start + 1,
            false => start_indices[d] == 0 && limit_indices[d] == pairs[d],
        }),
        _ => false,
    };
    match (
        readers_of(body, reshape.output).as_slice(),
        outputs.contains(&reshape.output),
    ) {
        ([a, b], false) => match (one(a, 0) && one(b, 1), one(b, 0) && one(a, 1)) {
            (true, _) => Some((a, b)),
            (_, true) => Some((b, a)),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn gemm_plain(body: &Graph, dot: &Node) -> bool {
    gemm_regions(body, dot).is_none()
        && gemm_pair(body, dot).is_none()
        && gemm_expanded(body, dot).is_none()
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Expansion {
    Columns,
    Pairs,
    Rows,
}

pub(crate) fn expansion(graph: &Graph, dot: &Node, concat: &Node) -> Option<Expansion> {
    let Primitive::Concatenate { dimension, .. } = concat.primitive else {
        return None;
    };
    let Primitive::DotGeneral {
        lhs_batch,
        lhs_contracting,
        ..
    } = &dot.primitive
    else {
        return None;
    };
    let shape = &graph.type_of(dot.output).shape;
    let rank = shape.len();
    let shaped = |of: &[usize]| {
        concat.inputs.len() == 2 && concat.inputs.iter().all(|&v| graph.type_of(v).shape == of)
    };
    let tupled = |at: usize| {
        let mut tupled = shape.clone();
        tupled.insert(at, 1);
        tupled
    };
    let rows = lhs_batch.is_empty()
        && rank >= 2
        && graph.type_of(dot.inputs[0]).shape.len() == lhs_contracting.len() + 1;
    match dimension {
        d if d + 1 == rank && shaped(shape) => Some(Expansion::Columns),
        d if d == rank && shaped(&tupled(rank)) => Some(Expansion::Pairs),
        1 if rows && shaped(&tupled(1)) => Some(Expansion::Columns),
        0 if rows && shaped(shape) => Some(Expansion::Rows),
        _ => None,
    }
}

pub(crate) fn gemm_expanded(body: &Graph, dot: &Node) -> Option<([Var; 2], Expansion)> {
    let out = body.outputs()[0];
    let concat = body.nodes().iter().find(|n| n.output == out)?;
    let layout = expansion(body, dot, concat)?;
    match concat.inputs.as_slice() {
        &[a, b] if gemm_pair(body, dot).is_none() => Some(([a, b], layout)),
        _ => None,
    }
}

pub(super) fn region_ends(body: &Graph, regions: &[Columns]) -> Vec<Vec<usize>> {
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

pub(super) fn gemm_regions_kernel(
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
    let interleaved =
        !matches!(&regions[0].region, Region::Group(m) if m[0].inputs[0] == dot.output);
    let mut store = String::new();
    for (c, ends) in regions.iter().zip(&ends) {
        let (offset, size, limit, k) = (c.offset, c.size(), c.offset + c.width, c.k());
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
    let (params, first) = io_params(body, by_value, w, &[]);
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
    for &(suffix, c) in variants {
        let (bm, bn, bk, threads, cols, simds) =
            (c.bm, c.bn, c.bk, c.threads, c.cols, c.threads / 32);
        let prefetch = c.prefetch;
        writedoc!(
            source,
            "
            [[max_total_threads_per_threadgroup({threads})]]
            kernel void NAME{suffix}({params}constant ulong *p [[buffer({first})]],
                                     {THREAD_ARGS}) {{
                threadgroup {t} lt[SG_LT_OF({bm}, {bk}, {t}, {a}, {simds})], rt[{bk} * {bn}];
                matmul_sg_impl<{t}, {a}, {o}, {bm}, {bn}, {bk}, {w},
                               NAME_epi, false, true, false, false, {interleaved}, true, {flag},
                               {threads}, {cols}, {prefetch}>(
                    {lhs}, {rhs}, out, p,
                    lt, rt,
                    group, tid.y * 16 + tid.x, sg, lane,
                    NAME_epi{{{members}}});
            }}
            "
        )
        .unwrap();
    }
    named(source)
}

pub(super) fn gemm_kernel(body: &Graph, by_value: &[bool], dot: &Node) -> (String, String) {
    gemm_kernel_with(body, by_value, dot, &DEFAULT_CONFIGS)
}

pub(crate) fn gemm_variant(
    body: &Graph,
    by_value: &[bool],
    config: GemmConfig,
) -> (String, String) {
    let dot = gemm_dot(body).expect("a matmul");
    gemm_kernel_with(body, by_value, dot, &[("", config)])
}

fn gemm_kernel_with(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    variants: &[(&str, GemmConfig)],
) -> (String, String) {
    if let Some(regions) = gemm_regions(body, dot) {
        return gemm_regions_kernel(body, by_value, dot, &regions, variants);
    }
    let Primitive::DotGeneral { accum_dtype, .. } = dot.primitive else {
        unreachable!("a dot")
    };
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
    let pair = gemm_pair(body, dot);
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
    let (params, first) = io_params(body, by_value, w, &[]);
    let members = members.join(", ");
    let (operands, paired) = match (pair, expanded) {
        (Some(_), _) if gemm_interleaved(body, dot).is_some() => (
            format!("{o} r1, {o} r2"),
            "NAME_epi, false, true, false, false, true",
        ),
        (Some(_), _) => (format!("{o} r1, {o} r2"), "NAME_epi, false, true"),
        (_, Some((_, Expansion::Pairs))) => (
            format!("{o} r"),
            "NAME_epi, false, false, true, false, true",
        ),
        (_, Some((_, Expansion::Rows))) => (
            format!("{o} r"),
            "NAME_epi, false, false, true, false, false, false, true",
        ),
        (_, Some(_)) => (format!("{o} r"), "NAME_epi, false, false, true"),
        _ => (format!("{o} r"), "NAME_epi"),
    };
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
    let plain = paired == "NAME_epi";
    let flags = 7 - paired.matches(',').count();
    let paired = format!("{paired}{}", ", false".repeat(flags));
    for &(suffix, c) in variants {
        let (bm, bn, bk, threads, cols, simds) =
            (c.bm, c.bn, c.bk, c.threads, c.cols, c.threads / 32);
        let prefetch = c.prefetch;
        assert!(
            plain || c.algorithm == Algorithm::Tiled,
            "only a plain epilogue runs on {:?}",
            c.algorithm
        );
        let call = match c.algorithm {
            Algorithm::Tiled => formatdoc!(
                "
                threadgroup {t} lt[SG_LT_OF({bm}, {bk}, {t}, {a}, {simds})], rt[{bk} * {bn}];
                matmul_sg_impl<{t}, {a}, {o}, {bm}, {bn}, {bk}, {w},
                {paired},
                {threads}, {cols}, {prefetch}>(
                {lhs}, {rhs}, out, p,
                lt, rt,
                group, tid.y * 16 + tid.x, sg, lane,
                NAME_epi{{{members}}});
                "
            ),
            Algorithm::Mpp => {
                let ty = |v: Var| body.type_of(v);
                let (left, right) = mpp_transposes(
                    &dot.primitive,
                    ty(dot.inputs[0]),
                    ty(dot.inputs[1]),
                    ty(dot.output),
                )
                .expect("a layout Metal Performance Primitives read");
                formatdoc!(
                    "
                    matmul_mpp<{t}, {o}, {bm}, {bn}, {bk}, {simds}, {left}, {right}, {w}, NAME_epi>(
                    {lhs}, {rhs}, out, p, group,
                    NAME_epi{{{members}}});
                    "
                )
            }
            Algorithm::Skinny => formatdoc!(
                "
                threadgroup float partials[{threads} / ({bn} / {bk}) * ({bn} + 1)];
                matmul_skinny<{t}, {o}, {bm}, {threads}, {bn}, {bk}, {w}, NAME_epi>(
                {lhs}, {rhs}, out, p, partials, group, tid.x,
                NAME_epi{{{members}}});
                "
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
