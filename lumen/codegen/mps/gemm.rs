use super::*;

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
        rhs_contracting,
        ..
    } = &dot.primitive
    else {
        return None;
    };
    let shape = &body.type_of(dot.output).shape;
    let rhs_rank = body.type_of(dot.inputs[1]).shape.len();
    let (rank, n) = (shape.len(), *shape.last()?);
    let outputs = body.outputs();
    if !lhs_batch.is_empty()
        || rhs_rank != rhs_contracting.len() + 1
        || outputs.contains(&dot.output)
    {
        return None;
    }
    let along_last = |node: &Node, of: &[usize]| match &node.primitive {
        Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } => (0..of.len() - 1)
            .all(|d| start_indices[d] == 0 && limit_indices[d] == of[d])
            .then(|| (start_indices[of.len() - 1], limit_indices[of.len() - 1])),
        _ => None,
    };
    let group = |tupling: &'a Node, width: usize| -> Option<Region<'a>> {
        let Primitive::Reshape { new_sizes: tuples } = &tupling.primitive else {
            return None;
        };
        let k = *tuples.last()?;
        let fits = tuples.len() == rank + 1
            && tuples[..rank - 1] == shape[..rank - 1]
            && tuples[rank - 1] * k == width
            && matches!(k, 2 | 4 | 8)
            && !outputs.contains(&tupling.output);
        if !fits {
            return None;
        }
        let mut members: Vec<Option<&Node>> = vec![None; k];
        for member in readers_of(body, tupling.output) {
            let (first, last) = along_last(member, tuples)?;
            if last != first + 1 || members[first].is_some() || outputs.contains(&member.output) {
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
                let (start, limit) = along_last(top, shape)?;
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
    let one_pair = regions.len() == 1 && regions[0].k() == 2;
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

pub(crate) fn gemm_expanded(body: &Graph, dot: &Node) -> Option<[Var; 2]> {
    let out = body.outputs()[0];
    let node = body.nodes().iter().find(|n| n.output == out)?;
    let shape = &body.type_of(dot.output).shape;
    let tupled: Vec<usize> = shape.iter().copied().chain([1]).collect();
    match (&node.primitive, node.inputs.as_slice()) {
        (Primitive::Concatenate { dimension, .. }, &[a, b])
            if gemm_pair(body, dot).is_none()
                && ((*dimension + 1 == shape.len()
                    && body.type_of(a).shape == *shape
                    && body.type_of(b).shape == *shape)
                    || (*dimension == shape.len()
                        && body.type_of(a).shape == tupled
                        && body.type_of(b).shape == tupled)) =>
        {
            Some([a, b])
        }
        _ => None,
    }
}

pub(super) fn gemm_expanded_interleaved(body: &Graph, dot: &Node) -> bool {
    let out = body.outputs()[0];
    let rank = body.type_of(dot.output).shape.len();
    gemm_expanded(body, dot).is_some()
        && body
            .nodes()
            .iter()
            .find(|n| n.output == out)
            .is_some_and(|n| matches!(n.primitive, Primitive::Concatenate { dimension, .. } if dimension == rank))
}

pub(super) fn region_ends(body: &Graph, regions: &[Columns]) -> Vec<usize> {
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
            body.outputs()
                .iter()
                .position(|&v| reaches(v, &members))
                .expect("each region's value an output")
        })
        .collect()
}

pub(super) fn gemm_regions_kernel(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    regions: &[Columns],
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
    let mut store = String::new();
    for (c, &end) in regions.iter().zip(&ends) {
        let target = match end {
            0 => "out0".to_string(),
            k => format!("out{k}"),
        };
        let (offset, size, limit, k) = (c.offset, c.size(), c.offset + c.width, c.k());
        let element = |locals: &[(Var, String)], index: &str| {
            let mut e = Emitter::new(body, by_value);
            let mut declared = String::new();
            for (t, (v, read)) in locals.iter().enumerate() {
                e.invariant[*v] = true;
                e.row_locals.insert(*v, format!("r{t}"));
                writeln!(declared, "{o} r{t} = {read};").unwrap();
            }
            let value = e.value(outputs[end], "j".into());
            let (hoisted, lines) = (&e.hoisted, &e.lines);
            formatdoc!(
                "
                {{
                    uint j = uint({index});
                    {declared}{hoisted}{lines}{target}[j] = {value};
                }}
                "
            )
        };
        let read = |column: &str| format!("row[({column}) / 2][({column}) % 2]");
        let branch = match &c.region {
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
        };
        writedoc!(
            store,
            "
            if (n < {limit}) {{
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
            {fields}inline void store(ulong m, ulong n, thread const vec<{o}, 2> *row) const {{
                uint c = uint(n % {ROW_COLUMNS});
        {store}    }}
        }};
        "
    );
    for (suffix, bm, bn, bk) in [
        ("", 128, 64, "SG_BK"),
        ("_mid", 64, 64, "SG_BK"),
        ("_small", 32, 32, "SMALL_BK"),
    ] {
        writedoc!(
            source,
            "
            kernel void NAME{suffix}({params}constant ulong *p [[buffer({first})]],
                                     {THREAD_ARGS}) {{
                threadgroup {t} lt[SG_LT({bm}, {bk}, {t}, {a})], rt[{bk} * {bn}];
                matmul_sg_impl<{t}, {a}, {o}, {bm}, {bn}, {bk}, {w},
                               NAME_epi, false, true, false, false, true, true>(
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
    if let Some(regions) = gemm_regions(body, dot) {
        return gemm_regions_kernel(body, by_value, dot, &regions);
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
        Some([a, b]) => {
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
        (_, Some(_)) if gemm_expanded_interleaved(body, dot) => (
            format!("{o} r"),
            "NAME_epi, false, false, true, false, true",
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
    for (suffix, bm, bn, bk) in [
        ("", 128, 64, "SG_BK"),
        ("_mid", 64, 64, "SG_BK"),
        ("_small", 32, 32, "SMALL_BK"),
    ] {
        writedoc!(
            source,
            "
            kernel void NAME{suffix}({params}constant ulong *p [[buffer({first})]],
                                     {THREAD_ARGS}) {{
                threadgroup {t} lt[SG_LT({bm}, {bk}, {t}, {a})], rt[{bk} * {bn}];
                matmul_sg_impl<{t}, {a}, {o}, {bm}, {bn}, {bk}, {w},
                               {paired}>(
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
