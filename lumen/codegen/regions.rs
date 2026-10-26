use crate::graph::{Graph, Node, Primitive, Var};

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
    let along = |axis: usize| regions_along(body, dot, axis).or_else(|| halves(body, dot, axis));
    let columns = one_free(1, rhs_contracting)
        .then(|| along(rank - 1))
        .flatten();
    columns.or_else(|| {
        (one_free(0, lhs_contracting) && rank > 1)
            .then(|| along(0))
            .flatten()
    })
}

fn halves<'a>(body: &'a Graph, dot: &Node, axis: usize) -> Option<Vec<Columns<'a>>> {
    let shape = &body.type_of(dot.output).shape;
    let n = shape[axis];
    let half = |node: &Node, start: usize| match &node.primitive {
        Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } => (0..shape.len()).all(|d| match d == axis {
            true => start_indices[d] == start && limit_indices[d] == start + n / 2,
            false => start_indices[d] == 0 && limit_indices[d] == shape[d],
        }),
        _ => false,
    };
    let (a, b) = match readers_of(body, dot.output).as_slice() {
        [a, b] if half(a, 0) && half(b, n / 2) => (*a, *b),
        [a, b] if half(b, 0) && half(a, n / 2) => (*b, *a),
        _ => return None,
    };
    n.is_multiple_of(2).then(|| {
        vec![Columns {
            axis,
            offset: 0,
            width: n,
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
    let returned = axis + 1 == rank
        && regions
            .iter()
            .any(|c| c.members().iter().any(|v| outputs.contains(v)));
    let one_pair = axis + 1 == rank && regions.len() == 1 && regions[0].k() == 2;
    let windows = one_pair || regions.iter().all(|c| c.width % ROW_COLUMNS == 0);
    (covers && (grouped || returned) && windows).then_some(regions)
}

pub(crate) fn gemm_pair<'a>(body: &'a Graph, dot: &Node) -> Option<(&'a Node, &'a Node)> {
    let rank = body.type_of(dot.output).shape.len();
    match gemm_regions(body, dot)?.as_slice() {
        [
            Columns {
                axis,
                region: Region::Group(members),
                ..
            },
        ] if axis + 1 == rank && members.len() == 2 => Some((members[0], members[1])),
        _ => None,
    }
}

pub(crate) fn readers_of(body: &Graph, v: Var) -> Vec<&Node> {
    body.nodes()
        .iter()
        .filter(|node| node.inputs.contains(&v))
        .collect()
}
