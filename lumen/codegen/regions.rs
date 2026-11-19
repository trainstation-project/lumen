//! Finds how a GEMM's output is split into column regions by its readers so each region's epilogue can fuse into the GEMM.
use crate::graph::{Graph, Node, Primitive, Var};

/// Column granularity a region's width must be a multiple of so row kernels can window it.
pub(crate) const ROW_COLUMNS: usize = 8;

/// The readers covering one column range of a GEMM output.
pub(crate) enum Region<'a> {
    /// Interleaved members of a `reshape` that splits the axis into tuples of `k`, ordered by tuple index;
    /// also used for the two contiguous halves of the output.
    Group(Vec<&'a Node>),
    /// A single slice of the output read as is.
    Plain(&'a Node),
}

/// A column range `offset..offset + width` along `axis` of a GEMM output and the region reading it.
pub(crate) struct Columns<'a> {
    pub(crate) axis: usize,
    pub(crate) offset: usize,
    pub(crate) width: usize,
    pub(crate) region: Region<'a>,
}

impl Columns<'_> {
    /// Returns the variables each member of the region produces.
    pub(crate) fn members(&self) -> Vec<Var> {
        match &self.region {
            Region::Group(members) => members.iter().map(|m| m.output).collect(),
            Region::Plain(s) => vec![s.output],
        }
    }

    /// Returns how many members share the range: the tuple size for a group, 1 for a plain slice.
    pub(crate) fn k(&self) -> usize {
        match &self.region {
            Region::Group(members) => members.len(),
            Region::Plain(_) => 1,
        }
    }

    /// Returns the width each member covers.
    pub(crate) fn size(&self) -> usize {
        self.width / self.k()
    }
}

/// Splits the output of `dot` into regions along its last axis (or its first, when the lhs has one free dim),
/// or `None` when the readers don't tile the output in a fusable way.
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

    // batched dots, dots that are graph outputs and scalars never fuse epilogues
    if !lhs_batch.is_empty() || body.outputs().contains(&dot.output) || rank == 0 {
        return None;
    }

    // prefer the column axis when the rhs contributes exactly one free dimension, otherwise try the row axis
    let along = |axis: usize| {
        regions_along(body, dot.output, axis, false).or_else(|| halves(body, dot.output, axis))
    };

    let columns = one_free(1, rhs_contracting)
        .then(|| along(rank - 1))
        .flatten();

    columns.or_else(|| {
        (one_free(0, lhs_contracting) && rank > 1)
            .then(|| along(0))
            .flatten()
    })
}

/// The column regions of a split-K combine's `value` (`combine_value`): as a GEMM's product's
/// (`regions_along`), along its last axis, without the gated pair's two halves. Readers of the whole value
/// (prefill attention reading a merged projection's values as a view of it) may remain: the epilogue then
/// stores the value too.
pub(crate) fn combine_regions(body: &Graph, value: Var) -> Option<Vec<Columns<'_>>> {
    let rank = body.type_of(value).shape.len();
    (rank > 0).then(|| regions_along(body, value, rank - 1, true))?
}

/// The value a split-K combine (`reduce`, a sum over its chunks) hands its epilogue as a GEMM hands its
/// product: the rounding cast reading it, when that is its only reader, else the sum itself.
pub(crate) fn combine_value(body: &Graph, reduce: &Node) -> Var {
    match readers_of(body, reduce.output).as_slice() {
        [cast]
            if matches!(cast.primitive, Primitive::Cast { .. })
                && body.type_of(cast.output).dtype.is_float()
                && body.type_of(cast.output).dtype.size_of()
                    < body.type_of(reduce.output).dtype.size_of()
                && !body.outputs().contains(&reduce.output) =>
        {
            cast.output
        }
        _ => reduce.output,
    }
}

/// Matches exactly two readers that slice the first and second halves of `product` along `axis`.
fn halves<'a>(body: &'a Graph, product: Var, axis: usize) -> Option<Vec<Columns<'a>>> {
    let shape = &body.type_of(product).shape;
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

    let (a, b) = match readers_of(body, product).as_slice() {
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

/// Collects the readers of `product` that slice disjoint ranges along `axis`, each possibly reshaped into
/// interleaved `k`-tuples, and returns them when they cover the axis and some region is worth fusing. With
/// `whole`, readers of the whole product (no such slice) are left to read it as stored.
fn regions_along<'a>(
    body: &'a Graph,
    product: Var,
    axis: usize,
    whole: bool,
) -> Option<Vec<Columns<'a>>> {
    let shape = &body.type_of(product).shape;
    let (rank, n) = (shape.len(), shape[axis]);
    let outputs = body.outputs();

    // a slice that keeps every dim but `axis` whole yields its `(start, limit)` on `axis`
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

    // a reshape of `width` columns into `(width / k, k)` whose readers each pick one tuple index
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

    for top in readers_of(body, product) {
        let (start, width, region) = match &top.primitive {
            Primitive::Reshape { .. } => match group(top, n) {
                Some(group) => (0, n, group),
                None if whole => continue,
                None => return None,
            },
            _ => {
                let (start, limit) = match along(top, shape, axis) {
                    Some(range) => range,
                    None if whole => continue,
                    None => return None,
                };
                let width = limit - start;

                // A slice reshaped into tuples is a group of interleaved members; reshaped otherwise (a
                // projection's heads), the slice is a plain region the reshape follows in its epilogue.
                let region = match readers_of(body, top.output).as_slice() {
                    [r] if matches!(r.primitive, Primitive::Reshape { .. }) => match group(r, width) {
                        Some(_) if outputs.contains(&top.output) => return None,
                        Some(group) => group,
                        None => Region::Plain(top),
                    },
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

    // regions must tile `0..n` exactly with no gaps or overlaps
    regions.sort_by_key(|c| c.offset);

    let covers = regions.first().is_some_and(|c| c.offset == 0)
        && regions
            .windows(2)
            .all(|w| w[0].offset + w[0].width == w[1].offset)
        && regions.last().is_some_and(|c| c.offset + c.width == n);

    // only worth a split when some region interleaves members, writes a graph output directly, or the output
    // splits into several regions (merged projections) some of which have an epilogue to run (a rotary
    // embedding); two halves are a gated pair, whose epilogue reads both (`halves`), and regions read only by
    // other kernels (attention reading its projections as views of one buffer) stay unsplit
    let several = regions.len() > 1
        && halves(body, product, axis).is_none()
        && regions
            .iter()
            .any(|c| c.members().iter().any(|&v| epilogue_read(body, v)));
    let grouped = regions.iter().any(|c| c.k() > 1) || several;

    let returned = axis + 1 == rank
        && regions
            .iter()
            .any(|c| c.members().iter().any(|v| outputs.contains(v)));

    let one_pair = axis + 1 == rank && regions.len() == 1 && regions[0].k() == 2;
    let windows = one_pair || regions.iter().all(|c| c.width % ROW_COLUMNS == 0);
    (covers && (grouped || returned) && windows).then_some(regions)
}

/// Whether an elementwise op or a pair swap reads `v`, directly or through reshapes: work a GEMM epilogue would run.
fn epilogue_read(body: &Graph, v: Var) -> bool {
    readers_of(body, v).iter().any(|r| match r.primitive {
        Primitive::Reshape { .. } => epilogue_read(body, r.output),
        ref p => crate::graph_compiler::elementwise(p) || matches!(p, Primitive::SwapPairs),
    })
}

/// Returns every node in `body` that reads `v`.
pub(crate) fn readers_of(body: &Graph, v: Var) -> Vec<&Node> {
    body.nodes()
        .iter()
        .filter(|node| node.inputs.contains(&v))
        .collect()
}

/// The two nodes consuming adjacent output column pairs of `dot`, when the epilogue is a single
/// two-member group along the last axis (e.g. a gated activation).
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

/// How a two-operand concat of dot-shaped values lays out the epilogue's two results.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Expansion {
    /// Concatenated along the last axis, as two column halves.
    Columns,
    /// Stacked on a new trailing axis, so the two results interleave.
    Pairs,
    /// Concatenated along the leading axis, as two row halves.
    Rows,
}

/// How `concat` lays out two dot-shaped values, or `None` if it isn't an expansion the GEMM
/// epilogue can write directly.
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

/// The two concat operands and their layout when the body's output is such an expansion.
pub(crate) fn gemm_expanded(body: &Graph, dot: &Node) -> Option<([Var; 2], Expansion)> {
    let out = body.outputs()[0];
    let concat = body.nodes().iter().find(|n| n.output == out)?;
    let layout = expansion(body, dot, concat)?;

    match concat.inputs.as_slice() {
        &[a, b] if gemm_pair(body, dot).is_none() => Some(([a, b], layout)),
        _ => None,
    }
}
