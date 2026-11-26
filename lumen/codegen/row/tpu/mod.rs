//! TPU row fusions: elementwise work around reductions over each row's last axis (norms, softmax),
//! with every value classified by how it lines up with the rows, generated as a spec for the Pallas row
//! program and launched.

#[cfg(lumen_tpu_linked)]
use std::collections::HashMap;

use crate::DType;
#[cfg(lumen_tpu_linked)]
use crate::codegen::devices::tpu::Tpu;
#[cfg(lumen_tpu_linked)]
use crate::codegen::devices::tpu::{Kernel, Rows};
#[cfg(lumen_tpu_linked)]
use crate::codegen::dot_general::tpu::block;
#[cfg(lumen_tpu_linked)]
use crate::codegen::row::RowCodegen;
use crate::device::tpu::layout::Layout;
use crate::graph::{Graph, Node, Primitive, TensorType, Var};
#[cfg(lumen_tpu_linked)]
use crate::plan::StepBase;

/// Most bytes a minimal float32 block of rows (16 rows of `n`) may take.
pub(crate) const ROW_BYTES: usize = 1 << 21;

/// Most bytes of table tiles a block of gathered rows fetches (one tile a row).
const GATHER_BYTES: usize = 4 << 20;

/// Most indices a gathered row fusion reads into scalar memory.
const GATHER_IDS: usize = 8192;

/// A row body opening with a row gather (`passes::gather_rows`): `body` reads the gathered rows as its first input in
/// place of the table and ids (inputs 0 and 1 of the whole body), `table` and `count` (the indices) the gather's, `flat`
/// whether the ids are one dimension (stored flat; else a tiled `[1, s]` row).
pub(crate) struct Gathered {
    pub(crate) body: Graph,
    pub(crate) table: TensorType,
    pub(crate) count: usize,
    #[cfg_attr(not(lumen_tpu_linked), allow(dead_code))]
    pub(crate) flat: bool,
}

impl Gathered {
    /// Table rows one TPU tile holds (8 for 32-bit, 16 for 16-bit).
    fn tile(&self) -> Option<usize> {
        Layout::new(self.table.dtype, &[1, 1]).ok().map(|l| l.rows)
    }

    /// Bytes of table tiles one gathered row fetches.
    fn row_bytes(&self) -> usize {
        self.tile().unwrap_or(8) * self.table.shape[1] * self.table.dtype.size_of()
    }
}

/// `body` split into its opening row gather and the rest, when it opens with one: a `Gather { axis: 0 }` of inputs 0
/// (a `[vocab, n]` table) and 1 (int32 indices, all dims 1 but the last), then a reshape keeping its rows when only
/// that reads it, and no other gather.
pub(crate) fn gathered(body: &Graph) -> Option<Gathered> {
    let nodes = body.nodes();
    let first = nodes.first()?;
    let inputs = body.inputs();

    // A reshape of the rows alone, keeping each one whole, is what the rest reads.
    let reshaped = nodes.get(1).is_some_and(|n| {
        matches!(n.primitive, Primitive::Reshape { .. })
            && n.inputs == [first.output]
            && body.type_of(n.output).shape.last() == body.type_of(first.output).shape.last()
            && !body.outputs().contains(&first.output)
            && nodes[2..].iter().all(|m| !m.inputs.contains(&first.output))
    });

    let (rows, more) = match reshaped {
        true => (nodes[1].output, nodes[2..].iter()),
        false => (first.output, nodes[1..].iter()),
    };

    let opens = matches!(first.primitive, Primitive::Gather { axis: 0 })
        && inputs.len() >= 2
        && first.inputs == inputs[..2]
        && !more
            .clone()
            .any(|n| matches!(n.primitive, Primitive::Gather { .. }));

    if !opens {
        return None;
    }

    let (table, ids) = (body.type_of(inputs[0]), body.type_of(inputs[1]));
    let rank = ids.shape.len();

    if table.shape.len() != 2
        || ids.dtype != DType::I32
        || rank == 0
        || ids.shape[..rank - 1].iter().any(|&d| d != 1)
    {
        return None;
    }

    let mut rest = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; body.types.len()];
    map[rows] = rest.input(body.type_of(rows).clone());

    for &v in &inputs[2..] {
        map[v] = rest.input(body.type_of(v).clone());
    }

    for node in more {
        let args: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = rest.apply(node.primitive.clone(), &args).ok()?;
    }

    let outputs: Vec<Var> = body.outputs().iter().map(|&v| map[v]).collect();
    rest.set_outputs(&outputs).ok()?;

    Some(Gathered {
        body: rest,
        table: table.clone(),
        count: ids.numel(),
        flat: rank == 1,
    })
}

/// How a row fusion value lines up with its `rows x n` operand.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Kind {
    /// Every element: `rows x n`.
    Full,
    /// One value per row, such as a reduction's result.
    PerRow,
    /// One value per column, shared by every row.
    Lanes,
    /// A single value.
    Scalar,
}

impl Kind {
    /// The kind's name in the kernel spec.
    #[cfg(lumen_tpu_linked)]
    pub(crate) fn name(self) -> &'static str {
        match self {
            Kind::Full => "full",
            Kind::PerRow => "row",
            Kind::Lanes => "lanes",
            Kind::Scalar => "scalar",
        }
    }
}

/// Whether `node` can join a TPU row fusion: 32-bit-or-narrower dtypes and an elementwise, reshape,
/// broadcast or reduction primitive, or an iota along the rows' last axis (an argmax's column index).
pub(crate) fn fusible(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;

    let narrow = node
        .inputs
        .iter()
        .chain([&node.output])
        .all(|&v| graph.type_of(v).dtype.bits() <= 32);

    let lanes = |dimension: usize| dimension + 1 == graph.type_of(node.output).shape.len();

    let row_op = crate::codegen::devices::tpu::fusible(&node.primitive)
        || matches!(
            node.primitive,
            Reshape { .. } | BroadcastInDim { .. } | ReduceSum { .. } | ReduceMax { .. }
        )
        || matches!(node.primitive, Iota { dimension, .. } if lanes(dimension));

    narrow && row_op
}

/// The axes `p` reduces, or `None` when it is not a sum or max reduction.
fn reduction(p: &Primitive) -> Option<&[usize]> {
    match p {
        Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => Some(axes),
        _ => None,
    }
}

/// The `(rows, n)` of the fusion's first reduction operand, viewed as rows of its last axis.
pub(crate) fn rows(body: &Graph) -> Option<(usize, usize)> {
    let node = body
        .nodes()
        .iter()
        .find(|n| reduction(&n.primitive).is_some())?;

    let x = body.type_of(node.inputs[0]);
    let n = *x.shape.last()?;
    Some((x.numel() / n.max(1), n))
}

/// Each value's kind in the row fusion `body`, or `None` when some value doesn't line up with
/// whole rows or a node mixes kinds it can't.
pub(crate) fn kinds(body: &Graph) -> Option<Vec<Kind>> {
    let (rows, n) = rows(body)?;
    let nodes = body.nodes();
    let full = |t: &TensorType| t.numel() == rows * n && t.shape.last() == Some(&n);

    let by_size = |t: &TensorType| match t.numel() {
        1 => Some(Kind::Scalar),
        _ if full(t) => Some(Kind::Full),
        m if m == rows => Some(Kind::PerRow),
        m if m == n => Some(Kind::Lanes),
        _ => None,
    };

    // Backward pass: a value takes the kind its readers need.
    let mut kind: Vec<Option<Kind>> = vec![None; body.types.len()];

    for node in nodes.iter().rev() {
        match &node.primitive {
            p if reduction(p).is_some() => kind[node.inputs[0]] = Some(Kind::Full),
            Primitive::BroadcastInDim {
                broadcast_dimensions,
                ..
            } => {
                let (x, out) = (node.inputs[0], body.type_of(node.output));
                let xt = body.type_of(x);
                // A broadcast whose last output dim comes from an `n`-wide operand dim spreads lanes; otherwise it spreads rows.
                let last = out.shape.len().saturating_sub(1);

                let onto_last = broadcast_dimensions
                    .iter()
                    .position(|&d| d == last)
                    .is_some_and(|k| xt.shape[k] == n);

                kind[x] = Some(match xt.numel() {
                    1 => Kind::Scalar,
                    _ if full(out) && onto_last && xt.numel() == n => Kind::Lanes,
                    _ if full(out) => Kind::PerRow,
                    _ => kind[node.output].or(by_size(xt))?,
                });
            }
            _ => {
                if let Some(k) = kind[node.output] {
                    for &v in &node.inputs {
                        if body.type_of(v).numel() > 1 && kind[v].is_none() {
                            kind[v] = Some(k);
                        }
                    }
                }
            }
        }
    }

    // Elements of a value of each kind.
    let sized = |k: Kind| match k {
        Kind::Full => rows * n,
        Kind::PerRow => rows,
        Kind::Lanes => n,
        Kind::Scalar => 1,
    };

    // Forward pass: settle each node output's kind from its operands.
    for node in nodes {
        let out = body.type_of(node.output);

        let k = match &node.primitive {
            p if reduction(p).is_some() => Kind::PerRow,
            Primitive::BroadcastInDim { .. } if full(out) => Kind::Full,
            Primitive::BroadcastInDim { .. } if out.numel() == 1 => Kind::Scalar,
            Primitive::BroadcastInDim { .. } => match kind[node.inputs[0]] {
                Some(Kind::Scalar) | None => kind[node.output].or(by_size(out))?,
                Some(k) => k,
            },
            Primitive::Reshape { .. } => kind[node.inputs[0]].or(by_size(out))?,
            // The widest operand kind of the output's size: one row's per-row value is no scalar.
            _ => {
                let from = node
                    .inputs
                    .iter()
                    .filter_map(|&v| kind[v])
                    .filter(|&k| sized(k) == out.numel())
                    .max_by_key(|&k| match k {
                        Kind::Full => 3,
                        Kind::PerRow => 2,
                        Kind::Lanes => 1,
                        Kind::Scalar => 0,
                    });

                kind[node.output].or(from).or(by_size(out))?
            }
        };

        kind[node.output] = Some(k);
    }

    for &v in body.inputs() {
        if kind[v].is_none() {
            kind[v] = by_size(body.type_of(v));
        }
    }

    let size = sized;

    // Every used value's size must match its kind; unused slots default to scalar.
    let mut known = Vec::with_capacity(kind.len());

    for (v, k) in kind.iter().enumerate() {
        let used = body.inputs().contains(&v) || nodes.iter().any(|n| n.output == v);

        match k {
            Some(k) if body.type_of(v).numel() == size(*k) => known.push(*k),
            None if !used => known.push(Kind::Scalar),
            _ => return None,
        }
    }

    // Check each node combines its operands' kinds in a way the kernel supports.
    for node in nodes {
        let out = known[node.output];
        let operands: Vec<Kind> = node.inputs.iter().map(|&v| known[v]).collect();

        let consistent = match &node.primitive {
            p if reduction(p).is_some() => {
                let x = body.type_of(node.inputs[0]);
                reduction(p) == Some(&[x.shape.len() - 1][..]) && operands == [Kind::Full]
            }
            Primitive::BroadcastInDim { .. } => match (operands[0], out) {
                (Kind::Scalar, _) => true,
                (Kind::PerRow | Kind::Lanes, Kind::Full) => true,
                (a, b) => a == b,
            },
            Primitive::Reshape { .. } => operands[0] == out,
            _ => operands.iter().all(|&k| k == out || k == Kind::Scalar),
        };

        if !consistent {
            return None;
        }
    }

    Some(known)
}

/// Whether `t` can be streamed as `kind`: rows of whole tiles, a lane vector or a scalar.
pub(crate) fn readable(t: &TensorType, kind: Kind) -> bool {
    let Ok(layout) = Layout::new(t.dtype, &t.shape) else {
        return false;
    };

    let rank = t.shape.len();

    let stacked = rank <= 2
        || t.shape[..rank - 2].iter().product::<usize>() == 1
        || t.shape[rank - 2].is_multiple_of(layout.rows);

    match kind {
        Kind::Full => rank >= 2 && stacked,
        // One element sits at its buffer's start in any layout: a one-row value of any rank.
        Kind::PerRow => (rank >= 2 && t.shape[rank - 1] == 1 && stacked) || t.numel() == 1,
        Kind::Lanes => rank == 1,
        Kind::Scalar => rank == 0,
    }
}

/// Whether the row fusion `body` runs on TPU: whole-row operands, row-shaped outputs and a
/// minimal block within `ROW_BYTES`.
pub(crate) fn fits(body: &Graph) -> bool {
    if body
        .nodes()
        .iter()
        .any(|n| matches!(n.primitive, Primitive::Gather { .. }))
    {
        return gathered(body).is_some_and(|g| gather_fits(&g) && fits(&g.body));
    }

    let Some((rows, n)) = rows(body) else {
        return false;
    };

    let Some(kinds) = kinds(body) else {
        return false;
    };

    let outer = body.inputs().iter().chain(body.outputs());
    let unit = 16.min(rows);

    outer.clone().all(|&v| readable(body.type_of(v), kinds[v]))
        && body
            .outputs()
            .iter()
            .all(|&v| matches!(kinds[v], Kind::Full | Kind::PerRow))
        && unit * n * 4 <= ROW_BYTES
}

/// Whether the row kernel fetches `g`'s rows itself: a float32, bfloat16 or int32 table of whole lanes, the indices
/// within `GATHER_IDS`, its rows the row fusion's (the gathered value a `Full` operand) and one row's tiles within
/// `GATHER_BYTES`.
fn gather_fits(g: &Gathered) -> bool {
    let n = g.table.shape[1];

    let full = kinds(&g.body).is_some_and(|k| k[g.body.inputs()[0]] == Kind::Full)
        && rows(&g.body) == Some((g.count, n));

    matches!(g.table.dtype, DType::F32 | DType::BF16 | DType::I32)
        && n.is_multiple_of(128)
        && g.count <= GATHER_IDS
        && full
        && g.row_bytes() <= GATHER_BYTES
}

/// The JSON node for `node` in a row kernel, tagged with its value's `kind`.
#[cfg(lumen_tpu_linked)]
pub(crate) fn row_node(node: &Node, args: &[usize], kind: Kind) -> String {
    let named = |params: String| {
        format!(
            "{{\"name\": \"{}\", \"params\": {{{params}}}, \"args\": {args:?}, \"kind\": \"{}\"}}",
            node.primitive.name(),
            kind.name()
        )
    };

    match &node.primitive {
        Primitive::ReduceSum { accum_dtype, .. } => {
            named(format!("\"accum_dtype\": \"{}\"", accum_dtype.full_name()))
        }
        Primitive::ReduceMax { .. }
        | Primitive::Reshape { .. }
        | Primitive::BroadcastInDim { .. } => named(String::new()),
        Primitive::Iota { dtype, .. } => named(format!("\"dtype\": \"{}\"", dtype.full_name())),
        // Splice the kind into the plain node's closing brace.
        p => {
            let plain = crate::codegen::devices::tpu::node(p, args);

            format!(
                "{}, \"kind\": \"{}\"}}",
                &plain[..plain.len() - 1],
                kind.name()
            )
        }
    }
}

/// TPU row fusions: the fixed Pallas row program, specialized by a spec of the rows, their block and the
/// body's nodes, each classified by how it lines up with the rows.
#[cfg(lumen_tpu_linked)]
impl RowCodegen for Tpu {
    /// The kernel's name and spec.
    type Code = (String, String);
    type Options = ();

    /// The spec of the row fusion `body`. Inputs are read as their buffers; a body opening with a row gather fetches
    /// its first operand's rows from the table at the ids (`gather`), the rest following them.
    fn row(body: &Graph, _: &[bool], _: &()) -> Result<(String, String), String> {
        let failed = |why: &str| format!("a row fusion on TPU: {why}");
        let gather = gathered(body);
        let body = gather.as_ref().map_or(body, |g| &g.body);
        let (rows, n) = rows(body).ok_or_else(|| failed("no reduction over rows"))?;
        let kinds = kinds(body).ok_or_else(|| failed("values not of whole rows"))?;

        let mut value: HashMap<Var, usize> = body
            .inputs()
            .iter()
            .enumerate()
            .map(|(k, &v)| (v, k))
            .collect();

        let mut nodes = Vec::new();

        for node in body.nodes() {
            let args: Vec<usize> = node.inputs.iter().map(|v| value[v]).collect();
            nodes.push(row_node(node, &args, kinds[node.output]));
            value.insert(node.output, body.inputs().len() + nodes.len() - 1);
        }

        let operand = |v: &Var| {
            format!(
                "[\"{}\", \"{}\"]",
                body.type_of(*v).dtype.full_name(),
                kinds[*v].name()
            )
        };

        let operands: Vec<String> = body.inputs().iter().map(operand).collect();

        let outputs: Vec<String> = body
            .outputs()
            .iter()
            .map(|v| {
                format!(
                    "[{}, \"{}\", \"{}\"]",
                    value[v],
                    body.type_of(*v).dtype.full_name(),
                    kinds[*v].name()
                )
            })
            .collect();

        // Block rows come in multiples of the tallest tile among the operand dtypes.
        let unit = body
            .inputs()
            .iter()
            .chain(body.outputs())
            .filter_map(|&v| Layout::new(body.type_of(v).dtype, &[1, 1]).ok())
            .map(|l| l.rows)
            .max()
            .unwrap_or(8);

        // A block's gathered rows fetch a table tile each.
        let most = gather
            .as_ref()
            .map_or(usize::MAX, |g| (GATHER_BYTES / g.row_bytes()).max(1));

        let bm = block_rows(rows, n, unit, most).map_err(|e| failed(&e))?;

        let fetched = gather.as_ref().map(|g| {
            format!(
                ", \"gather\": {{\"vocab\": {}, \"dtype\": \"{}\", \"flat\": {}}}",
                g.table.shape[0],
                g.table.dtype.full_name(),
                g.flat,
            )
        });

        let spec = format!(
            "\"height\": {rows}, \"n\": {n}, \"block\": {bm}, \"inputs\": [{}], \"nodes\": [{}], \"outputs\": [{}]{}",
            operands.join(", "),
            nodes.join(", "),
            outputs.join(", "),
            fetched.as_deref().unwrap_or(""),
        );

        let kernel = match gather {
            Some(_) => format!("row_gather_{rows}x{n}_{bm}"),
            None => format!("row_{rows}x{n}_{bm}"),
        };

        Ok((kernel, spec))
    }
}

/// Picks how many of `rows` rows of `n` elements a step holds: a multiple of `unit` dividing `rows` within the
/// ROW_BYTES budget (counted at 4 bytes an element) and `most` rows, or all rows when there are fewer than `unit` or
/// no such multiple divides them and they fit twice both.
#[cfg(lumen_tpu_linked)]
pub(crate) fn block_rows(rows: usize, n: usize, unit: usize, most: usize) -> Result<usize, String> {
    if rows < unit {
        return Ok(rows);
    }

    let cap = (ROW_BYTES / (n * 4)).min(most).max(unit);

    block(rows, unit, cap)
        .or_else(|| (rows * n * 4 <= 2 * ROW_BYTES && rows <= 2 * most).then_some(rows))
        .ok_or_else(|| "its rows have no block of whole tiles".into())
}

/// The kernel of the row fusion `step`.
#[cfg(lumen_tpu_linked)]
pub(crate) fn kernel(step: &StepBase) -> Result<Kernel, String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("a row fusion")
    };

    let (kernel, spec) = Tpu::row(body, &[], &()).map_err(|e| format!("{}: {e}", step.label))?;

    Ok(Kernel::new(
        "lumen.codegen.row.tpu.kernel",
        kernel,
        spec,
        Rows::Split,
    ))
}
