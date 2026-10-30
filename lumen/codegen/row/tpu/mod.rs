//! TPU row fusions: elementwise work around reductions over each row's last axis (norms, softmax),
//! with every value classified by how it lines up with the rows, generated as a spec for the Pallas row
//! program and launched.

#[cfg(lumen_tpu_linked)]
use std::collections::HashMap;

#[cfg(lumen_tpu_linked)]
use crate::codegen::dot_general::tpu::block;
#[cfg(lumen_tpu_linked)]
use crate::codegen::row::RowCodegen;
#[cfg(lumen_tpu_linked)]
use crate::codegen::utils::tpu::Tpu;
use crate::device::tpu::layout::Layout;
#[cfg(lumen_tpu_linked)]
use crate::graph::Var;
use crate::graph::{Graph, Node, Primitive, TensorType};
#[cfg(lumen_tpu_linked)]
use crate::graph_compiler::plan::Step;
#[cfg(lumen_tpu_linked)]
use crate::kernels::tpu::{Launch, launch};

/// Most bytes a minimal float32 block of rows (16 rows of `n`) may take.
pub(crate) const ROW_BYTES: usize = 1 << 21;

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
/// broadcast or reduction primitive.
pub(crate) fn fusible(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;

    let narrow = node
        .inputs
        .iter()
        .chain([&node.output])
        .all(|&v| graph.type_of(v).dtype.bits() <= 32);

    let row_op = crate::codegen::utils::tpu::fusible(&node.primitive)
        || matches!(
            node.primitive,
            Reshape { .. } | BroadcastInDim { .. } | ReduceSum { .. } | ReduceMax { .. }
        );

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
            _ => kind[node.output].or(by_size(out))?,
        };

        kind[node.output] = Some(k);
    }

    for &v in body.inputs() {
        if kind[v].is_none() {
            kind[v] = by_size(body.type_of(v));
        }
    }

    let size = |k: Kind| match k {
        Kind::Full => rows * n,
        Kind::PerRow => rows,
        Kind::Lanes => n,
        Kind::Scalar => 1,
    };

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
        Kind::PerRow => rank >= 2 && t.shape[rank - 1] == 1 && stacked,
        Kind::Lanes => rank == 1,
        Kind::Scalar => rank == 0,
    }
}

/// Whether the row fusion `body` runs on TPU: whole-row operands, row-shaped outputs and a
/// minimal block within `ROW_BYTES`.
pub(crate) fn fits(body: &Graph) -> bool {
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
        // Splice the kind into the plain node's closing brace.
        p => {
            let plain = crate::codegen::utils::tpu::node(p, args);

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

    /// The spec of the row fusion `body`. Inputs are read as their buffers.
    fn row(body: &Graph, _: &[bool], _: &()) -> Result<(String, String), String> {
        let failed = |why: &str| format!("a row fusion on TPU: {why}");
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

        let bm = block_rows(rows, n, unit).map_err(|e| failed(&e))?;

        let spec = format!(
            "\"height\": {rows}, \"n\": {n}, \"block\": {bm}, \"inputs\": [{}], \"nodes\": [{}], \"outputs\": [{}]",
            operands.join(", "),
            nodes.join(", "),
            outputs.join(", "),
        );

        Ok((format!("row_{rows}x{n}_{bm}"), spec))
    }
}

/// Picks how many of `rows` rows of `n` elements a step holds: a multiple of `unit` dividing `rows` within the
/// ROW_BYTES budget (counted at 4 bytes an element), or all rows when there are fewer than `unit`.
#[cfg(lumen_tpu_linked)]
pub(crate) fn block_rows(rows: usize, n: usize, unit: usize) -> Result<usize, String> {
    if rows < unit {
        return Ok(rows);
    }

    let cap = (ROW_BYTES / (n * 4)).max(unit);
    block(rows, unit, cap).ok_or_else(|| "its rows have no block of whole tiles".into())
}

/// Launches the row fusion `step` on TPU `index`, writing into `output`.
#[cfg(lumen_tpu_linked)]
pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("a row fusion")
    };

    let (kernel, spec) = Tpu::row(body, &[], &()).map_err(|e| format!("{}: {e}", step.label))?;
    let n_in = step.inputs.len();

    let rows_of: Vec<*const u8> = inputs[..n_in]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(inputs[n_in..].iter().copied())
        .collect();

    let launched = Launch {
        module: "lumen.codegen.row.tpu",
        kernel,
        spec,
        rows: &rows_of,
        scalars: &[],
        host: &[],
        sparse: None,
    };

    launch(index, launched)
}
