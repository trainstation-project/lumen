#[cfg(lumen_tpu_linked)]
use std::collections::HashMap;

use crate::device::tpu::layout::Layout;
#[cfg(lumen_tpu_linked)]
use crate::graph::Var;
use crate::graph::{Graph, Node, Primitive, TensorType};
#[cfg(lumen_tpu_linked)]
use crate::graph_compiler::plan::Step;
#[cfg(lumen_tpu_linked)]
use crate::kernels::tpu::{Launch, launch};

pub(crate) const ROW_BYTES: usize = 1 << 21;

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Kind {
    Full,
    PerRow,
    Lanes,
    Scalar,
}

impl Kind {
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

pub(crate) fn fusible(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;
    let narrow = node
        .inputs
        .iter()
        .chain([&node.output])
        .all(|&v| graph.type_of(v).dtype.bits() <= 32);
    let row_op = super::fusible(&node.primitive)
        || matches!(
            node.primitive,
            Reshape { .. } | BroadcastInDim { .. } | ReduceSum { .. } | ReduceMax { .. }
        );
    narrow && row_op
}

fn reduction(p: &Primitive) -> Option<&[usize]> {
    match p {
        Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => Some(axes),
        _ => None,
    }
}

pub(crate) fn rows(body: &Graph) -> Option<(usize, usize)> {
    let node = body
        .nodes()
        .iter()
        .find(|n| reduction(&n.primitive).is_some())?;
    let x = body.type_of(node.inputs[0]);
    let n = *x.shape.last()?;
    Some((x.numel() / n.max(1), n))
}

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
    let mut known = Vec::with_capacity(kind.len());
    for (v, k) in kind.iter().enumerate() {
        let used = body.inputs().contains(&v) || nodes.iter().any(|n| n.output == v);
        match k {
            Some(k) if body.type_of(v).numel() == size(*k) => known.push(*k),
            None if !used => known.push(Kind::Scalar),
            _ => return None,
        }
    }
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
        p => {
            let plain = super::node(p, args);
            format!(
                "{}, \"kind\": \"{}\"}}",
                &plain[..plain.len() - 1],
                kind.name()
            )
        }
    }
}

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
    let failed = |why: &str| format!("{} on TPU: {why}", step.label);
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
    let unit = body
        .inputs()
        .iter()
        .chain(body.outputs())
        .filter_map(|&v| Layout::new(body.type_of(v).dtype, &[1, 1]).ok())
        .map(|l| l.rows)
        .max()
        .unwrap_or(8);
    let bm = crate::kernels::row::tpu::block_rows(rows, n, unit).map_err(|e| failed(&e))?;
    let spec = format!(
        "\"height\": {rows}, \"n\": {n}, \"block\": {bm}, \"inputs\": [{}], \"nodes\": [{}], \"outputs\": [{}]",
        operands.join(", "),
        nodes.join(", "),
        outputs.join(", "),
    );
    let n_in = step.inputs.len();
    let rows_of: Vec<*const u8> = inputs[..n_in]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(inputs[n_in..].iter().copied())
        .collect();
    let launched = Launch {
        module: "lumen.kernels.row.tpu",
        kernel: format!("row_{rows}x{n}_{bm}"),
        spec,
        rows: &rows_of,
        scalars: &[],
        host: &[],
        sparse: None,
    };
    launch(index, launched)
}
