//! TPU small fusions: a group of tiny values computed from inputs alone (a rotary's tables, the sign of its rotation),
//! run as one kernel holding every value in VMEM. Each value is computed as a 2-D `(rows, columns)` plane, its last axis
//! the columns and the rest the rows: elementwise ops, fills, iotas, broadcasts, reshapes that keep the plane and
//! concatenations along the last axis. Inputs and outputs are read and written whole.

use crate::DType;
#[cfg(lumen_tpu_linked)]
use crate::codegen::devices::tpu::{Kernel, Rows};
use crate::graph::{Graph, Node, Primitive, TensorType};
#[cfg(lumen_tpu_linked)]
use crate::plan::StepBase;

/// The fusion name of a small fusion.
pub(crate) const SMALL: &str = "small";

/// Most bytes of any one value a small fusion holds.
const VALUE_BYTES: usize = 64 << 10;

/// The `(rows, columns)` plane a value of `shape` is computed as: its last axis the columns, the rest the rows; `None`
/// for a scalar.
pub(crate) fn plane(shape: &[usize]) -> Option<(usize, usize)> {
    shape
        .split_last()
        .map(|(&n, rest)| (rest.iter().product(), n))
}

/// Whether a value of type `t` is small enough, of a dtype the kernel computes in.
fn holds(t: &TensorType) -> bool {
    matches!(
        t.dtype,
        DType::F32 | DType::BF16 | DType::I32 | DType::U32 | DType::Bool
    ) && t.numel() * t.dtype.size_of().max(1) <= VALUE_BYTES
}

/// Whether `node` of `graph` can be computed in a small fusion, its values held as planes.
pub(crate) fn takes(graph: &Graph, node: &Node) -> bool {
    let out = graph.type_of(node.output);
    let ins: Vec<&TensorType> = node.inputs.iter().map(|&v| graph.type_of(v)).collect();

    if !holds(out) || !ins.iter().all(|t| holds(t)) || out.shape.is_empty() {
        return false;
    }

    let same = |t: &TensorType| plane(&t.shape) == plane(&out.shape);

    match &node.primitive {
        Primitive::Full { .. } | Primitive::Iota { .. } => true,
        Primitive::Reshape { .. } => same(ins[0]),
        Primitive::BroadcastInDim {
            broadcast_dimensions,
            ..
        } => {
            let x = ins[0];
            let (rank, from) = (out.shape.len(), x.shape.len());

            // A scalar spreads anywhere; a value of the same plane is unchanged; else a value of the result's rank,
            // the identity on dims, broadcast along its rows or columns as a whole.
            from == 0
                || same(x)
                || (from == rank
                    && broadcast_dimensions
                        .iter()
                        .enumerate()
                        .all(|(k, &d)| k == d)
                    && {
                        let (r, c) = plane(&x.shape).expect("a matrix");
                        let (rr, cc) = plane(&out.shape).expect("a matrix");
                        (r == 1 || r == rr) && (c == 1 || c == cc)
                    })
        }
        Primitive::Concatenate { dimension, .. } => {
            *dimension + 1 == out.shape.len()
                && ins
                    .iter()
                    .all(|t| plane(&t.shape).map(|p| p.0) == plane(&out.shape).map(|p| p.0))
        }
        p => {
            crate::codegen::devices::tpu::fusible(p)
                && !matches!(p, Primitive::BroadcastInDim { .. })
                && ins.iter().all(|t| t.shape.is_empty() || same(t))
        }
    }
}

/// Whether a small fusion can read or write `t` whole as its plane: a scalar of 32 bits read into scalar memory, or
/// a tensor stored as its plane (a matrix, leading dims of 1 or rows of whole tiles, or a short 32-bit vector).
pub(crate) fn buffer(t: &TensorType, written: bool) -> bool {
    let rank = t.shape.len();
    let wide = t.dtype.bits() == 32 && t.dtype != DType::Bool;

    match rank {
        0 => !written && wide,
        1 => wide && t.shape[0] <= 128,
        _ => {
            let rows =
                crate::device::tpu::layout::Layout::new(t.dtype, &[1, 1]).map_or(8, |l| l.rows);
            t.dtype != DType::Bool
                && (t.shape[..rank - 2].iter().product::<usize>() == 1
                    || t.shape[rank - 2].is_multiple_of(rows))
        }
    }
}

/// The kernel of the small fusion `step`: its inputs read whole into scalar memory or VMEM, its nodes computed as
/// planes, its outputs written whole (`lumen.codegen.elementwise.tpu.small`).
#[cfg(lumen_tpu_linked)]
pub(crate) fn kernel(step: &StepBase) -> Result<Kernel, String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("a small fusion")
    };

    let mut index = vec![usize::MAX; body.types.len()];
    let mut spec_inputs = Vec::new();

    for (k, &v) in body.inputs().iter().enumerate() {
        index[v] = k;
        let t = body.type_of(v);

        spec_inputs.push(format!(
            "[\"{}\", {}]",
            t.dtype.full_name(),
            plane(&t.shape).map_or("null".into(), |(r, c)| format!("[{r}, {c}]"))
        ));
    }

    let mut nodes = Vec::new();

    for (i, node) in body.nodes().iter().enumerate() {
        index[node.output] = body.inputs().len() + i;
        let args: Vec<usize> = node.inputs.iter().map(|&v| index[v]).collect();
        let out = body.type_of(node.output);
        let (r, c) = plane(&out.shape).expect("a plane");

        let (name, params) = match &node.primitive {
            Primitive::Iota { dimension, .. } => {
                let rank = out.shape.len();

                // Along the last axis, each column's index; along another, each row's index in that dim.
                let (lanes, inner, size) = match *dimension + 1 == rank {
                    true => (true, 1, c),
                    false => (
                        false,
                        out.shape[dimension + 1..rank - 1].iter().product::<usize>(),
                        out.shape[*dimension],
                    ),
                };

                (
                    "iota".to_string(),
                    format!("\"lanes\": {lanes}, \"inner\": {inner}, \"size\": {size}"),
                )
            }
            Primitive::BroadcastInDim { .. } => ("broadcast".into(), String::new()),
            Primitive::Reshape { .. } => ("reshape".into(), String::new()),
            Primitive::Concatenate { .. } => ("concatenate".into(), String::new()),
            p => {
                let plain = crate::codegen::devices::tpu::node(p, &args);
                nodes.push(format!(
                    "{}, \"dtype\": \"{}\", \"plane\": [{r}, {c}]}}",
                    &plain[..plain.len() - 1],
                    out.dtype.full_name()
                ));
                continue;
            }
        };

        nodes.push(format!(
            "{{\"name\": \"{name}\", \"params\": {{{params}}}, \"args\": {args:?}, \"dtype\": \"{}\", \"plane\": [{r}, {c}]}}",
            out.dtype.full_name()
        ));
    }

    let outputs: Vec<String> = body
        .outputs()
        .iter()
        .map(|&v| {
            let t = body.type_of(v);
            let (r, c) = plane(&t.shape).expect("a plane");
            format!("[{}, \"{}\", [{r}, {c}]]", index[v], t.dtype.full_name())
        })
        .collect();

    let spec = format!(
        "\"inputs\": [{}], \"nodes\": [{}], \"outputs\": [{}]",
        spec_inputs.join(", "),
        nodes.join(", "),
        outputs.join(", ")
    );

    let mut kernel = Kernel::new(
        "lumen.codegen.elementwise.tpu.small",
        format!("small_{}", body.nodes().len()),
        spec,
        Rows::Split,
    );

    kernel.context = Some(step.label.to_string());
    Ok(kernel)
}
