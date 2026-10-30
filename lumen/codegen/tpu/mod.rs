//! TPU code generation: the Rust side that turns fused steps into JSON specs for the Pallas kernels.

pub(crate) mod attention;
#[cfg(lumen_tpu_linked)]
mod elementwise;
#[cfg(lumen_tpu_linked)]
pub(crate) mod gemm;
#[cfg(lumen_tpu_linked)]
pub(crate) mod gemm_row;
pub(crate) mod indexing;
pub(crate) mod reduction;
pub(crate) mod row;

use crate::Scalar;
use crate::graph::Primitive;
use crate::graph::UnaryOp;
#[cfg(lumen_tpu_linked)]
use crate::graph_compiler::plan::Step;
#[cfg(lumen_tpu_linked)]
pub(crate) use elementwise::encode;

/// The TPU code generator.
pub(crate) struct Tpu;

/// Whether `p` can run inside a plain elementwise TPU loop (only scalar broadcasts among layout ops).
pub(crate) fn fusible(p: &Primitive) -> bool {
    use Primitive::*;
    if let BroadcastInDim {
        broadcast_dimensions,
        ..
    } = p
    {
        return broadcast_dimensions.is_empty();
    }
    matches!(
        p,
        Add | Sub
            | Mul
            | Div
            | Max
            | Eq
            | Lt
            | Neg
            | Exp
            | Log
            | Sqrt
            | Tanh
            | Logistic
            | Cast { .. }
            | Full { .. }
            | Select
            | Unary(_)
            | Binary(_)
    )
}

/// The JSON node `{"name", "params", "args"}` the Python kernels rebuild `p` from, with `args` indexing earlier values.
pub(crate) fn node(p: &Primitive, args: &[usize]) -> String {
    let params = match p {
        Primitive::Full {
            fill_value, dtype, ..
        } => {
            // JSON has no NaN or infinity literals; Python's json module reads these spellings.
            let value = match *fill_value {
                Scalar::Bool(b) => b.to_string(),
                Scalar::Int(i) => i.to_string(),
                Scalar::Float(f) if f.is_nan() => "NaN".into(),
                Scalar::Float(f) if f.is_infinite() => {
                    format!("{}Infinity", if f < 0.0 { "-" } else { "" })
                }
                Scalar::Float(f) => format!("{f:?}"),
            };
            format!(
                "\"fill_value\": {value}, \"dtype\": \"{}\"",
                dtype.full_name()
            )
        }
        Primitive::BroadcastInDim { .. } => "\"broadcast_dimensions\": []".into(),
        Primitive::Cast { new_dtype } => format!("\"new_dtype\": \"{}\"", new_dtype.full_name()),
        Primitive::Unary(UnaryOp::ReducePrecision {
            exponent_bits,
            mantissa_bits,
        }) => format!("\"exponent_bits\": {exponent_bits}, \"mantissa_bits\": {mantissa_bits}"),
        _ => String::new(),
    };
    format!(
        "{{\"name\": \"{}\", \"params\": {{{params}}}, \"args\": {args:?}}}",
        p.name()
    )
}

/// The `(row stride, column offset)` of output `k` when it is written into a window of a wider matrix.
#[cfg(lumen_tpu_linked)]
pub(crate) fn window(step: &Step, k: usize) -> Option<(usize, usize)> {
    let view = step.output_views.get(k)?.as_ref()?;
    let rank = view.strides.len();
    (rank >= 2).then(|| (view.strides[rank - 2], view.offset % view.strides[rank - 2]))
}
