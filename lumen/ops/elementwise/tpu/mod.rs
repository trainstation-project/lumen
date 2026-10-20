use crate::Tensor;
use crate::device::tpu::layout::Layout;
use crate::graph::TensorType;
use crate::graph::{Primitive, UnaryOp};
use crate::graph_compiler::plan::{Buffer, Step};
use crate::ops::dot_general::tpu::block;
use crate::ops::tpu::{Launch, launch};
use crate::{DType, Scalar};

pub(crate) fn elementwise(p: &Primitive) -> bool {
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

pub(crate) fn node(p: &Primitive, args: &[usize]) -> String {
    let params = match p {
        Primitive::Full {
            fill_value, dtype, ..
        } => {
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

fn blocks(shape: &[usize], dtypes: &[DType]) -> Result<Vec<usize>, String> {
    let failed = |what: &str| format!("elementwise on TPU: {what} has no block of whole tiles");
    if shape.len() < 2 {
        let n = shape.iter().product::<usize>().max(1);
        return Ok(vec![
            block(n, 1024, 64 * 1024).ok_or_else(|| failed("its length"))?,
        ]);
    }
    let rows = dtypes
        .iter()
        .map(|&d| Layout::new(d, &[1, 1]).map(|l| l.rows))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(8);
    let (m, n) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    Ok(vec![
        block(m, rows, 512).ok_or_else(|| failed("its rows"))?,
        block(n, 128, 2048).ok_or_else(|| failed("its columns"))?,
    ])
}

pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let out = &step.output.1;
    if out.numel() == 0 {
        return Ok(());
    }
    if step
        .inputs
        .iter()
        .any(|(_, t)| t.shape != out.shape && !t.shape.is_empty())
    {
        return Err(format!(
            "elementwise on TPU: operands of the output's shape, got {}",
            step.primitive.name()
        ));
    }
    let broadcast: Vec<bool> = step
        .inputs
        .iter()
        .map(|(_, t)| t.shape != out.shape)
        .collect();
    let dtypes: Vec<DType> = step
        .inputs
        .iter()
        .map(|(_, t)| t.dtype)
        .chain([out.dtype])
        .collect();
    let names: Vec<String> = step
        .inputs
        .iter()
        .map(|(_, t)| format!("\"{}\"", t.dtype.full_name()))
        .collect();
    let args: Vec<usize> = (0..step.inputs.len()).collect();
    let blocks = blocks(&out.shape, &dtypes)?;
    let mut kernel = step.primitive.name().to_owned();
    if let Some((_, t)) = step.inputs.first().filter(|(_, t)| t.dtype != out.dtype) {
        kernel += &format!("_{}", t.dtype);
    }
    let shown: Vec<String> = blocks.iter().map(usize::to_string).collect();
    kernel += &format!("_{}_{}", out.dtype, shown.join("x"));
    let spec = format!(
        "\"shape\": {:?}, \"inputs\": [{}], \"broadcast\": {broadcast:?}, \"nodes\": [{}], \"output\": \"{}\", \"blocks\": {:?}",
        out.shape,
        names.join(", "),
        node(&step.primitive, &args),
        out.dtype.full_name(),
        blocks,
    );
    let rows: Vec<*const u8> = inputs
        .iter()
        .copied()
        .chain([output.cast_const()])
        .collect();
    let launched = Launch {
        module: "lumen.ops.elementwise.tpu",
        kernel,
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
    };
    launch(index, launched)
}

pub(crate) fn fill(t: &Tensor, value: Scalar) -> Result<(), String> {
    assert!(
        t.is_contiguous(),
        "a TPU tensor is filled whole: contiguous"
    );
    let ty = TensorType::new(t.dtype(), t.shape());
    let step = Step {
        primitive: Primitive::Full {
            shape: ty.shape.clone(),
            fill_value: value,
            dtype: ty.dtype,
        },
        label: "fill_",
        inputs: Vec::new(),
        views: Vec::new(),
        output: (Buffer::Workspace(0), ty),
        extra_outputs: Vec::new(),
        output_views: vec![None],
        scratch: None,
        scope: &[],
        sources: Vec::new(),
    };
    encode(t.device().index(), &step, &[], t.data_ptr())
}
