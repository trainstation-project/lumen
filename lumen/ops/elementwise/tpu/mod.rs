//! Elementwise primitives on TPU: the spec of their kernel (`__init__.py`
//! beside this file, in lumen's Pallas): its operands' and output's dtypes,
//! their shape (one), the nodes it applies (a primitive each, by name and
//! parameters, of the operands and the nodes before) and its blocks.

use crate::DType;
use crate::device::tpu::layout::Layout;
use crate::graph::{Primitive, UnaryOp};
use crate::graph_compiler::plan::Step;
use crate::ops::dot_general::tpu::block;
use crate::ops::tpu::{Launch, launch};

/// Whether `p` is elementwise: its kernel runs it.
pub(crate) fn elementwise(p: &Primitive) -> bool {
    use Primitive::*;
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
            | Select
            | Unary(_)
            | Binary(_)
    )
}

/// `p` as a node of the spec, of the values `args` (the operands, then the
/// nodes before): its name and parameters as `lumen.graph.prims` takes them.
fn node(p: &Primitive, args: &[usize]) -> String {
    let params = match p {
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

/// The blocks of an elementwise kernel over `shape` of `dtypes`: of its
/// last two dimensions (whole tiles of every dtype's, or the dimension),
/// or of its elements (one dimension or none); an error if none divides.
fn blocks(shape: &[usize], dtypes: &[DType]) -> Result<Vec<usize>, String> {
    // Memory-bound: blocks large enough that a step's copies dwarf its
    // bookkeeping, small enough for every operand's two in VMEM.
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
    if step.inputs.iter().any(|(_, t)| t.shape != out.shape) {
        return Err(format!(
            "elementwise on TPU: operands of the output's shape, got {}",
            step.primitive.name()
        ));
    }
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
    let spec = format!(
        "\"shape\": {:?}, \"inputs\": [{}], \"nodes\": [{}], \"output\": \"{}\", \"blocks\": {:?}",
        out.shape,
        names.join(", "),
        node(&step.primitive, &args),
        out.dtype.full_name(),
        blocks(&out.shape, &dtypes)?,
    );
    let rows: Vec<*const u8> = inputs
        .iter()
        .copied()
        .chain([output.cast_const()])
        .collect();
    let launched = Launch {
        module: "lumen.ops.elementwise.tpu",
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
    };
    launch(index, launched)
}
