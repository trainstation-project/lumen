use std::collections::HashMap;

use super::{node, window};
use crate::DType;
use crate::device::tpu::layout::Layout;
use crate::graph::{Primitive, TensorType, Var};
use crate::graph_compiler::plan::Step;
use crate::ops::dot_general::tpu::block;
use crate::ops::tpu::{Launch, launch};

const BLOCK_BYTES: usize = 1 << 20;

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
    let bm = block(m, rows, 512).ok_or_else(|| failed("its rows"))?;
    let bn = block(n, 128, 2048).ok_or_else(|| failed("its columns"))?;
    let lead: usize = shape[..shape.len() - 2].iter().product();
    let width = dtypes.iter().map(|d| d.size_of()).max().unwrap_or(4);
    let tile = bm.div_ceil(rows) * rows * bn.div_ceil(128) * 128 * width;
    let bl = (1..=lead)
        .rev()
        .find(|&b| lead.is_multiple_of(b) && b * tile <= BLOCK_BYTES)
        .unwrap_or(1);
    Ok(vec![bl, bm, bn])
}

fn gcd(a: usize, b: usize) -> usize {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("a loop fusion")
    };
    let out = &step.output.1;
    if out.numel() == 0 {
        return Ok(());
    }
    let operands: Vec<&TensorType> = body.inputs().iter().map(|&v| body.type_of(v)).collect();
    if let Some(t) = operands
        .iter()
        .find(|t| t.shape != out.shape && !t.shape.is_empty())
    {
        return Err(format!(
            "{} on TPU: operands of the output's shape {out}, got {t}",
            step.label
        ));
    }
    let broadcast: Vec<bool> = operands.iter().map(|t| t.shape != out.shape).collect();
    let dtypes: Vec<DType> = operands
        .iter()
        .map(|t| t.dtype)
        .chain([out.dtype])
        .collect();
    let names: Vec<String> = operands
        .iter()
        .map(|t| format!("\"{}\"", t.dtype.full_name()))
        .collect();
    let mut value: HashMap<Var, usize> = body
        .inputs()
        .iter()
        .enumerate()
        .map(|(k, &v)| (v, k))
        .collect();
    let mut nodes = Vec::new();
    for n in body.nodes() {
        let args: Vec<usize> = n.inputs.iter().map(|v| value[v]).collect();
        nodes.push(node(&n.primitive, &args));
        value.insert(n.output, operands.len() + nodes.len() - 1);
    }
    let window = window(step, 0);
    let mut blocks = blocks(&out.shape, &dtypes)?;
    if let Some((_, start)) = window {
        let n = *out.shape.last().expect("a matrix");
        blocks[2] = block(gcd(n, start), 128, 2048)
            .ok_or_else(|| format!("{} on TPU: a window of no whole columns' block", step.label))?;
    }
    let leading = blocks.len() == 3 && blocks[0] == 1;
    let shown: Vec<String> = blocks
        .iter()
        .skip(usize::from(leading))
        .map(usize::to_string)
        .collect();
    let kernel = format!("loop_{}_{}", out.dtype, shown.join("x"));
    let spec = format!(
        "\"shape\": {:?}, \"inputs\": [{}], \"broadcast\": {broadcast:?}, \"nodes\": [{}], \"result\": {}, \"output\": \"{}\", \"blocks\": {:?}, \"window\": {}",
        out.shape,
        names.join(", "),
        nodes.join(", "),
        value[&body.outputs()[0]],
        out.dtype.full_name(),
        blocks,
        window.map_or("null".into(), |(n, start)| format!("[{n}, {start}]")),
    );
    let rows: Vec<*const u8> = inputs
        .iter()
        .copied()
        .chain([output.cast_const()])
        .collect();
    let launched = Launch {
        module: "lumen.codegen.tpu.elementwise",
        kernel,
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
    };
    launch(index, launched)
}
