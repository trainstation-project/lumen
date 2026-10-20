use crate::DType;
#[cfg(lumen_tpu_linked)]
use crate::graph::Primitive;
use crate::graph::{Graph, TensorType, Var};
use crate::graph_compiler::attention::{self, Access, Score};
#[cfg(lumen_tpu_linked)]
use crate::graph_compiler::plan::Step;
#[cfg(lumen_tpu_linked)]
use crate::ops::tpu::{Launch, launch};
use crate::tensor::contiguous_strides;

const LANES: usize = 128;

const BLOCK: usize = 128;

struct Operand {
    row: usize,
    ty: TensorType,
    acc: Access,
    width: usize,
}

fn view(op: &Operand, batch: &[usize], matrix: bool) -> Result<String, String> {
    let shape = &op.ty.shape;
    let failed = |why: &str| Err(format!("flash attention on TPU: {why}"));
    if shape.len() < 2 {
        return failed("an operand of fewer than two dimensions");
    }
    let (rows, width) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let slab = rows * width;
    let slabs = op.ty.numel() / slab.max(1);
    let (row, col) = (op.acc.row, op.acc.col);
    let fits = match matrix {
        true => col == 1 && row == width,
        false => col == width,
    };
    if !fits {
        return failed("an operand whose rows are not its tiles' rows");
    }
    let unit = match matrix {
        true => op.width,
        false => 1,
    };
    if matrix && !(op.width.is_multiple_of(LANES) || op.width == width) {
        return failed("a head dimension that is neither a multiple of 128 nor its rows' width");
    }
    let split = |a: usize| (a / slab, (a % slab) / width, (a % slab) % width);
    let (l0, r0, c0) = split(op.acc.offset);
    let mut dims = Vec::new();
    for (&size, &(stride, div)) in batch.iter().zip(&op.acc.batch) {
        let (dl, dr, dc) = split(stride);
        dims.push((size, div, dl, dr, dc));
    }
    let heads: usize = batch.iter().product();
    for g in 0..heads {
        let (mut rest, mut address, mut at) = (g, op.acc.offset, (l0, r0, c0));
        for (k, &(size, div, dl, dr, dc)) in dims.iter().enumerate().rev() {
            let idx = (rest % size) / div;
            rest /= size;
            address += idx * op.acc.batch[k].0;
            at = (at.0 + idx * dl, at.1 + idx * dr, at.2 + idx * dc);
        }
        let whole = split(address) == at && at.1 == 0 && at.0 < slabs;
        if !whole || !at.2.is_multiple_of(unit) || at.2 + unit > width {
            return failed("heads that are not whole tiles of their operand");
        }
    }
    let shown: Vec<String> = dims
        .iter()
        .map(|&(size, div, dl, _, dc)| format!("[{size}, {div}, {dl}, {}]", dc / unit))
        .collect();
    Ok(format!(
        "{{\"row\": {}, \"dtype\": \"{}\", \"view\": [{slabs}, {rows}, {width}], \"at\": [{l0}, {}], \"dims\": [{}]}}",
        op.row,
        op.ty.dtype.full_name(),
        c0 / unit,
        shown.join(", ")
    ))
}

fn scores(scores: &[Score]) -> String {
    let shown: Vec<String> = scores
        .iter()
        .map(|s| match s {
            Score::Round(d) => format!("[\"round\", \"{}\"]", d.full_name()),
            Score::Mul(c, d) => format!("[\"mul\", {c:?}, \"{}\"]", d.full_name()),
        })
        .collect();
    format!("[{}]", shown.join(", "))
}

fn rows_of(body: &Graph, acc: &Access, n: usize) -> usize {
    body.inputs()
        .iter()
        .position(|&v| v == acc.base)
        .unwrap_or(n)
}

fn per_head(ty: &TensorType, batch: &[usize]) -> Access {
    let strides = contiguous_strides(&ty.shape);
    let nb = batch.len();
    Access {
        base: 0,
        offset: 0,
        batch: strides[..nb].iter().map(|&s| (s, 1)).collect(),
        row: 0,
        col: strides[nb],
    }
}

fn block(n: usize) -> usize {
    match n.is_multiple_of(BLOCK) {
        true => BLOCK,
        false => n,
    }
}

#[cfg(lumen_tpu_linked)]
fn run(index: usize, kind: &str, spec: String, rows: &[*const u8]) -> Result<(), String> {
    let module = match kind {
        "forward" => "lumen.ops.attention.tpu.forward",
        _ => "lumen.ops.attention.tpu.backward",
    };
    let launched = Launch {
        module,
        kernel: format!("flash_attention_{kind}"),
        spec,
        rows,
        scalars: &[],
        host: &[],
    };
    launch(index, launched)
}

#[cfg(lumen_tpu_linked)]
pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let Primitive::Fusion { name, body, .. } = &step.primitive else {
        unreachable!("an attention fusion")
    };
    let n = step.inputs.len();
    let rows: Vec<*const u8> = inputs[..n]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(inputs[n..].iter().copied())
        .collect();
    let (kind, spec) = match name.as_str() {
        "flash_attention" => forward(body)?,
        _ => backward(body)?,
    };
    run(index, kind, spec, &rows)
}

fn operand(body: &Graph, acc: &Access, width: usize) -> Operand {
    let n = body.inputs().len();
    Operand {
        row: rows_of(body, acc, n),
        ty: body.type_of(acc.base).clone(),
        acc: acc.clone(),
        width,
    }
}

fn output(body: &Graph, k: usize, acc: Access, width: usize) -> Operand {
    let n = body.inputs().len();
    Operand {
        row: n + k,
        ty: body.type_of(body.outputs()[k]).clone(),
        acc,
        width,
    }
}

fn unsupported(dropout: bool, dtype: DType) -> Result<(), String> {
    if dropout {
        return Err("flash attention on TPU: dropout does not run on TPU yet".into());
    }
    if !matches!(dtype, DType::F32 | DType::BF16) {
        return Err(format!(
            "flash attention on TPU: operands of float32 or bfloat16, got {dtype}"
        ));
    }
    Ok(())
}

pub(crate) fn forward(body: &Graph) -> Result<(&'static str, String), String> {
    let a = attention::of_body(body).ok_or("flash attention on TPU: no attention in its body")?;
    unsupported(a.dropout.is_some(), a.dtype)?;
    let views = [
        view(&operand(body, &a.q, a.h), &a.batch, true)?,
        view(&operand(body, &a.k, a.h), &a.batch, true)?,
        view(&operand(body, &a.v, a.hv), &a.batch, true)?,
        view(&output(body, 0, a.out.clone(), a.hv), &a.batch, true)?,
    ];
    let lse = match a.lse {
        Some(_) => {
            let k = body.outputs().len() - 1;
            let ty = body.type_of(body.outputs()[k]).clone();
            let acc = per_head(&ty, &a.batch);
            view(&output(body, k, acc, 1), &a.batch, false)?
        }
        None => "null".into(),
    };
    let spec = format!(
        "\"kind\": \"forward\", \"batch\": {:?}, \"sq\": {}, \"sk\": {}, \"h\": {}, \"hv\": {}, \"blocks\": [{}, {}], \
         \"scores\": {}, \"causal\": {}, \"q\": {}, \"k\": {}, \"v\": {}, \"o\": {}, \"lse\": {lse}",
        a.batch,
        a.sq,
        a.sk,
        a.h,
        a.hv,
        block(a.sq),
        block(a.sk),
        scores(&a.scores),
        a.causal.map_or("null".into(), |c| c.to_string()),
        views[0],
        views[1],
        views[2],
        views[3],
    );
    Ok(("forward", spec))
}

pub(crate) fn backward(body: &Graph) -> Result<(&'static str, String), String> {
    let b = attention::backward_of_body(body)
        .ok_or("flash attention backward on TPU: no attention backward in its body")?;
    unsupported(b.dropout.is_some(), b.dtype)?;
    let ins = [
        view(&operand(body, &b.q, b.h), &b.batch, true)?,
        view(&operand(body, &b.k, b.h), &b.batch, true)?,
        view(&operand(body, &b.v, b.hv), &b.batch, true)?,
        view(&operand(body, &b.d_o, b.hv), &b.batch, true)?,
        view(&operand(body, &b.lse, 1), &b.batch, false)?,
        view(&operand(body, &b.delta, 1), &b.batch, false)?,
    ];
    let outputs: Vec<Var> = body.outputs().to_vec();
    let at = |k: usize| outputs.iter().position(|&o| o == k).expect("an output");
    let kind = match (&b.dq_out, &b.dk_out, &b.dv_out) {
        (Some(_), None, None) => "dq",
        (None, Some(_), Some(_)) => "dkv",
        _ => return Err("flash attention backward on TPU: dq alone, or dk and dv".into()),
    };
    let written = |out: &Option<(usize, Access)>, width: usize| -> Result<String, String> {
        let (_, acc) = out.as_ref().expect("an output");
        view(
            &output(body, at(acc.base), acc.clone(), width),
            &b.batch,
            true,
        )
    };
    let outs = match kind {
        "dq" => format!("\"dq\": {}", written(&b.dq_out, b.h)?),
        _ => format!(
            "\"dk\": {}, \"dv\": {}, \"group\": {}",
            written(&b.dk_out, b.h)?,
            written(&b.dv_out, b.hv)?,
            group(&b.k, &b.batch)
        ),
    };
    let spec = format!(
        "\"kind\": \"{kind}\", \"batch\": {:?}, \"sq\": {}, \"sk\": {}, \"h\": {}, \"hv\": {}, \"blocks\": [{}, {}], \
         \"scores\": {}, \"causal\": {}, \"ds_scale\": {:?}, \"q\": {}, \"k\": {}, \"v\": {}, \"do\": {}, \
         \"lse\": {}, \"delta\": {}, {outs}",
        b.batch,
        b.sq,
        b.sk,
        b.h,
        b.hv,
        block(b.sq),
        block(b.sk),
        scores(&b.scores),
        b.causal.map_or("null".into(), |c| c.to_string()),
        b.ds_scale,
        ins[0],
        ins[1],
        ins[2],
        ins[3],
        ins[4],
        ins[5],
    );
    Ok((kind, spec))
}

fn group(k: &Access, batch: &[usize]) -> usize {
    match (k.batch.last(), batch.last()) {
        (Some(&(_, div)), Some(&size)) if size.is_multiple_of(div) => div,
        _ => 1,
    }
}
