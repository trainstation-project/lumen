use crate::DType;
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::graph_compiler::attention::{self, Access, Score};
#[cfg(lumen_tpu_linked)]
use crate::graph_compiler::plan::Step;
#[cfg(lumen_tpu_linked)]
use crate::ops::tpu::{Launch, launch};
use crate::tensor::contiguous_strides;

const LANES: usize = 128;

const QUERIES: usize = 512;

const KEYS: usize = 1024;

struct Operand {
    row: usize,
    ty: TensorType,
    acc: Access,
    width: usize,
}

fn view(name: &str, op: &Operand, batch: &[usize], matrix: bool) -> Result<String, String> {
    let shape = &op.ty.shape;
    let failed = |why: &str| {
        Err(format!(
            "flash attention on TPU: {name}, read from {} (offset {}, rows {} apart, columns {} apart): {why}",
            op.ty, op.acc.offset, op.acc.row, op.acc.col
        ))
    };
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

fn block(n: usize, cap: usize) -> usize {
    (1..=cap / LANES)
        .rev()
        .map(|k| k * LANES)
        .find(|&b| n.is_multiple_of(b))
        .unwrap_or(n)
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
        view("q", &operand(body, &a.q, a.h), &a.batch, true)?,
        view("k", &operand(body, &a.k, a.h), &a.batch, true)?,
        view("v", &operand(body, &a.v, a.hv), &a.batch, true)?,
        view(
            "the output",
            &output(body, 0, written(body, &a.out, &a.batch)?.1, a.hv),
            &a.batch,
            true,
        )?,
    ];
    let lse = match a.lse {
        Some(_) => {
            let k = body.outputs().len() - 1;
            let ty = body.type_of(body.outputs()[k]).clone();
            let acc = per_head(&ty, &a.batch);
            view("the log-sum-exp", &output(body, k, acc, 1), &a.batch, false)?
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
        block(a.sq, QUERIES),
        block(a.sk, KEYS),
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
        view("q", &operand(body, &b.q, b.h), &b.batch, true)?,
        view("k", &operand(body, &b.k, b.h), &b.batch, true)?,
        view("v", &operand(body, &b.v, b.hv), &b.batch, true)?,
        view("dO", &operand(body, &b.d_o, b.hv), &b.batch, true)?,
        view(
            "the log-sum-exp",
            &operand(body, &b.lse, 1),
            &b.batch,
            false,
        )?,
    ];
    let delta = match rows_of(body, &b.delta, usize::MAX) {
        usize::MAX => {
            let o = inside(body, &b.delta, &b.d_o, &b.batch, b.sq)?;
            format!(
                "\"delta\": null, \"o\": {}",
                view("o", &operand(body, &o, b.hv), &b.batch, true)?
            )
        }
        _ => format!(
            "\"delta\": {}",
            view("delta", &operand(body, &b.delta, 1), &b.batch, false)?
        ),
    };
    let kind = match (&b.dq_out, &b.dk_out, &b.dv_out) {
        (Some(_), None, None) => "dq",
        (None, Some(_), Some(_)) => "dkv",
        _ => return Err("flash attention backward on TPU: dq alone, or dk and dv".into()),
    };
    let out = |name: &str, out: &Option<(usize, Access)>, width: usize| -> Result<String, String> {
        let (_, acc) = out.as_ref().expect("an output");
        let (k, acc) = written(body, acc, &b.batch)?;
        view(name, &output(body, k, acc, width), &b.batch, true)
    };
    let outs = match kind {
        "dq" => format!("\"dq\": {}", out("dq", &b.dq_out, b.h)?),
        _ => format!(
            "\"dk\": {}, \"dv\": {}, \"group\": {}",
            out("dk", &b.dk_out, b.h)?,
            out("dv", &b.dv_out, b.hv)?,
            group(&written(body, &b.dk_out.as_ref().expect("dk").1, &b.batch)?.1)
        ),
    };
    let spec = format!(
        "\"kind\": \"{kind}\", \"batch\": {:?}, \"sq\": {}, \"sk\": {}, \"h\": {}, \"hv\": {}, \"blocks\": [{}, {}], \
         \"scores\": {}, \"causal\": {}, \"ds_scale\": {:?}, \"q\": {}, \"k\": {}, \"v\": {}, \"do\": {}, \
         \"lse\": {}, {delta}, {outs}",
        b.batch,
        b.sq,
        b.sk,
        b.h,
        b.hv,
        block(b.sq, QUERIES),
        block(b.sk, KEYS),
        scores(&b.scores),
        b.causal.map_or("null".into(), |c| c.to_string()),
        b.ds_scale,
        ins[0],
        ins[1],
        ins[2],
        ins[3],
        ins[4],
    );
    Ok((kind, spec))
}

fn inside(
    body: &Graph,
    delta: &Access,
    d_o: &Access,
    batch: &[usize],
    sq: usize,
) -> Result<Access, String> {
    let failed =
        || Err("flash attention backward on TPU: delta neither an operand nor rowsum(dO o)".into());
    let producer = |v: Var| body.nodes().iter().find(|n| n.output == v);
    let Some(dot) = producer(delta.base) else {
        return failed();
    };
    let Primitive::DotGeneral {
        lhs_batch,
        rhs_batch,
        lhs_contracting,
        rhs_contracting,
        ..
    } = &dot.primitive
    else {
        return failed();
    };
    let based = |mut v: Var| {
        while let Some(n) = producer(v).filter(|n| matches!(n.primitive, Primitive::Reshape { .. }))
        {
            v = n.inputs[0];
        }
        v
    };
    let side = match (
        based(dot.inputs[0]) == d_o.base,
        based(dot.inputs[1]) == d_o.base,
    ) {
        (true, false) => 1,
        (false, true) => 0,
        _ => return failed(),
    };
    let (dims, contracting) = match side {
        0 => (lhs_batch, lhs_contracting),
        _ => (rhs_batch, rhs_contracting),
    };
    let operand = dot.inputs[side];
    let shape = body.type_of(operand).shape.clone();
    let out = body.type_of(dot.output).shape.clone();
    if contracting.len() != 1 || out.len() != dims.len() {
        return failed();
    }
    let (strides, out_strides) = (contiguous_strides(&shape), contiguous_strides(&out));
    let at = |a: usize, h: usize| {
        let mut address = h * strides[contracting[0]];
        for (k, &d) in dims.iter().enumerate() {
            address += ((a / out_strides[k]) % out[k]) * strides[d];
        }
        address
    };
    let offset = at(delta.offset, 0);
    let mut fitted = Vec::new();
    for (d, &(stride, div)) in delta.batch.iter().enumerate() {
        let size = batch[d];
        let of = |i: usize| at(delta.offset + (i / div) * stride, 0);
        let period = (1..size).find(|&i| of(i) != of(0)).unwrap_or(size);
        let scaled = if period < size { of(period) - of(0) } else { 0 };
        if (0..size).any(|i| of(i) != of(0) + (i / period) * scaled) {
            return failed();
        }
        fitted.push((scaled, period));
    }
    let o = Access {
        base: based(operand),
        offset,
        batch: fitted,
        row: at(delta.offset + delta.col, 0) - offset,
        col: at(delta.offset, 1) - offset,
    };
    let heads: usize = batch.iter().product();
    for g in 0..heads {
        for q in [0, sq - 1] {
            let (mut rest, mut from, mut to) =
                (g, delta.offset + q * delta.col, o.offset + q * o.row);
            for d in (0..batch.len()).rev() {
                let idx = rest % batch[d];
                rest /= batch[d];
                from += (idx / delta.batch[d].1) * delta.batch[d].0;
                to += (idx / o.batch[d].1) * o.batch[d].0;
            }
            if at(from, 0) != to {
                return failed();
            }
        }
    }
    Ok(o)
}

fn group(dk: &Access) -> usize {
    dk.batch.last().map_or(1, |&(_, div)| div)
}

fn written(body: &Graph, acc: &Access, batch: &[usize]) -> Result<(usize, Access), String> {
    let failed = || {
        Err("flash attention on TPU: an output written through other than reshapes and sums".into())
    };
    let producer = |v: Var| body.nodes().iter().find(|n| n.output == v);
    let mut found = None;
    for (k, &o) in body.outputs().iter().enumerate() {
        let (mut chain, mut v) = (Vec::new(), o);
        while v != acc.base {
            match producer(v) {
                Some(n)
                    if matches!(
                        n.primitive,
                        Primitive::Reshape { .. }
                            | Primitive::ReduceSum { .. }
                            | Primitive::Cast { .. }
                    ) =>
                {
                    chain.push(n);
                    v = n.inputs[0];
                }
                _ => break,
            }
        }
        if v == acc.base {
            chain.reverse();
            found = Some((k, chain));
            break;
        }
    }
    let Some((k, chain)) = found else {
        return failed();
    };
    let f = |mut a: usize| {
        for n in &chain {
            if let Primitive::ReduceSum { axes, .. } = &n.primitive {
                let from = &body.type_of(n.inputs[0]).shape;
                let to = contiguous_strides(&body.type_of(n.output).shape);
                let mut out = 0;
                let mut kept = 0;
                for (d, s) in contiguous_strides(from).iter().enumerate() {
                    let idx = (a / s) % from[d];
                    if !axes.contains(&d) {
                        out += idx * to[kept];
                        kept += 1;
                    }
                }
                a = out;
            }
        }
        a
    };
    let offset = f(acc.offset);
    let step = |s: usize| {
        if s == 0 {
            0
        } else {
            f(acc.offset + s) - offset
        }
    };
    let mut dims = Vec::new();
    for (d, &(stride, div)) in acc.batch.iter().enumerate() {
        let size = batch[d];
        let at = |i: usize| f(acc.offset + (i / div) * stride);
        let period = (1..size).find(|&i| at(i) != at(0)).unwrap_or(size);
        let scaled = if period < size { at(period) - at(0) } else { 0 };
        if (0..size).any(|i| at(i) != at(0) + (i / period) * scaled) {
            return failed();
        }
        dims.push((scaled, period));
    }
    let mapped = Access {
        base: body.outputs()[k],
        offset,
        batch: dims,
        row: step(acc.row),
        col: step(acc.col),
    };
    let heads: usize = batch.iter().product();
    for g in 0..heads {
        let (mut rest, mut before, mut after) = (g, acc.offset, mapped.offset);
        for d in (0..batch.len()).rev() {
            let idx = rest % batch[d];
            rest /= batch[d];
            before += (idx / acc.batch[d].1) * acc.batch[d].0;
            after += (idx / mapped.batch[d].1) * mapped.batch[d].0;
        }
        if f(before) != after {
            return failed();
        }
    }
    Ok((k, mapped))
}
