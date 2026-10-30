//! TPU flash attention: checks that each attention operand and output is read or written as whole
//! tiles per head and encodes the forward and backward kernel specs.

use super::Tpu;
use crate::DType;
use crate::codegen::AttentionCodegen;
use crate::device::tpu::layout::Layout;
use crate::graph::{Graph, Node, Primitive, TensorType, Var};
use crate::graph_compiler::common::attention::{
    self, Access, Attention, Backward, Dropout, Score, Strides,
};
use crate::graph_compiler::elementwise;
use crate::kernels::attention::tpu::blocks;
use crate::tensor::contiguous_strides;

/// Lanes of a TPU vreg; head widths are whole multiples of it unless a head spans its rows.
const LANES: usize = 128;

/// An attention operand or output as the kernel addresses it.
struct Operand {
    /// Its position among the kernel's buffers: body inputs first, then outputs.
    row: usize,
    ty: TensorType,
    /// Where each head's elements sit in `ty`.
    acc: Access,
    /// Columns each head spans.
    width: usize,
}

/// The JSON view the kernel reads `op` through: its memory as `[slabs, rows, width]` and each
/// batch dim's step in slabs and column tiles.
///
/// `matrix` is the sequence length of a per-head matrix, `None` for a per-head vector such as the
/// log-sum-exp. Fails unless every head starts at a whole slab and a whole head-width column.
fn view(
    name: &str,
    op: &Operand,
    batch: &[usize],
    matrix: Option<usize>,
) -> Result<String, String> {
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
    let tiles = Layout::new(op.ty.dtype, shape)?.rows;
    // A sequence that divides the rows in whole tiles makes each slab one head's matrix.
    let rows = match matrix {
        Some(seq) if rows.is_multiple_of(seq) && seq.is_multiple_of(tiles) => seq,
        _ => rows,
    };

    let slab = rows * width;
    let slabs = op.ty.numel() / slab.max(1);
    let (row, col) = (op.acc.row, op.acc.col);

    let fits = match matrix {
        Some(_) => col == 1 && row == width,
        None => col == width,
    };
    if !fits {
        return failed("an operand whose rows are not its tiles' rows");
    }

    let unit = match matrix {
        Some(_) => op.width,
        None => 1,
    };

    if matrix.is_some() && !(op.width.is_multiple_of(LANES) || op.width == width) {
        return failed("a head dimension that is neither a multiple of 128 nor its rows' width");
    }

    // Splits an element offset into (slab, row, column).
    let split = |a: usize| (a / slab, (a % slab) / width, (a % slab) % width);
    let (l0, r0, c0) = split(op.acc.offset);
    let mut dims = Vec::new();

    for (&size, &(stride, div)) in batch.iter().zip(&op.acc.batch) {
        let (dl, dr, dc) = split(stride);
        dims.push((size, div, dl, dr, dc));
    }

    let heads: usize = batch.iter().product();
    // Check every head's start lands where the per-dim slab and column steps put it.
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

/// The JSON list of the transforms applied to the scores before the softmax.
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

/// The input position of `acc`'s base, or `n` when it is not a body input.
fn rows_of(body: &Graph, acc: &Access, n: usize) -> usize {
    body.inputs()
        .iter()
        .position(|&v| v == acc.base)
        .unwrap_or(n)
}

/// A contiguous access of `ty` with one head per leading `batch` index.
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

/// The operand `acc` reads, `width` columns per head.
fn operand(body: &Graph, acc: &Access, width: usize) -> Operand {
    let n = body.inputs().len();

    Operand {
        row: rows_of(body, acc, n),
        ty: body.type_of(acc.base).clone(),
        acc: acc.clone(),
        width,
    }
}

/// Body output `k` written at `acc`, `width` columns per head.
fn output(body: &Graph, k: usize, acc: Access, width: usize) -> Operand {
    let n = body.inputs().len();

    Operand {
        row: n + k,
        ty: body.type_of(body.outputs()[k]).clone(),
        acc,
        width,
    }
}

/// Fails for the dtypes the TPU attention kernels don't run: other than float32 or bfloat16.
fn unsupported(dtype: DType) -> Result<(), String> {
    if !matches!(dtype, DType::F32 | DType::BF16) {
        return Err(format!(
            "flash attention on TPU: operands of float32 or bfloat16, got {dtype}"
        ));
    }

    Ok(())
}

/// The kernel's kind and spec for the attention fusion `body` named `name`: `flash_attention` or
/// `flash_attention_backward`.
pub(crate) fn kernel(name: &str, body: &Graph) -> Result<(&'static str, String), String> {
    // The TPU kernels read their operands contiguous.
    let strides = Strides::default();

    match name {
        "flash_attention" => {
            let a = attention::of_body(body)
                .ok_or("flash attention on TPU: no attention in its body")?;
            Tpu::forward(body, &a, &strides)
        }
        _ => {
            let b = attention::backward_of_body(body)
                .ok_or("flash attention backward on TPU: no attention backward in its body")?;
            Tpu::backward(body, &b, &strides)
        }
    }
}

/// The Pallas flash-attention kernels are fixed; each fusion gets a JSON spec of how they read and write
/// its operands.
impl AttentionCodegen for Tpu {
    /// The kernel's kind and its JSON spec.
    type Code = (&'static str, String);

    /// The forward kernel's kind and spec for the attention fusion `body`, with an optional
    /// elementwise epilogue on the output and an optional log-sum-exp output.
    fn forward(body: &Graph, a: &Attention, _: &Strides) -> Result<Self::Code, String> {
        unsupported(a.dtype)?;

        if let Some(d) = &a.dropout {
            Self::dropout(d, a.dtype, a.sq, a.sk)?;
        }

        let views = [
            view("q", &operand(body, &a.q, a.h), &a.batch, Some(a.sq))?,
            view("k", &operand(body, &a.k, a.h), &a.batch, Some(a.sk))?,
            view("v", &operand(body, &a.v, a.hv), &a.batch, Some(a.sk))?,
        ];

        // The log-sum-exp, when produced, is the last output.
        let lse_k = a.lse.map(|_| body.outputs().len() - 1);
        let (mut raw, mut ends) = ("null".to_string(), ("null".to_string(), "null".to_string()));

        for k in (0..body.outputs().len()).filter(|&k| Some(k) != lse_k) {
            let chain = path(body, a.out.base, k)
                .ok_or("flash attention on TPU: an output written through other than its layout")?;

            let acc = written_at(body, &a.out, &a.batch, k)?.1;
            let shown = view(
                "the output",
                &output(body, k, acc, a.hv),
                &a.batch,
                Some(a.sq),
            )?;

            match epilogue(&chain) {
                true => ends = (shown, epilogue_spec(body, &chain, a.out.base)?),
                false => raw = shown,
            }
        }

        let lse = match lse_k {
            Some(k) => {
                let ty = body.type_of(body.outputs()[k]).clone();
                let acc = per_head(&ty, &a.batch);
                view("the log-sum-exp", &output(body, k, acc, 1), &a.batch, None)?
            }
            None => "null".into(),
        };

        let spec = format!(
            "\"kind\": \"forward\", \"batch\": {:?}, \"sq\": {}, \"sk\": {}, \"h\": {}, \"hv\": {}, \"blocks\": [{}, {}], \
             \"scores\": {}, \"causal\": {}, \"q\": {}, \"k\": {}, \"v\": {}, \"o\": {raw}, \"e\": {}, \"epilogue\": {}, \
             \"o_dtype\": \"{}\", \"lse\": {lse}",
            a.batch,
            a.sq,
            a.sk,
            a.h,
            a.hv,
            blocks(a.sq, a.sk).0,
            blocks(a.sq, a.sk).1,
            scores(&a.scores),
            a.causal.map_or("null".into(), |c| c.to_string()),
            views[0],
            views[1],
            views[2],
            ends.0,
            ends.1,
            a.out_dtype.full_name(),
        );

        Ok(("forward", spec))
    }

    /// The backward kernel's kind (`dq`, or `dkv` for dk and dv) and spec for the attention backward
    /// fusion `body`.
    fn backward(body: &Graph, b: &Backward, _: &Strides) -> Result<Self::Code, String> {
        unsupported(b.dtype)?;

        if let Some(d) = &b.dropout {
            Self::dropout(d, DType::F32, b.sq, b.sk)?;
        }

        let ins = [
            view("q", &operand(body, &b.q, b.h), &b.batch, Some(b.sq))?,
            view("k", &operand(body, &b.k, b.h), &b.batch, Some(b.sk))?,
            view("v", &operand(body, &b.v, b.hv), &b.batch, Some(b.sk))?,
            view("dO", &operand(body, &b.d_o, b.hv), &b.batch, Some(b.sq))?,
            view("the log-sum-exp", &operand(body, &b.lse, 1), &b.batch, None)?,
        ];

        // Delta is either a body input or rowsum(dO * o), which the kernel recomputes from o.
        let delta = match rows_of(body, &b.delta, usize::MAX) {
            usize::MAX => {
                let o = inside(body, &b.delta, &b.d_o, &b.batch, b.sq)?;
                format!(
                    "\"delta\": null, \"o\": {}",
                    view("o", &operand(body, &o, b.hv), &b.batch, Some(b.sq))?
                )
            }
            _ => format!(
                "\"delta\": {}",
                view("delta", &operand(body, &b.delta, 1), &b.batch, None)?
            ),
        };

        let kind = match (&b.dq_out, &b.dk_out, &b.dv_out) {
            (Some(_), None, None) => "dq",
            (None, Some(_), Some(_)) => "dkv",
            _ => return Err("flash attention backward on TPU: dq alone, or dk and dv".into()),
        };

        let out = |name: &str,
                   out: &Option<(usize, Access)>,
                   width: usize,
                   seq: usize|
         -> Result<String, String> {
            let (_, acc) = out.as_ref().expect("an output");
            let (k, acc) = written(body, acc, &b.batch)?;
            view(name, &output(body, k, acc, width), &b.batch, Some(seq))
        };

        let outs = match kind {
            "dq" => format!("\"dq\": {}", out("dq", &b.dq_out, b.h, b.sq)?),
            _ => format!(
                "\"dk\": {}, \"dv\": {}, \"group\": {}",
                out("dk", &b.dk_out, b.h, b.sk)?,
                out("dv", &b.dv_out, b.hv, b.sk)?,
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
            blocks(b.sq, b.sk).0,
            blocks(b.sq, b.sk).1,
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

    /// The Pallas attention kernels have no dropout yet.
    fn dropout(_: &Dropout, _: DType, _: usize, _: usize) -> Result<String, String> {
        Err("flash attention on TPU: dropout does not run on TPU yet".into())
    }
}

/// Where `o` is read, when `delta` is computed in the body as a dot of `dO` with `o`; fails otherwise.
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

    // Follows reshapes and transposes back to the value they view.
    let based = |mut v: Var| {
        let mut chain = Vec::new();

        while let Some(n) = producer(v).filter(|n| {
            matches!(
                n.primitive,
                Primitive::Reshape { .. } | Primitive::Transpose { .. }
            )
        }) {
            chain.push(n);
            v = n.inputs[0];
        }

        (v, chain)
    };

    let side = match (
        based(dot.inputs[0]).0 == d_o.base,
        based(dot.inputs[1]).0 == d_o.base,
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
    let (base, chain) = based(operand);
    let shape = body.type_of(operand).shape.clone();

    // Maps an element offset of the dot operand back through its transposes to its base.
    let through = |mut a: usize| {
        for n in &chain {
            if let Primitive::Transpose { permutation } = &n.primitive {
                let (from, to) = (
                    &body.type_of(n.output).shape,
                    &body.type_of(n.inputs[0]).shape,
                );

                let (fs, ts) = (contiguous_strides(from), contiguous_strides(to));
                a = (0..from.len())
                    .map(|d| ((a / fs[d]) % from[d]) * ts[permutation[d]])
                    .sum();
            }
        }
        a
    };

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
        through(address)
    };

    let offset = at(delta.offset, 0);
    // Fit each batch dim's step as (stride, period): heads within one period read the same place.
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
        base,
        offset,
        batch: fitted,
        row: at(delta.offset + delta.col, 0) - offset,
        col: at(delta.offset, 1) - offset,
    };

    // Check the first and last query row of every head against the fitted access.
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

/// Query heads per key/value head, from dk's innermost batch divisor.
fn group(dk: &Access) -> usize {
    dk.batch.last().map_or(1, |&(_, div)| div)
}

/// The nodes from `from` to body output `k`, when each is a reshape, a sum or elementwise.
fn path(body: &Graph, from: Var, k: usize) -> Option<Vec<&Node>> {
    let mut depends = vec![false; body.types.len()];
    depends[from] = true;
    for n in body.nodes() {
        depends[n.output] |= n.inputs.iter().any(|&v| depends[v]);
    }

    let producer = |v: Var| body.nodes().iter().find(|n| n.output == v);
    let (mut chain, mut v) = (Vec::new(), body.outputs()[k]);

    while v != from {
        let n = producer(v)?;
        let step = matches!(
            n.primitive,
            Primitive::Reshape { .. } | Primitive::ReduceSum { .. }
        ) || elementwise(&n.primitive);

        if !step {
            return None;
        }

        chain.push(n);
        v = *n.inputs.iter().find(|&&x| depends[x])?;
    }

    chain.reverse();
    Some(chain)
}

fn epilogue(chain: &[&Node]) -> bool {
    chain.iter().any(|n| {
        !matches!(
            n.primitive,
            Primitive::Reshape { .. } | Primitive::ReduceSum { .. } | Primitive::Cast { .. }
        )
    })
}

/// The JSON epilogue applying `chain` to the attention output `from`; it may read only scalar body
/// inputs besides values computed from them.
fn epilogue_spec(body: &Graph, chain: &[&Node], from: Var) -> Result<String, String> {
    let failed = || Err("flash attention on TPU: an epilogue reading other than scalars".into());
    let nodes = body.nodes();
    let producer = |v: Var| nodes.iter().position(|n| n.output == v);

    let on_path: Vec<Var> = std::iter::once(from)
        .chain(chain.iter().map(|n| n.output))
        .collect();

    let mut needed = vec![false; nodes.len()];
    let mut stack: Vec<Var> = chain
        .iter()
        .flat_map(|n| n.inputs.iter().copied())
        .filter(|v| !on_path.contains(v))
        .collect();

    let mut scalars: Vec<Var> = Vec::new();
    // Collect the non-path values the chain reads, failing on any non-scalar body input.
    while let Some(v) = stack.pop() {
        match producer(v) {
            Some(i) if !needed[i] => {
                needed[i] = true;
                stack.extend(&nodes[i].inputs);
            }
            Some(_) => {}
            None if body.type_of(v).shape.is_empty() => {
                if !scalars.contains(&v) {
                    scalars.push(v);
                }
            }
            None => return failed(),
        }
    }

    let mut value: std::collections::HashMap<Var, usize> =
        std::collections::HashMap::from([(from, 0)]);

    for (k, &v) in scalars.iter().enumerate() {
        value.insert(v, k + 1);
    }

    let mut shown = Vec::new();

    for (i, n) in nodes.iter().enumerate() {
        let wanted = needed[i] || chain.iter().any(|m| m.output == n.output);
        if !wanted {
            continue;
        }

        // Reshapes don't change a block's elements, so they alias their operand.
        if matches!(n.primitive, Primitive::Reshape { .. }) {
            let x = value[&n.inputs[0]];
            value.insert(n.output, x);
            continue;
        }

        let args: Vec<usize> = n.inputs.iter().map(|v| value[v]).collect();
        shown.push(crate::codegen::tpu::node(&n.primitive, &args));
        value.insert(n.output, scalars.len() + shown.len());
    }

    let rows: Vec<String> = scalars
        .iter()
        .map(|&v| {
            let row = body
                .inputs()
                .iter()
                .position(|&u| u == v)
                .expect("an input");
            format!("[{row}, \"{}\"]", body.type_of(v).dtype.full_name())
        })
        .collect();

    let end = chain.last().expect("a step").output;

    Ok(format!(
        "{{\"scalars\": [{}], \"nodes\": [{}], \"result\": {}}}",
        rows.join(", "),
        shown.join(", "),
        value[&end]
    ))
}

/// The first output `acc` reaches through only reshapes, sums and casts, with `acc` mapped onto it.
fn written(body: &Graph, acc: &Access, batch: &[usize]) -> Result<(usize, Access), String> {
    let k = (0..body.outputs().len())
        .find(|&k| path(body, acc.base, k).is_some_and(|c| !epilogue(&c)))
        .ok_or("flash attention on TPU: an output written through other than reshapes and sums")?;

    written_at(body, acc, batch, k)
}

/// `acc` mapped through the reductions on the path to body output `k`, failing unless every head
/// still lands at a regular per-dim stride.
fn written_at(
    body: &Graph,
    acc: &Access,
    batch: &[usize],
    k: usize,
) -> Result<(usize, Access), String> {
    let failed =
        || Err("flash attention on TPU: an output written through other than its layout".into());

    let Some(chain) = path(body, acc.base, k) else {
        return failed();
    };

    // Maps an element offset through each reduce_sum on the path, dropping the reduced axes.
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

    // Fit each batch dim's step as (stride, period): heads within one period write the same place.
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
