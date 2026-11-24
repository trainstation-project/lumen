//! TPU flash attention: the Pallas forward and backward programs (`forward_kernel.py`, `backward_kernel.py` and their
//! helpers in `utils.py`, beside this file) and the specs generated per fusion for them: each attention
//! operand and output checked to be read or written as whole tiles per head, and the blocks chosen
//! (`config.rs`); and launching an attention fusion's program (`encode`).

mod config;

use super::AttentionCodegen;
use crate::DType;
use crate::codegen::devices::tpu::Tpu;
#[cfg(lumen_tpu_linked)]
use crate::codegen::devices::tpu::{Kernel, Rows};
use crate::device::tpu::layout::Layout;
use crate::graph::{Graph, Node, Primitive, TensorType, Var};
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::elementwise;
use crate::graph_compiler::passes::attention::{
    self, Access, Attention, Backward, Dropout, Score, Strides,
};
#[cfg(lumen_tpu_linked)]
use crate::plan::StepBase;
use crate::tensor::contiguous_strides;
use config::blocks;

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
///
/// With `shared` (the forward kernel's outputs), heads narrower than 128 lanes may share a 128-lane block (its
/// `pack`): consecutive heads fill each block's lanes in order, as the kernel writes them one after another.
fn view(
    name: &str,
    op: &Operand,
    batch: &[usize],
    matrix: Option<usize>,
    shared: bool,
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

    let narrow = op.width < LANES && LANES.is_multiple_of(op.width) && width.is_multiple_of(LANES);

    let pack = match matrix {
        Some(_) if shared && narrow => LANES / op.width,
        _ => 1,
    };

    if matrix.is_some() && pack == 1 && !(op.width.is_multiple_of(LANES) || op.width == width) {
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

    if !heads.is_multiple_of(pack) {
        return failed("heads that don't fill their shared 128-lane blocks");
    }

    // Each run of `pack` heads' (slab, 128-lane block).
    let mut blocks: Vec<(usize, usize)> = Vec::new();

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

        if pack == 1 {
            continue;
        }

        // A run's first head opens a block no other run writes; the rest join it at the next lanes.
        let block = (at.0, at.2 / LANES);
        let first = g.is_multiple_of(pack);
        let joins = blocks.last() == Some(&block);
        let opens = first && !blocks.contains(&block);

        if (at.2 % LANES) / unit != g % pack || (first && !opens) || (!first && !joins) {
            return failed("heads that don't fill their shared 128-lane blocks in order");
        }

        if first {
            blocks.push(block);
        }
    }

    let shown: Vec<String> = dims
        .iter()
        .map(|&(size, div, dl, _, dc)| format!("[{size}, {div}, {dl}, {}]", dc / unit))
        .collect();

    Ok(format!(
        "{{\"row\": {}, \"dtype\": \"{}\", \"view\": [{slabs}, {rows}, {width}], \"at\": [{l0}, {}], \"dims\": [{}], \
         \"pack\": {pack}}}",
        op.row,
        op.ty.dtype.full_name(),
        c0 / unit,
        shown.join(", ")
    ))
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

        // One query a head: the decode kernel, every head's query a row of one block.
        if a.sq == 1 {
            return decode(body, a);
        }

        if a.offset.is_some() {
            return Err(
                "flash attention on TPU: a runtime causal offset with more than one query".into(),
            );
        }

        if let Some(d) = &a.dropout {
            Self::dropout(d, a.dtype, a.sq, a.sk)?;
        }

        let views = [
            view("q", &operand(body, &a.q, a.h), &a.batch, Some(a.sq), false)?,
            view("k", &operand(body, &a.k, a.h), &a.batch, Some(a.sk), false)?,
            view("v", &operand(body, &a.v, a.hv), &a.batch, Some(a.sk), false)?,
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
                true,
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
                view(
                    "the log-sum-exp",
                    &output(body, k, acc, 1),
                    &a.batch,
                    None,
                    false,
                )?
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
            Self::score(&a.scores),
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
            view("q", &operand(body, &b.q, b.h), &b.batch, Some(b.sq), false)?,
            view("k", &operand(body, &b.k, b.h), &b.batch, Some(b.sk), false)?,
            view("v", &operand(body, &b.v, b.hv), &b.batch, Some(b.sk), false)?,
            view(
                "dO",
                &operand(body, &b.d_o, b.hv),
                &b.batch,
                Some(b.sq),
                false,
            )?,
            view(
                "the log-sum-exp",
                &operand(body, &b.lse, 1),
                &b.batch,
                None,
                false,
            )?,
        ];

        // Delta is either a body input or rowsum(dO * o), which the kernel recomputes from o.
        let delta = match rows_of(body, &b.delta, usize::MAX) {
            usize::MAX => {
                let o = inside(body, &b.delta, &b.d_o, &b.batch, b.sq)?;

                format!(
                    "\"delta\": null, \"o\": {}",
                    view("o", &operand(body, &o, b.hv), &b.batch, Some(b.sq), false)?
                )
            }
            _ => format!(
                "\"delta\": {}",
                view("delta", &operand(body, &b.delta, 1), &b.batch, None, false)?
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
            view(
                name,
                &output(body, k, acc, width),
                &b.batch,
                Some(seq),
                false,
            )
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
            Self::score(&b.scores),
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

    /// The JSON list of the transforms applied to the scores before the softmax.
    fn score(scores: &[Score]) -> String {
        let shown: Vec<String> = scores
            .iter()
            .map(|s| match s {
                Score::Round(d) => format!("[\"round\", \"{}\"]", d.full_name()),
                Score::Mul(c, d) => format!("[\"mul\", {c:?}, \"{}\"]", d.full_name()),
            })
            .collect();

        format!("[{}]", shown.join(", "))
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
        shown.push(crate::codegen::devices::tpu::node(&n.primitive, &args));
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

/// The kernel of an attention fusion step, binding its inputs, then its output, then any extra outputs.
#[cfg(lumen_tpu_linked)]
pub(crate) fn step_kernel(step: &StepBase) -> Result<Kernel, String> {
    let Primitive::Fusion { name, body, .. } = &step.primitive else {
        unreachable!("an attention fusion")
    };

    // The kernels read each operand's heads and head dim (its last two dims) contiguous.
    for (strides, (_, ty)) in step.input_strides().iter().zip(&step.inputs) {
        let rank = ty.shape.len();
        let contiguous = contiguous_strides(&ty.shape);

        if rank > 0 && strides[rank.saturating_sub(2)..] != contiguous[rank.saturating_sub(2)..] {
            return Err(format!(
                "{}: flash attention on TPU reads its operands' last two dims (heads, head dim) \
                 contiguous; one has strides {strides:?}",
                step.label
            ));
        }
    }

    let (kind, spec) = kernel(name, body)?;

    let module = match kind {
        "forward" => "lumen.codegen.attention.tpu.forward_kernel",
        "decode" => "lumen.codegen.attention.tpu.decode_kernel",
        _ => "lumen.codegen.attention.tpu.backward_kernel",
    };

    Ok(Kernel::new(
        module,
        format!("flash_attention_{kind}"),
        spec,
        Rows::Split,
    ))
}

/// Block of keys the decode kernel takes at a time: the largest multiple of 128 up to 512 dividing `sk`, else `sk`.
fn decode_block(sk: usize) -> usize {
    (1..=4)
        .rev()
        .map(|i| i * LANES)
        .find(|&b| b <= sk && sk.is_multiple_of(b))
        .unwrap_or(sk)
}

/// The decode kernel's kind and spec for attention `a` of fusion `body`, one query a head (JAX's paged attention's
/// form): its batch dims `[batch, heads]`, the query and output each batch index's `[heads, head dim]` matrix (the
/// output may instead be its `[1, heads * head dim]` row, `flat`, as the output projection reads it), the
/// keys and values heads-major (`[batch, kv heads, keys, head dim]`, a query head reading kv head `head / group`), no
/// epilogue or log-sum-exp; the causal diagonal plus a runtime offset (`offset`, an input's row) when it has one.
fn decode(body: &Graph, a: &Attention) -> Result<(&'static str, String), String> {
    let failed = |why: &str| Err(format!("attention decode on TPU: {why}"));

    let &[b, n] = a.batch.as_slice() else {
        return failed("batch dims other than [batch, heads]");
    };

    // Whether `acc` reads each batch index's `[heads, width]` matrix, the heads its rows.
    let rows = |acc: &Access, width: usize| {
        acc.offset == 0
            && acc.col == 1
            && acc.batch[1] == (width, 1)
            && (b == 1 || acc.batch[0] == (n * width, 1))
    };

    // Whether `acc` reads the keys or values heads-major, returning their group (query heads a kv head).
    let heads_major = |acc: &Access, width: usize| {
        let (stride, group) = acc.batch[1];
        let kv = n / group;

        let fits = acc.offset == 0
            && acc.col == 1
            && acc.row == width
            && stride == a.sk * width
            && (b == 1 || acc.batch[0] == (kv * a.sk * width, 1));

        fits.then_some(group)
    };

    // Its buffer's last two dims, whose tiles the kernel reads.
    let matrix = |v: Var| {
        body.type_of(v)
            .shape
            .iter()
            .rev()
            .take(2)
            .rev()
            .copied()
            .collect::<Vec<_>>()
    };

    if !rows(&a.q, a.h) || matrix(a.q.base) != [n, a.h] {
        return failed("a query other than each batch index's [heads, head dim] matrix");
    }

    let (Some(group), Some(v_group)) = (heads_major(&a.k, a.h), heads_major(&a.v, a.hv)) else {
        return failed("keys or values other than heads-major [batch, kv heads, keys, head dim]");
    };

    if group != v_group || a.lse.is_some() || body.outputs().len() != 1 {
        return failed("a log-sum-exp, extra outputs or keys and values of different groups");
    }

    let chain = path(body, a.out.base, 0)
        .ok_or("attention decode on TPU: an output written through other than its layout")?;

    if epilogue(&chain) {
        return failed("an epilogue");
    }

    let (_, out) = written_at(body, &a.out, &a.batch, 0)?;

    // Each batch index's heads as the rows of a matrix, or side by side in one row (`flat`), as a projection reads them.
    let flat = matrix(body.outputs()[0]) == [1, n * a.hv];

    if !rows(&out, a.hv) || !(flat || matrix(body.outputs()[0]) == [n, a.hv]) {
        return failed(
            "an output other than each batch index's [heads, head dim] matrix or [1, heads * head dim] row",
        );
    }

    let n_inputs = body.inputs().len();
    let row = |acc: &Access| rows_of(body, acc, n_inputs);

    let offset = match a.offset {
        Some(v) => body
            .inputs()
            .iter()
            .position(|&u| u == v)
            .map_or("null".into(), |r| r.to_string()),
        None => "null".into(),
    };

    let spec = format!(
        "\"kind\": \"decode\", \"b\": {b}, \"n\": {n}, \"group\": {group}, \"sk\": {}, \"h\": {}, \"hv\": {},          \"bk\": {}, \"scores\": {}, \"causal\": {}, \"offset\": {offset}, \"q\": {}, \"k\": {}, \"v\": {}, \"o\": {n_inputs},          \"flat\": {flat}, \"dtype\": \"{}\", \"o_dtype\": \"{}\"",
        a.sk,
        a.h,
        a.hv,
        decode_block(a.sk),
        Tpu::score(&a.scores),
        a.causal.map_or("null".into(), |c| c.to_string()),
        row(&a.q),
        row(&a.k),
        row(&a.v),
        a.dtype.full_name(),
        a.out_dtype.full_name(),
    );

    Ok(("decode", spec))
}

/// Flash-attention matching for TPU: scalar epilogue leaves, no extra scalars, runtime offsets for decode.
impl crate::graph_compiler::passes::attention::Attentions for DeviceModule<'_, Tpu> {
    const WITH_FUSE: bool = false;

    /// Whether attention may absorb elementwise epilogues.
    fn contraction_epilogues(&self) -> bool {
        self.options.config.contraction_epilogues
    }

    /// Epilogues reading only scalar leaves, through reshapes but not transposes.
    fn epilogues(&self) -> crate::graph_compiler::passes::attention::Epilogues {
        crate::graph_compiler::passes::attention::Epilogues {
            leaves: |graph, v| graph.type_of(v).shape.is_empty(),
            reshaped: true,
            transposed: false,
        }
    }

    /// None: TPU kernels take no by-value scalars.
    fn scalars(&self) -> Vec<crate::graph::Var> {
        Vec::new()
    }

    /// Names the forward fusion `flash_attention`.
    fn kernel(&mut self) -> String {
        "flash_attention".into()
    }

    /// Yes: the decode kernel (one query a head) reads its causal offset at run time.
    fn runtime_offsets(&self) -> bool {
        true
    }
}

/// Each attention backward runs as a dK/dV kernel and a dQ kernel.
impl crate::graph_compiler::passes::attention::AttentionBackwards for DeviceModule<'_, Tpu> {
    const WITH_FUSE: bool = false;

    /// None: the TPU backward runs dK/dV and dQ as separate kernels.
    fn together(
        &self,
    ) -> impl Fn(&crate::graph_compiler::passes::attention::Backward) -> bool + 'static {
        |_| false
    }

    /// Names the backward fusion `flash_attention_backward`.
    fn backward_kernel(&mut self) -> String {
        "flash_attention_backward".into()
    }
}

/// TPU's attention kernels read and write whole tiles: `attention_layouts` adapts its attention fusions, checking
/// each against `codegen::attention::tpu::kernel`.
impl crate::codegen::attention::TiledAttentions for Tpu {
    /// What `codegen::attention::tpu::kernel` makes of the fusion.
    fn check(name: &str, body: &crate::graph::Graph) -> Result<(), String> {
        crate::codegen::attention::tpu::kernel(name, body).map(|_| ())
    }
}
