//! `dot_general` on TPU: the spec of its kernel (`tpu.py` beside this file,
//! in lumen's Pallas, which says how it works): the dot's form, its blocks
//! and dtypes. A dot the kernel cannot run errors here, saying why.

use crate::DType;
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::ops::tpu::{Launch, launch};

/// A `dot_general` as `out[b, l] = lhs[b, l] · rhs[b]`: `batch` times
/// `lead` (the lhs's free dimensions before its last) of `[m, k]` times
/// `[k, n]` (`[n, k]` if `transposed`). A TPU's tensors are tiled on their
/// last two dimensions, the others merging freely: the lhs is viewed as
/// `[batch · lead, m, k]`, the output as `[batch · lead, m, n]`.
#[derive(Debug, PartialEq)]
struct Form {
    batch: usize,
    lead: usize,
    m: usize,
    n: usize,
    k: usize,
    transposed: bool,
}

/// `p`'s form on operands of shapes `lhs` and `rhs`: an error unless the
/// lhs is batch, free then one contracting dimension (a free one at
/// least), and the rhs batch then its contracting and one free dimension
/// (either order).
fn form(p: &Primitive, lhs: &[usize], rhs: &[usize]) -> Result<Form, String> {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
        ..
    } = p
    else {
        unreachable!("a dot_general's form")
    };
    let run = |dims: &[usize], start: usize| dims.iter().enumerate().all(|(i, &d)| d == start + i);
    let nb = lhs_batch.len();
    let product = |dims: &[usize]| dims.iter().product::<usize>();
    if !(run(lhs_batch, 0) && run(rhs_batch, 0)) {
        return Err("dot_general on TPU: the operands' batch dimensions must lead them".into());
    }
    if *lhs_contracting != [lhs.len() - 1] || lhs.len() < nb + 2 {
        return Err(
            "dot_general on TPU: the lhs must be batch, free then one contracting dimension".into(),
        );
    }
    if rhs.len() != nb + 2 || rhs_contracting.len() != 1 {
        return Err(
            "dot_general on TPU: the rhs must be batch, then one contracting and one free dimension"
                .into(),
        );
    }
    let transposed = rhs_contracting[0] == nb + 1;
    Ok(Form {
        batch: product(&lhs[..nb]),
        lead: product(&lhs[nb..lhs.len() - 2]),
        m: lhs[lhs.len() - 2],
        n: rhs[if transposed { nb } else { nb + 1 }],
        k: lhs[lhs.len() - 1],
        transposed,
    })
}

/// A block of a dimension of `n`: the whole if at most `cap`, else the
/// largest multiple of `unit` at most `cap` that divides it.
pub(crate) fn block(n: usize, unit: usize, cap: usize) -> Option<usize> {
    if n <= cap {
        return Some(n);
    }
    (1..=cap / unit)
        .rev()
        .map(|i| i * unit)
        .find(|&b| n.is_multiple_of(b))
}

/// The VMEM a dot's blocks may take: Mosaic's scoped limit (100 MiB,
/// `ops/tpu`) less what it allocates itself.
const VMEM: usize = 88 << 20;

/// The largest blocks (bm, bn, bk) worth taking of operands of `dtype` (k
/// of `k`), as measured on a v6e (`benchmarks/tpu_matmul.py`): about 87%
/// of its peak on large dots, XLA's or better. Large blocks feed the
/// matrix unit more flops per byte read: `bm · bn / (bm + bn)` a byte, and
/// HBM moves a byte while it does ~800. A k of one block needs no
/// accumulator (the product is written as it is), which pays at k ≤ 4096;
/// float32 (full precision: six bfloat16 passes) needs smaller blocks.
fn caps(dtype: DType, k: usize) -> (usize, usize, usize) {
    match dtype {
        DType::F32 => (2048, 1024, 1024),
        _ if k <= 4096 => (2048, 1024, k),
        _ => (2048, 2048, 2048),
    }
}

/// The VMEM a dot's blocks take: two of each operand's and the output's
/// (double-buffered), the float32 accumulator (more than one block of k),
/// and float32 operands' bfloat16 parts (full precision splits them).
fn footprint(bm: usize, bn: usize, bk: usize, kb: usize, operand: DType, output: DType) -> usize {
    let operands = bm * bk + bk * bn;
    let mut bytes = 2 * operands * operand.size_of() + 2 * bm * bn * output.size_of();
    if kb > 1 {
        bytes += bm * bn * 4;
    }
    if operand == DType::F32 {
        bytes += operands * 4;
    }
    bytes
}

/// The rows of a tile of `dtype`: 8 for float32, 16 for bfloat16.
fn tile_rows(dtype: DType, what: &str) -> Result<usize, String> {
    match dtype {
        DType::F32 => Ok(8),
        DType::BF16 => Ok(16),
        _ => Err(format!(
            "dot_general on TPU: {what} must be float32 or bfloat16, got {dtype}"
        )),
    }
}

pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let (lhs, rhs) = (&step.inputs[0].1, &step.inputs[1].1);
    let out = &step.output.1;
    if out.numel() == 0 {
        return Ok(());
    }
    let Primitive::DotGeneral { accum_dtype, .. } = step.primitive else {
        unreachable!("a dot_general step")
    };
    let f = form(&step.primitive, &lhs.shape, &rhs.shape)?;
    if lhs.dtype != rhs.dtype || accum_dtype != DType::F32 {
        return Err(format!(
            "dot_general on TPU: operands of one dtype accumulating in float32, got {} · {} in {}",
            lhs.dtype, rhs.dtype, accum_dtype
        ));
    }
    // Blocks of whole tiles (or whole dimensions), as large as fit VMEM.
    let unit_m = tile_rows(lhs.dtype, "the operands")?.max(tile_rows(out.dtype, "the output")?);
    let (mut cap_m, mut cap_n, mut cap_k) = caps(lhs.dtype, f.k);
    let mut blocks;
    loop {
        blocks = (
            block(f.m, unit_m, cap_m),
            block(f.n, 128, cap_n),
            block(f.k, 128, cap_k),
        );
        let (Some(bm), Some(bn), Some(bk)) = blocks else {
            break;
        };
        if footprint(bm, bn, bk, f.k / bk, lhs.dtype, out.dtype) <= VMEM {
            break;
        }
        // Smaller: k's first, then n's, then m's.
        match (cap_m, cap_n, cap_k) {
            (_, _, k) if k > 512 => cap_k = k / 2,
            (_, n, _) if n > 512 => cap_n = n / 2,
            (m, _, _) if m > 512 => cap_m = m / 2,
            _ => break,
        }
    }
    // A few steps at least, so its copies overlap its products: n's
    // blocks halved (the lhs's block stays, the rhs's stream past it).
    while let (Some(bm), Some(bn), Some(bk)) = blocks {
        let steps = f.batch * f.lead * (f.m / bm) * (f.n / bn) * (f.k / bk);
        if steps >= 4 || cap_n <= 256 || bn < 512 {
            break;
        }
        cap_n = bn / 2;
        blocks.1 = block(f.n, 128, cap_n);
    }
    let (Some(bm), Some(bn), Some(bk)) = blocks else {
        let failed = [
            (
                blocks.0,
                format!("M ({}) at most {cap_m} or a multiple of {unit_m}", f.m),
            ),
            (
                blocks.1,
                format!("N ({}) at most {cap_n} or a multiple of 128", f.n),
            ),
            (
                blocks.2,
                format!("K ({}) at most {cap_k} or a multiple of 128", f.k),
            ),
        ];
        let failed: Vec<String> = failed
            .into_iter()
            .filter_map(|(b, why)| b.is_none().then_some(why))
            .collect();
        return Err(format!(
            "dot_general on TPU of {} needs {}",
            lhs.dtype,
            failed.join(" and ")
        ));
    };
    let spec = format!(
        "\"batch\": {}, \"lead\": {}, \"m\": {}, \"n\": {}, \"k\": {}, \"transposed\": {}, \
         \"operands\": \"{}\", \"output\": \"{}\", \"blocks\": [{bm}, {bn}, {bk}]",
        f.batch,
        f.lead,
        f.m,
        f.n,
        f.k,
        f.transposed,
        lhs.dtype.full_name(),
        out.dtype.full_name()
    );
    let rows = [inputs[0], inputs[1], output.cast_const()];
    let launched = Launch {
        module: "lumen.ops.dot_general.tpu",
        // Its operands', accumulator's and output's dtypes, and blocks.
        kernel: format!("matmul_{}_f32_{}_{bm}x{bn}x{bk}", lhs.dtype, out.dtype),
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
    };
    launch(index, launched)
}
