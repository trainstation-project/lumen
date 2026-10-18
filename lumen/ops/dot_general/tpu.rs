//! `dot_general` on TPU: the spec of its Pallas kernel (`tpu.py` beside
//! this file, which says how it works): the dot's form, its blocks and
//! dtypes. A dot the kernel cannot run errors here, saying why.

use crate::DType;
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::ops::tpu::{Launch, launch};

/// A `dot_general` as `out[b] = lhs[b] · rhs[b]`: `batch` of `[m, k]` times
/// `[k, n]` (`[n, k]` if `transposed`), each operand's batch, free and
/// contracting dimensions collapsed.
#[derive(Debug, PartialEq)]
struct Form {
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    transposed: bool,
}

/// `p`'s form on operands of shapes `lhs` and `rhs`: an error unless the
/// lhs is batch, free then contracting dimensions and the rhs batch then
/// contracting and free dimensions (either order), each group in order.
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
    let (nb, nc) = (lhs_batch.len(), lhs_contracting.len());
    let product = |dims: &[usize]| dims.iter().product::<usize>();
    let in_order = run(lhs_batch, 0) && run(rhs_batch, 0) && run(lhs_contracting, lhs.len() - nc);
    let transposed = if run(rhs_contracting, nb) {
        false
    } else if run(rhs_contracting, rhs.len() - nc) {
        true
    } else {
        return Err(
            "dot_general on TPU: the rhs's contracting dimensions must follow its batch \
                    dimensions or end it"
                .into(),
        );
    };
    if !in_order {
        return Err(
            "dot_general on TPU: the lhs must be batch, free then contracting dimensions".into(),
        );
    }
    let free = match transposed {
        false => &rhs[nb + nc..],
        true => &rhs[nb..rhs.len() - nc],
    };
    Ok(Form {
        batch: product(&lhs[..nb]),
        m: product(&lhs[nb..lhs.len() - nc]),
        n: product(free),
        k: product(&lhs[lhs.len() - nc..]),
        transposed,
    })
}

/// The largest multiple of `unit` at most `cap` that divides `n`.
fn block(n: usize, unit: usize, cap: usize) -> Option<usize> {
    (1..=cap / unit)
        .rev()
        .map(|i| i * unit)
        .find(|&b| n.is_multiple_of(b))
}

/// Elements of `dtype` in a 32-bit word: 1 for float32, 2 for bfloat16.
fn packing(dtype: DType, what: &str) -> Result<usize, String> {
    match dtype {
        DType::F32 => Ok(1),
        DType::BF16 => Ok(2),
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
    let pin = packing(lhs.dtype, "the operands")?;
    let pout = packing(out.dtype, "the output")?;
    // The output's columns split by parity only where a word holds two:
    // a rhs whose rows hold N pairs them, so its output must too.
    if (pin == 1 && pout == 2) || (pin == 2 && !f.transposed && pout == 1) {
        return Err(format!(
            "dot_general on TPU: {} operands with a {} output are not supported yet",
            lhs.dtype, out.dtype
        ));
    }
    // Blocks of whole rows of 128 words (a pair of rows, a word two
    // elements), as large as fit VMEM: tuned on a v6e.
    let unit_n = 128 * pout.max(if f.transposed { 1 } else { pin });
    let unit_k = 128 * pin;
    let (cap_m, cap_n, cap_k) = match pin {
        1 => (512, 512, 512),
        _ => (1024, 1024, 512),
    };
    // An lhs block's rows are M's whole, or a multiple of 8 (the batch's
    // matrices are one).
    let blocks = (
        match f.batch == 1 && f.m <= cap_m {
            true => Some(f.m),
            false => block(f.m, 8, cap_m),
        },
        block(f.n, unit_n, cap_n),
        block(f.k, unit_k, cap_k),
    );
    let (Some(bm), Some(bn), Some(bk)) = blocks else {
        let failed = [
            (
                blocks.0,
                format!(
                    "M ({}) a multiple of 8 (or at most {cap_m}, unbatched)",
                    f.m
                ),
            ),
            (blocks.1, format!("N ({}) a multiple of {unit_n}", f.n)),
            (blocks.2, format!("K ({}) a multiple of {unit_k}", f.k)),
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
        "\"batch\": {}, \"m\": {}, \"n\": {}, \"k\": {}, \"transposed\": {}, \"packing\": \
         [{pin}, {pout}], \"blocks\": [{bm}, {bn}, {bk}]",
        f.batch, f.m, f.n, f.k, f.transposed
    );
    let rows = [inputs[0], inputs[1], output.cast_const()];
    let launched = Launch {
        module: "lumen.ops.dot_general.tpu",
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
    };
    launch(index, launched)
}
