//! `dot_general` on TPU: one Mosaic kernel ([`crate::ops::tpu::mosaic`]) of
//! the matrix unit, over a grid of output blocks and K blocks: each step
//! copies its operands' blocks from the device's memory to VMEM, multiplies
//! them into a float32 accumulator, and the last K block writes the output
//! block.
//!
//! The device's memory holds tensors as the host does, row-major, which is
//! not how a TPU lays out a 16-bit matrix (it packs two rows' elements in a
//! 32-bit word); a kernel reads it as 32-bit words. A word of a bfloat16
//! row holds two neighbours: elements `2w` (low half) and `2w + 1`. So a
//! bfloat16 kernel splits each operand by parity, its words' halves
//! (widened to float32 exactly, then narrowed back to bfloat16 for the
//! matrix unit), and sums parities:
//! `out[:, q::2] = Σ_p lhs[:, p::2] · rhs[p::2, q::2]`, four matmuls of a
//! quarter of the work each; a rhs whose rows hold N pairs its rows by k
//! parity (two rows a row of `[K/2, 2 · N/2]` words), and one whose rows
//! hold K pairs them by n parity. A bfloat16 output is written as words
//! too: its even columns' bits in the low halves, its odd columns' in the
//! high.

use crate::DType;
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::ops::tpu::mosaic::{Kernel, Space, Type, Value};
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
    let unit_n = 128 * pout.max(if f.transposed { 1 } else { pin });
    let unit_k = 128 * pin;
    let blocks = (
        match f.m <= 512 {
            true => Some(f.m),
            false => block(f.m, 8, 512),
        },
        block(f.n, unit_n, 512),
        block(f.k, unit_k, 512 * pin),
    );
    let (Some(bm), Some(bn), Some(bk)) = blocks else {
        return Err(format!(
            "dot_general on TPU: M ({}) must be at most 512 or a multiple of 8, N ({}) a \
             multiple of {unit_n} and K ({}) of {unit_k}",
            f.m, f.n, f.k
        ));
    };
    let rows = crate::allocator::tpu::rows(index)?;
    let body = kernel(&f, (pin, pout), (bm, bn, bk), rows);
    let name = format!("dot_general_{}_{}", lhs.dtype, out.dtype);
    let rows = [inputs[0], inputs[1], output.cast_const()];
    let launched = Launch {
        name: &name,
        body,
        rows: &rows,
        scalars: &[],
        host: &[],
    };
    launch(index, launched)
}

/// `a + Σ v · c` of i32 scalars.
fn sum(k: &mut Kernel, terms: &[(&Value, usize)], constant: usize) -> Value {
    let mut acc = k.i32(constant as i64);
    for (v, c) in terms {
        acc = k.affine(&acc, v, *c as i64);
    }
    acc
}

/// The kernel: `f` of operands of `pin` and an output of `pout` elements
/// a word, in blocks of `bm × bn` outputs and `bk` contracting, on a device
/// of `rows` rows of memory. Its scalars are the rows of the lhs, the rhs
/// and the output.
fn kernel(
    f: &Form,
    (pin, pout): (usize, usize),
    (bm, bn, bk): (usize, usize, usize),
    rows: usize,
) -> String {
    let Form {
        batch,
        m,
        n,
        k: kk,
        transposed,
    } = *f;
    let words = |shape: Vec<usize>| Type::Memref(shape, "i32", Space::Vmem);
    let rhs_block = match (pin, transposed) {
        (1, false) => vec![bk, bn],
        (1, true) => vec![bn, bk],
        (_, false) => vec![bk / 2, bn / 2],
        (_, true) => vec![bn / pout, bk / 2],
    };
    let rhs_buffers = match (pin, transposed, pout) {
        (2, false, _) | (2, true, 2) => 2,
        _ => 1,
    };
    let mut scratch = vec![words(vec![bm, bk / pin])];
    scratch.extend((0..rhs_buffers).map(|_| words(rhs_block.clone())));
    let acc_shape = vec![bm, bn / pout];
    scratch.extend((0..pout).map(|_| Type::Memref(acc_shape.clone(), "f32", Space::Vmem)));
    scratch.push(words(vec![bm, bn / pout]));
    let grid = [batch, m / bm, n / bn, kk / bk];
    let mut k = Kernel::new(&grid, 3, None, rows, &scratch);
    let (b, i, j, step) = (
        k.program_id(0),
        k.program_id(1),
        k.program_id(2),
        k.program_id(3),
    );
    let xv = k.scratch(0);
    let yv: Vec<Value> = (0..rhs_buffers).map(|r| k.scratch(1 + r)).collect();
    let accs: Vec<Value> = (0..pout).map(|q| k.scratch(1 + rhs_buffers + q)).collect();
    let ov = k.scratch(1 + rhs_buffers + pout);

    // A matrix of `shape` words at the row of scalar `s`, in `memory`.
    let matrix = |k: &mut Kernel, memory: &Value, s: usize, shape: [usize; 2]| {
        let row = k.scalar(s);
        let zero = k.i32(0);
        let all = k.slice(
            memory,
            &[&row, &zero, &zero],
            &[shape[0] * shape[1] / 128, 1, 128],
        );
        k.reshape(&all, &shape)
    };
    // `matrix`'s `shape` at row `r`, column `c` (a multiple of 128).
    let at = |k: &mut Kernel, matrix: &Value, r: &Value, c: &Value, shape: &[usize]| {
        let c = k.multiple_of(c, 128);
        k.slice(matrix, &[r, &c], shape)
    };

    let memory = k.memory();
    let x = matrix(&mut k, &memory, 0, [batch * m, kk / pin]);
    let r = sum(&mut k, &[(&b, m), (&i, bm)], 0);
    let c = sum(&mut k, &[(&step, bk / pin)], 0);
    let x = at(&mut k, &x, &r, &c, &[bm, bk / pin]);
    k.copy(&x, &xv);

    // The rhs: a matrix of words whose blocks `yv` holds.
    let y_shape = match (pin, transposed, pout) {
        (1, false, _) => [batch * kk, n],
        (1, true, _) => [batch * n, kk],
        (_, false, _) => [batch * kk / 2, n],
        (_, true, 2) => [batch * n / 2, kk],
        (_, true, _) => [batch * n, kk / 2],
    };
    let y = matrix(&mut k, &memory, 1, y_shape);
    let (r, c, half) = match (pin, transposed, pout) {
        (1, false, _) => (
            sum(&mut k, &[(&b, kk), (&step, bk)], 0),
            sum(&mut k, &[(&j, bn)], 0),
            0,
        ),
        (1, true, _) => (
            sum(&mut k, &[(&b, n), (&j, bn)], 0),
            sum(&mut k, &[(&step, bk)], 0),
            0,
        ),
        (_, false, _) => (
            sum(&mut k, &[(&b, kk / 2), (&step, bk / 2)], 0),
            sum(&mut k, &[(&j, bn / 2)], 0),
            n / 2,
        ),
        (_, true, 2) => (
            sum(&mut k, &[(&b, n / 2), (&j, bn / 2)], 0),
            sum(&mut k, &[(&step, bk / 2)], 0),
            kk / 2,
        ),
        (_, true, _) => (
            sum(&mut k, &[(&b, n), (&j, bn)], 0),
            sum(&mut k, &[(&step, bk / 2)], 0),
            0,
        ),
    };
    for (h, yv) in yv.iter().enumerate() {
        let shift = k.i32((h * half) as i64);
        let c = k.add(&c, &shift);
        let y = at(&mut k, &y, &r, &c, &rhs_block);
        k.copy(&y, yv);
    }

    // The operands' parities, as matrix unit inputs.
    let unpack = |k: &mut Kernel, words: &Value, high: bool| {
        let shape = words.shape().to_vec();
        let half = match high {
            false => {
                let sixteen = k.splat_i32(16, &shape);
                k.shift_left(words, &sixteen)
            }
            true => {
                let mask = k.splat_i32(-65536, &shape);
                k.and(words, &mask)
            }
        };
        let float = k.bitcast(&half, "f32");
        k.truncate(&float, "bf16")
    };
    let xw = k.load(&xv);
    let ys: Vec<Value> = yv.iter().map(|yv| k.load(yv)).collect();
    let xs: Vec<Value> = match pin {
        1 => vec![k.bitcast(&xw, "f32")],
        _ => vec![unpack(&mut k, &xw, false), unpack(&mut k, &xw, true)],
    };
    // `y[p][q]`: the rhs of k parity `p` and n parity `q`.
    let y: Vec<Vec<Value>> = match (pin, transposed, pout) {
        (1, _, _) => vec![vec![k.bitcast(&ys[0], "f32")]],
        (_, false, _) => ys
            .iter()
            .map(|w| vec![unpack(&mut k, w, false), unpack(&mut k, w, true)])
            .collect(),
        (_, true, 2) => (0..2)
            .map(|p| ys.iter().map(|w| unpack(&mut k, w, p == 1)).collect())
            .collect(),
        (_, true, _) => (0..2)
            .map(|p| vec![unpack(&mut k, &ys[0], p == 1)])
            .collect(),
    };
    let zeros = k.zeros(&acc_shape);
    let parts: Vec<Value> = (0..pout)
        .map(|q| {
            (0..pin).fold(zeros.clone(), |acc, p| {
                k.matmul(&xs[p], &y[p][q], &acc, transposed)
            })
        })
        .collect();

    let zero = k.i32(0);
    let first = k.equal(&step, &zero);
    k.when(&first, |k| {
        for acc in &accs {
            k.store(&zeros, acc);
        }
    });
    for (acc, part) in accs.iter().zip(&parts) {
        let old = k.load(acc);
        let new = k.add(&old, part);
        k.store(&new, acc);
    }
    let last = k.i32((kk / bk - 1) as i64);
    let last = k.equal(&step, &last);
    k.when(&last, |k| {
        let rounded = |k: &mut Kernel, acc: &Value| {
            let v = k.load(acc);
            let v = k.truncate(&v, "bf16");
            let v = k.extend(&v, "f32");
            k.bitcast(&v, "i32")
        };
        let words = match pout {
            1 => {
                let v = k.load(&accs[0]);
                k.bitcast(&v, "i32")
            }
            _ => {
                let even = rounded(k, &accs[0]);
                let sixteen = k.splat_i32(16, &acc_shape);
                let even = k.shift_right(&even, &sixteen);
                let odd = rounded(k, &accs[1]);
                let mask = k.splat_i32(-65536, &acc_shape);
                let odd = k.and(&odd, &mask);
                k.or(&even, &odd)
            }
        };
        k.store(&words, &ov);
        let result = k.result();
        let o = matrix(k, &result, 2, [batch * m, n / pout]);
        let r = sum(k, &[(&b, m), (&i, bm)], 0);
        let c = sum(k, &[(&j, bn / pout)], 0);
        let o = at(k, &o, &r, &c, &[bm, bn / pout]);
        k.copy(&ov, &o);
    });
    k.finish("dot_general")
}
