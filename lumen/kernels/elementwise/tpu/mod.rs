//! Elementwise Pallas kernels on TPU: block-size selection and launching a fused loop's program.

use std::collections::HashMap;

use crate::DType;
use crate::codegen::elementwise::ElementwiseCodegen;
use crate::codegen::tpu::Tpu;
use crate::device::tpu::layout::Layout;
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::kernels::dot_general::tpu::block;
use crate::kernels::tpu::{GENERATED, Launch, launch};

/// Upper bound in bytes on one block including its batched leading dimension.
const BLOCK_BYTES: usize = 1 << 20;

/// Byte budget for all operand blocks streamed together, split evenly across dtypes.
const STREAM_BYTES: usize = 32 << 20;

/// Picks a Pallas block shape for an elementwise op over `shape`: one block length for rank < 2,
/// otherwise `[leading, rows, columns]` made of whole TPU tiles for every dtype in `dtypes`.
pub(crate) fn blocks(shape: &[usize], dtypes: &[DType]) -> Result<Vec<usize>, String> {
    let failed = |what: &str| format!("elementwise on TPU: {what} has no block of whole tiles");
    if shape.len() < 2 {
        let n = shape.iter().product::<usize>().max(1);
        return Ok(vec![
            block(n, 1024, 64 * 1024).ok_or_else(|| failed("its length"))?,
        ]);
    }
    // The row tile is the tallest sublane count among the dtypes, so blocks align for all of them.
    let rows = dtypes
        .iter()
        .map(|&d| Layout::new(d, &[1, 1]).map(|l| l.rows))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(8);
    let (m, n) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let mut bm = block(m, rows, 512).ok_or_else(|| failed("its rows"))?;
    let mut bn = block(n, 128, 2048).ok_or_else(|| failed("its columns"))?;
    let lead: usize = shape[..shape.len() - 2].iter().product();
    let width = dtypes.iter().map(|d| d.size_of()).max().unwrap_or(4);
    let budget = STREAM_BYTES / dtypes.len();
    // Halve rows first, then columns, while the block still divides the shape in whole tiles.
    while bm * bn * width > budget {
        let halves = |b: usize, unit: usize, of: usize| {
            (b / 2).is_multiple_of(unit) && of.is_multiple_of(b / 2)
        };
        if halves(bm, rows, m) {
            bm /= 2;
        } else if halves(bn, 128, n) {
            bn /= 2;
        } else {
            break;
        }
    }
    // Bytes of one padded (bm, bn) tile; batch as many leading rows as fit in BLOCK_BYTES.
    let tile = bm.div_ceil(rows) * rows * bn.div_ceil(128) * 128 * width;
    let bl = (1..=lead)
        .rev()
        .find(|&b| lead.is_multiple_of(b) && b * tile <= BLOCK_BYTES)
        .unwrap_or(1);
    Ok(vec![bl, bm, bn])
}

/// Greatest common divisor of `a` and `b`.
pub(crate) fn gcd(a: usize, b: usize) -> usize {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

/// Launches the fused elementwise loop `step` on TPU `index`, writing into `output`.
pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("a loop fusion")
    };
    if step.output.1.numel() == 0 {
        return Ok(());
    }
    let views: HashMap<usize, (usize, Vec<usize>)> = step
        .output_views
        .iter()
        .enumerate()
        .filter_map(|(k, v)| v.as_ref().map(|v| (k, (v.offset, v.strides.clone()))))
        .collect();
    let code =
        Tpu::elementwise(body, &[], &views, &()).map_err(|e| format!("{}: {e}", step.label))?;
    // An indexed loop takes its inputs, then its output; a plain one its output after the step's own inputs.
    let n = match code.indexed {
        true => inputs.len(),
        false => step.inputs.len(),
    };
    let rows: Vec<*const u8> = inputs[..n]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(inputs[n..].iter().copied())
        .collect();
    let launched = Launch {
        module: GENERATED,
        kernel: code.kernel,
        spec: code.source,
        rows: &rows,
        scalars: &[],
        host: &[],
        sparse: None,
    };
    launch(index, launched)
}
