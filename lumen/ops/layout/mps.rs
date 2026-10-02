//! Layout primitives on MPS (`mps.metal`): a plan's reshapes (copies; the
//! others alias their operand), broadcast_in_dim and transpose, as a
//! dtype-agnostic gather, or a tiled transpose when a transpose swaps two
//! blocks of dimensions.

use crate::Tensor;
use crate::graph::Primitive::*;
use crate::graph::plan::Step;
use crate::ops::mps::{Grid, dims32_arg, launch_step, u32_arg};
use crate::tensor::contiguous_strides;

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (x, out) = (&step.inputs[0].1, &step.output.1);
    let xs = contiguous_strides(&x.shape);
    let strides = match &step.primitive {
        Reshape { .. } => {
            return gather(step, &[out.numel()], &[1], inputs, output, keep);
        }
        BroadcastInDim {
            broadcast_dimensions,
            ..
        } => {
            let mut strides = vec![0; out.shape.len()];
            for (k, &d) in broadcast_dimensions.iter().enumerate() {
                if x.shape[k] != 1 {
                    strides[d] = xs[k];
                }
            }
            strides
        }
        Transpose { permutation } => {
            if let Some((batch, rows, cols)) = swapped_blocks(&x.shape, permutation) {
                let grid = Grid::Groups([cols.div_ceil(32), rows.div_ceil(32), batch]);
                let args = [u32_arg(rows as u32), u32_arg(cols as u32)];
                let kernel = format!("transpose_{}", out.dtype.size_of());
                return launch_step(step, &kernel, inputs, output, &args, grid, keep);
            }
            permutation.iter().map(|&d| xs[d]).collect()
        }
        _ => unreachable!("a layout primitive"),
    };
    gather(step, &out.shape, &strides, inputs, output, keep)
}

/// A gather of `shape` from the input at `strides`: a thread per element,
/// on a grid of (innermost dimension, rows).
fn gather(
    step: &Step,
    shape: &[usize],
    strides: &[usize],
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let inner = shape.last().copied().unwrap_or(1);
    let inner_stride = strides.last().copied().unwrap_or(0);
    let outer = &shape[..shape.len().saturating_sub(1)];
    let args = [
        u32_arg(outer.len() as u32),
        dims32_arg(outer.iter().copied()),
        dims32_arg(strides[..outer.len()].iter().copied()),
        u32_arg(inner_stride as u32),
    ];
    let grid = Grid::Threads([inner, outer.iter().product(), 1]);
    let kernel = format!("gather_{}", step.output.1.dtype.size_of());
    launch_step(step, &kernel, inputs, output, &args, grid, keep)
}

/// A transpose by `permutation` as a swap of two blocks of dimensions after
/// some unchanged ones: `(batch, rows, cols)` for viewing the input as
/// [batch, rows, cols] and the output as [batch, cols, rows], if it is one
/// (any transpose of the last two dimensions, `(2, 0, 1)`, ...).
fn swapped_blocks(shape: &[usize], permutation: &[usize]) -> Option<(usize, usize, usize)> {
    let k = permutation
        .iter()
        .enumerate()
        .take_while(|&(i, &d)| i == d)
        .count();
    let rest = &permutation[k..];
    // rest = (j, j + 1, ..., n - 1, k, ..., j - 1), with j = rest[0].
    let j = *rest.first()?;
    let swapped = (j..shape.len()).chain(k..j);
    if !rest.iter().copied().eq(swapped) {
        return None;
    }
    let size = |dims: &[usize]| dims.iter().product::<usize>();
    Some((size(&shape[..k]), size(&shape[k..j]), size(&shape[j..])))
}
