//! Layout primitives on MPS (`mps.metal`): a plan's reshapes (copies; the
//! others alias their operand), broadcast_in_dim and transpose, as a
//! dtype-agnostic gather: a tiled transpose where the output's innermost
//! dimension would read the input at a stride.

use crate::Tensor;
use crate::graph::Primitive::*;
use crate::graph::plan::Step;
use crate::ops::mps::{BYTES_PER_THREAD, Grid, dims32_arg, launch, u32_arg};
use crate::tensor::contiguous_strides;

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (x, out) = (&step.inputs[0].1, &step.output.1);
    let xs = contiguous_strides(&x.shape);
    let (width, name) = (out.dtype.size_of(), step.primitive.name());
    let strides = match &step.primitive {
        Reshape { .. } => {
            return gather(width, &[out.numel()], &[1], inputs[0], output, keep, name);
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
        Transpose { permutation } => permutation.iter().map(|&d| xs[d]).collect(),
        _ => unreachable!("a layout primitive"),
    };
    gather(width, &out.shape, &strides, inputs[0], output, keep, name)
}

/// Copy the elements of `src` at `strides` (in elements of `width` bytes)
/// into `dst` as a contiguous tensor of `shape`, recorded in the profiler as
/// `name`. Dimensions that are contiguous with each other in the input too
/// are merged first. Where the innermost one reads the input at a stride
/// while another reads it contiguously, a tiled transpose of the two (the
/// others its batch); else a gather on a grid of (innermost dimension,
/// rows): a few elements a thread where the innermost dimension reads
/// contiguous elements or a broadcast one, else one (strided reads need
/// the threads in flight).
pub(crate) fn gather(
    width: usize,
    shape: &[usize],
    strides: &[usize],
    src: *const u8,
    dst: *mut u8,
    keep: Vec<Tensor>,
    name: &'static str,
) -> Result<(), String> {
    let mut dims: Vec<(usize, usize)> = Vec::new();
    for (&size, &stride) in shape.iter().zip(strides) {
        match dims.last_mut() {
            _ if size == 1 => {}
            Some(last) if last.1 == size * stride => *last = (last.0 * size, stride),
            _ => dims.push((size, stride)),
        }
    }
    let buffers = [src, dst.cast_const()];
    let inner_stride = dims.last().map_or(0, |d| d.1);
    let a = dims.iter().position(|d| d.1 == 1);
    if let (true, Some(a)) = (inner_stride > 1, a) {
        let b = dims.len() - 1;
        let out_strides = contiguous_strides(&dims.iter().map(|d| d.0).collect::<Vec<_>>());
        let batch: Vec<usize> = (0..b).filter(|&d| d != a).collect();
        let args = [
            u32_arg(batch.len() as u32),
            dims32_arg(batch.iter().map(|&d| dims[d].0)),
            dims32_arg(batch.iter().map(|&d| dims[d].1)),
            dims32_arg(batch.iter().map(|&d| out_strides[d])),
            u32_arg(dims[a].0 as u32),
            u32_arg(dims[b].0 as u32),
            u32_arg(out_strides[a] as u32),
            u32_arg(dims[b].1 as u32),
        ];
        let batches = batch.iter().map(|&d| dims[d].0).product();
        let grid = Grid::Groups([dims[a].0.div_ceil(32), dims[b].0.div_ceil(32), batches]);
        return launch(
            &format!("transpose_{width}"),
            &buffers,
            &args,
            grid,
            keep,
            name,
        );
    }
    let inner = dims.last().map_or(1, |d| d.0);
    let outer = &dims[..dims.len().saturating_sub(1)];
    let args = [
        u32_arg(outer.len() as u32),
        dims32_arg(outer.iter().map(|d| d.0)),
        dims32_arg(outer.iter().map(|d| d.1)),
        u32_arg(inner_stride as u32),
        u32_arg(inner as u32),
    ];
    let per_thread = if inner_stride <= 1 {
        (BYTES_PER_THREAD / width).max(1)
    } else {
        1
    };
    let grid = Grid::Threads([
        inner.div_ceil(per_thread),
        outer.iter().map(|d| d.0).product(),
        1,
    ]);
    launch(
        &format!("gather_{width}"),
        &buffers,
        &args,
        grid,
        keep,
        name,
    )
}
