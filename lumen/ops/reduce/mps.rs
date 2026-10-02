//! reduce_sum and reduce_max on MPS (`mps.metal`). Consecutive axes are a
//! view [a, count, b] reduced over the middle: rows when b = 1, columns
//! otherwise, split into chunks when there are too few outputs to fill
//! the GPU. Other axes take a generic kernel: a threadgroup per output, or
//! a thread per output for outputs of few elements.

use crate::graph::Primitive::*;
use crate::graph::TensorType;
use crate::graph::plan::Step;
use crate::graph::primitive::free_dims;
use crate::ops::mps::{
    Grid, dims_arg, dims32_arg, element_arg, launch, launch_step, u32_arg, u64_arg,
};
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar, Tensor, TensorOptions};

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (ReduceSum { axes } | ReduceMax { axes }) = &step.primitive else {
        unreachable!("a reduction")
    };
    let x = &step.inputs[0].1;
    let mut reduced = axes.clone();
    reduced.sort_unstable();
    if reduced.windows(2).all(|w| w[1] == w[0] + 1) {
        return consecutive(step, x, &reduced, inputs[0], output, keep);
    }
    // Other axes: a threadgroup per output (indexing the reduced
    // dimensions in 32 bits), or, for outputs of few elements, a thread per
    // output, which sums in the reference's (row-major) order.
    let strides = contiguous_strides(&x.shape);
    let kept: Vec<usize> = free_dims(x.shape.len(), &reduced).collect();
    let init = match step.primitive {
        ReduceSum { .. } => Scalar::Int(0),
        _ => lowest(x.dtype),
    };
    let count: usize = reduced.iter().map(|&d| x.shape[d]).product();
    let grouped = count >= REDUCE_THREADS && x.numel() <= u32::MAX as usize;
    let reduced_sizes = reduced.iter().map(|&d| x.shape[d]);
    let reduced_strides = reduced.iter().map(|&d| strides[d]);
    let mut args = vec![
        element_arg(x.dtype, init),
        u32_arg(kept.len() as u32),
        dims_arg(kept.iter().map(|&d| x.shape[d])),
        dims_arg(kept.iter().map(|&d| strides[d])),
        u32_arg(reduced.len() as u32),
    ];
    let (name, outputs) = (step.primitive.name(), step.output.1.numel());
    let (kernel, grid) = if grouped {
        args.extend([
            dims32_arg(reduced_sizes),
            dims32_arg(reduced_strides),
            u32_arg(count as u32),
        ]);
        (
            format!("{name}_grouped_{}", x.dtype),
            Grid::Groups([outputs, 1, 1]),
        )
    } else {
        args.extend([
            dims_arg(reduced_sizes),
            dims_arg(reduced_strides),
            u64_arg(count),
        ]);
        (
            format!("{name}_{}", x.dtype),
            Grid::Threads([outputs, 1, 1]),
        )
    };
    launch_step(step, &kernel, inputs, output, &args, grid, keep)
}

/// The threads of a threadgroup that reduce one output together
/// (`REDUCE_THREADS` in `mps.metal`): a [`Grid::Groups`] threadgroup.
const REDUCE_THREADS: usize = 256;

/// The smallest value of `dtype`: the identity of `max`.
fn lowest(dtype: DType) -> Scalar {
    match dtype {
        d if d.is_float() => Scalar::Float(f64::NEG_INFINITY),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            Scalar::Int(i64::MIN >> (64 - 8 * dtype.size_of()))
        }
        _ => Scalar::Int(0),
    }
}

/// Reduce `x` over the consecutive axes `reduced`, viewed as [a, count, b]:
/// rows when b = 1, columns otherwise. One launch, or, with too few outputs
/// to fill the GPU, two: `count` split into chunks reduced in parallel into
/// partials, then the partials.
fn consecutive(
    step: &Step,
    x: &TensorType,
    reduced: &[usize],
    input: *const u8,
    output: *mut u8,
    mut keep: Vec<Tensor>,
) -> Result<(), String> {
    let first = reduced.first().copied().unwrap_or(x.shape.len());
    let end = reduced.last().map_or(first, |last| last + 1);
    let a: usize = x.shape[..first].iter().product();
    let count: usize = x.shape[first..end].iter().product();
    let b: usize = x.shape[end..].iter().product();
    let (name, dtype, rows) = (step.primitive.name(), x.dtype, b == 1);
    if a * b == 0 {
        return Ok(());
    }
    // Chunks per output, so enough run at once: rows take a threadgroup of
    // 256 threads per chunk, columns a thread.
    let chunks = if rows {
        if a >= 256 {
            1
        } else {
            256usize.div_ceil(a).min(count.div_ceil(4096))
        }
    } else if a * b >= 1 << 16 {
        1
    } else {
        (1usize << 16).div_ceil(a * b).min(count.div_ceil(64))
    };
    let chunk = count.div_ceil(chunks.max(1));
    let chunks = if chunk == 0 { 1 } else { count.div_ceil(chunk) };
    let layout = if rows { "rows" } else { "cols" };
    let pass =
        |kernel: String, src: *const u8, dst: *const u8, count, chunk, chunks: usize, keep| {
            if rows {
                let args = [u64_arg(count), u64_arg(chunk)];
                launch(
                    &kernel,
                    &[src, dst],
                    &args,
                    Grid::Groups([chunks, a, 1]),
                    keep,
                    name,
                )
            } else {
                let args = [
                    u32_arg(b as u32),
                    u64_arg(count),
                    u64_arg(chunk),
                    u32_arg(chunks as u32),
                ];
                let grid = Grid::Threads([a * chunks * b, 1, 1]);
                launch(&kernel, &[src, dst], &args, grid, keep, name)
            }
        };
    if chunks == 1 {
        let kernel = format!("{name}_{layout}_{dtype}");
        return pass(kernel, input, output, count, count, 1, keep);
    }
    // Partials accumulate in float for the 16-bit floats, as the kernels do.
    let acc = if matches!(dtype, DType::F16 | DType::BF16) {
        DType::F32
    } else {
        dtype
    };
    let options = TensorOptions::new().dtype(acc).device(crate::Device::Mps);
    // SAFETY: the first launch writes every partial before the second reads.
    let partials = unsafe { Tensor::empty(&[a * chunks * b], options) };
    keep.push(partials.clone());
    let p = partials.data_ptr().cast_const();
    let kernel = format!("{name}_{layout}_partial_{dtype}");
    pass(kernel, input, p, count, chunk, chunks, keep.clone())?;
    let kernel = format!("{name}_{layout}_final_{dtype}");
    pass(kernel, p, output, chunks, chunks, 1, keep)
}
