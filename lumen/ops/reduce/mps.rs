//! reduce_sum and reduce_max on MPS (`mps.metal`). Consecutive axes are a
//! view [a, count, b] reduced over the middle: rows when b = 1, columns
//! otherwise, split into chunks when there are too few outputs to fill
//! the GPU. Other axes take a generic kernel, with up to a threadgroup
//! (fewer threads for outputs of fewer elements) reducing each output.

use crate::graph::Primitive::{self, *};
use crate::graph::TensorType;
use crate::graph::plan::Step;
use crate::graph::primitive::free_dims;
use crate::ops::mps::{Grid, dims_arg, dims32_arg, element_arg, launch, u32_arg, u64_arg};
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar, Tensor};

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let x = &step.inputs[0].1;
    let name = step.label;
    encode_reduction(
        &step.primitive,
        x,
        None,
        name,
        &inputs[..1],
        output,
        &[],
        scratch,
        keep,
    )
}

/// How a reduction of `x` over `axes` runs: the kernel template its (first)
/// launch instantiates, as [`encode_reduction`] launches it.
pub(crate) enum Layout {
    /// Any axes, a thread an output (`reduce`): inputs of 2^32 elements or
    /// more.
    Generic,
    /// Any axes, lanes of threads an output (`reduce_grouped`).
    Grouped,
    /// Consecutive axes, viewed as [a, count, b]: rows when b = 1
    /// (`reduce_rows`), columns otherwise (`reduce_cols`); `split` into
    /// partials, which a second launch reduces, when there are too few
    /// outputs to fill the GPU.
    Rows {
        split: bool,
    },
    Cols {
        split: bool,
    },
}

pub(crate) fn layout(x: &TensorType, axes: &[usize]) -> Layout {
    let mut reduced = axes.to_vec();
    reduced.sort_unstable();
    if reduced.windows(2).all(|w| w[1] == w[0] + 1) {
        let Split { b, chunks, .. } = split(x, &reduced);
        return match b {
            1 => Layout::Rows { split: chunks > 1 },
            _ => Layout::Cols { split: chunks > 1 },
        };
    }
    match x.numel() <= u32::MAX as usize {
        true => Layout::Grouped,
        false => Layout::Generic,
    }
}

/// Reduce `x` with `op` (reduce_sum or reduce_max), its first launch the
/// kernel `fused` if given (a reduction fusion's, computing its input from
/// `inputs`, and writing its `extra` outputs after `output`), else the
/// primitive's own (reading `inputs[0]`), recorded in the profiler as
/// `name`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_reduction(
    op: &Primitive,
    x: &TensorType,
    fused: Option<&str>,
    name: &'static str,
    inputs: &[*const u8],
    output: *mut u8,
    extra: &[*const u8],
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (ReduceSum { axes } | ReduceMax { axes }) = op else {
        unreachable!("a reduction")
    };
    let (op_name, dtype) = (op.name(), x.dtype);
    let kernel = |own: String| fused.map_or(own, str::to_owned);
    let mut buffers = inputs.to_vec();
    buffers.push(output.cast_const());
    buffers.extend(extra);
    let mut reduced = axes.clone();
    reduced.sort_unstable();
    let layout = layout(x, &reduced);
    if let Layout::Rows { .. } | Layout::Cols { .. } = layout {
        return consecutive(
            op_name, name, x, &reduced, fused, inputs, output, extra, scratch, keep,
        );
    }
    // Other axes: a power-of-two number of lanes per output, indexing the
    // reduced dimensions in 32 bits; inputs of 2^32 elements or more take a
    // thread per output, which sums in the reference's (row-major) order.
    let strides = contiguous_strides(&x.shape);
    let kept: Vec<usize> = free_dims(x.shape.len(), &reduced).collect();
    let init = match op {
        ReduceSum { .. } => Scalar::Int(0),
        _ => lowest(dtype),
    };
    let count: usize = reduced.iter().map(|&d| x.shape[d]).product();
    let outputs: usize = kept.iter().map(|&d| x.shape[d]).product();
    let reduced_sizes = reduced.iter().map(|&d| x.shape[d]);
    let reduced_strides = reduced.iter().map(|&d| strides[d]);
    let mut args = vec![
        element_arg(dtype, init),
        u32_arg(kept.len() as u32),
        dims_arg(kept.iter().map(|&d| x.shape[d])),
        dims_arg(kept.iter().map(|&d| strides[d])),
        u32_arg(reduced.len() as u32),
    ];
    let (kernel, grid) = if let Layout::Grouped = layout {
        // About 32 bytes a lane: enough lanes to read the elements
        // coalesced, few enough that each lane's loads stay in flight.
        let per_lane = (LANE_BYTES / dtype.size_of()).max(1);
        let lanes = count
            .div_ceil(per_lane)
            .next_power_of_two()
            .min(REDUCE_THREADS);
        args.extend([
            dims32_arg(reduced_sizes),
            dims32_arg(reduced_strides),
            u32_arg(count as u32),
            u32_arg(lanes as u32),
            u32_arg(outputs as u32),
        ]);
        let per_group = REDUCE_THREADS / lanes;
        (
            kernel(format!("{op_name}_grouped_{dtype}")),
            Grid::Groups([outputs.div_ceil(per_group), 1, 1]),
        )
    } else {
        args.extend([
            dims_arg(reduced_sizes),
            dims_arg(reduced_strides),
            u64_arg(count),
        ]);
        (
            kernel(format!("{op_name}_{dtype}")),
            Grid::Threads([outputs, 1, 1]),
        )
    };
    launch(&kernel, &buffers, &args, grid, keep, name)
}

/// The threads of a threadgroup that reduce together (`REDUCE_THREADS` in
/// `mps.metal`): a [`Grid::Groups`] threadgroup.
const REDUCE_THREADS: usize = 256;

/// The bytes of input each lane of a grouped reduction takes.
const LANE_BYTES: usize = 32;

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

/// How [`consecutive`] reduces `x` over the consecutive axes `reduced`,
/// viewed as [a, count, b]: rows when b = 1, columns otherwise, in `chunks`
/// chunks of `chunk` elements (one: a single launch; more: partials, then
/// the partials).
struct Split {
    a: usize,
    count: usize,
    b: usize,
    chunks: usize,
    chunk: usize,
}

fn split(x: &TensorType, reduced: &[usize]) -> Split {
    let first = reduced.first().copied().unwrap_or(x.shape.len());
    let end = reduced.last().map_or(first, |last| last + 1);
    let a: usize = x.shape[..first].iter().product();
    let count: usize = x.shape[first..end].iter().product();
    let b: usize = x.shape[end..].iter().product();
    // Chunks per output, so enough run at once: rows take a threadgroup of
    // 256 threads per chunk, columns a thread.
    let chunks = if b == 1 {
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
    Split {
        a,
        count,
        b,
        chunks,
        chunk,
    }
}

/// The type a split reduction's partials have: float for the 16-bit
/// floats, as the kernels accumulate.
fn partial_dtype(dtype: DType) -> DType {
    if matches!(dtype, DType::F16 | DType::BF16) {
        DType::F32
    } else {
        dtype
    }
}

/// The scratch a reduction of `x` over `axes` needs: a split reduction's
/// partials (none for the others).
pub(crate) fn scratch_bytes(x: &TensorType, axes: &[usize]) -> usize {
    let mut reduced = axes.to_vec();
    reduced.sort_unstable();
    if !reduced.windows(2).all(|w| w[1] == w[0] + 1) {
        return 0;
    }
    let Split { a, b, chunks, .. } = split(x, &reduced);
    if chunks == 1 || a * b == 0 {
        return 0;
    }
    a * chunks * b * partial_dtype(x.dtype).size_of()
}

/// Reduce `x` over the consecutive axes `reduced`, viewed as [a, count, b]:
/// rows when b = 1, columns otherwise. One launch, or, with too few outputs
/// to fill the GPU, two: `count` split into chunks reduced in parallel into
/// partials (in `scratch`, which the planner set aside), then the partials.
/// The first launch is the kernel `fused` if given (reading `inputs`, and
/// writing its `extra` outputs), else `op`'s own (reading `inputs[0]`).
#[allow(clippy::too_many_arguments)]
fn consecutive(
    op: &str,
    name: &'static str,
    x: &TensorType,
    reduced: &[usize],
    fused: Option<&str>,
    inputs: &[*const u8],
    output: *mut u8,
    extra: &[*const u8],
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let Split {
        a,
        count,
        b,
        chunks,
        chunk,
    } = split(x, reduced);
    let (dtype, rows) = (x.dtype, b == 1);
    if a * b == 0 {
        return Ok(());
    }
    let layout = if rows { "rows" } else { "cols" };
    let pass = |kernel: String,
                srcs: &[*const u8],
                dst: *const u8,
                extra: &[*const u8],
                count,
                chunk,
                chunks: usize,
                keep| {
        let mut buffers = srcs.to_vec();
        buffers.push(dst);
        buffers.extend(extra);
        if rows {
            let args = [u64_arg(count), u64_arg(chunk)];
            launch(
                &kernel,
                &buffers,
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
            launch(&kernel, &buffers, &args, grid, keep, name)
        }
    };
    let first = |own: String| fused.map_or(own, str::to_owned);
    if chunks == 1 {
        let kernel = first(format!("{op}_{layout}_{dtype}"));
        return pass(kernel, inputs, output, extra, count, count, 1, keep);
    }
    if scratch.is_null() {
        return Err(format!(
            "{name}: a split reduction needs scratch for its partials: compile the graph for MPS (lumen.compile, Plan(graph, \"mps\"))"
        ));
    }
    // The first launch writes every partial before the second reads them.
    let p = scratch.cast_const();
    let kernel = first(format!("{op}_{layout}_partial_{dtype}"));
    pass(kernel, inputs, p, extra, count, chunk, chunks, keep.clone())?;
    let kernel = format!("{op}_{layout}_final_{dtype}");
    pass(kernel, &[p], output, &[], chunks, chunks, 1, keep)
}
