//! Encodes ReduceSum and ReduceMax steps as Metal kernel launches, picking a strided, grouped, row or column kernel
//! and splitting long reductions into a partials pass plus a final pass.
use crate::graph::Primitive::{self, *};
use crate::graph::TensorType;
use crate::graph::primitive::free_dims;
use crate::graph_compiler::plan::Step;
use crate::kernels::mps::{Grid, dims_arg, dims32_arg, element_arg, launch, u32_arg, u64_arg};
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar, Tensor};

/// Encodes a plain reduction step reading its single input into `output`.
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
        step,
        &step.primitive,
        x,
        None,
        None,
        (name, name),
        name,
        &inputs[..1],
        output,
        &[],
        &[],
        scratch,
        keep,
    )
}

/// Which Metal reduction kernel family handles a reduction.
pub(crate) enum Layout {
    /// One thread per output over arbitrary strides; for inputs too large for 32-bit indexing.
    Generic,
    /// Several lanes per output with 32-bit indexing.
    Grouped,
    /// Contiguous reduced axes at the end: chunked per-row threadgroups.
    Rows,
    /// Contiguous reduced axes followed by kept ones: one thread per column chunk.
    Cols,
}

/// Chooses the kernel family for reducing `axes` of `x`.
pub(crate) fn layout(x: &TensorType, axes: &[usize]) -> Layout {
    let (x, reduced) = normalize(x, axes);
    let grouped = x.numel() <= u32::MAX as usize;
    // reduced axes form one block after normalisation; short trailing rows prefer the grouped kernel
    if reduced.windows(2).all(|w| w[1] == w[0] + 1) {
        let Split { count, b, .. } = split(&x, &reduced);
        let lanes = count.div_ceil((LANE_BYTES / x.dtype.size_of()).max(1));
        return match b {
            1 if grouped && lanes < REDUCE_THREADS => Layout::Grouped,
            1 => Layout::Rows,
            _ => Layout::Cols,
        };
    }
    match grouped {
        true => Layout::Grouped,
        false => Layout::Generic,
    }
}

/// Drops size-1 dims and merges adjacent dims that are both reduced or both kept; returns the new type and the
/// reduced dims' positions in it.
fn normalize(x: &TensorType, axes: &[usize]) -> (TensorType, Vec<usize>) {
    let (mut shape, mut reduced) = (Vec::new(), Vec::new());
    let mut last = None;
    for (d, &n) in x.shape.iter().enumerate().filter(|&(_, &n)| n != 1) {
        let r = axes.contains(&d);
        match (last == Some(r), shape.last_mut()) {
            (true, Some(merged)) => *merged *= n,
            _ => {
                if r {
                    reduced.push(shape.len());
                }
                shape.push(n);
            }
        }
        last = Some(r);
    }
    (TensorType::new(x.dtype, &shape), reduced)
}

/// Encodes reduction `op` over `x`, running kernel `fused` (and `last` for the final pass) in place of the
/// built-in ones when a fused prologue/epilogue kernel was generated.
///
/// `launches` names the partial and final passes for the profiler; `extra` buffers follow the output and
/// `scalars` precede the reduction's own arguments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_reduction(
    step: &Step,
    op: &Primitive,
    x: &TensorType,
    fused: Option<&str>,
    last: Option<&str>,
    launches: (&'static str, &'static str),
    name: &'static str,
    inputs: &[*const u8],
    output: *mut u8,
    extra: &[*const u8],
    scalars: &[Vec<u8>],
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (ReduceSum { axes, .. } | ReduceMax { axes }) = op else {
        unreachable!("a reduction")
    };
    let (op_name, accum) = (op.name(), accum_dtype(op, x.dtype));
    let dtype = kernel_dtype(x.dtype, accum);
    let kernel = |own: String| fused.map_or(own, str::to_owned);
    let mut buffers = inputs.to_vec();
    buffers.push(output.cast_const());
    buffers.extend(extra);
    let (x, reduced) = normalize(x, axes);
    let x = &x;
    let layout = layout(x, &reduced);
    if let Layout::Rows | Layout::Cols = layout {
        return consecutive(
            step, op_name, name, x, accum, &reduced, fused, last, launches, inputs, output, extra,
            scalars, scratch, keep,
        );
    }
    let strides = contiguous_strides(&x.shape);
    let kept: Vec<usize> = free_dims(x.shape.len(), &reduced).collect();
    let init = match op {
        ReduceSum { .. } => Scalar::Int(0),
        _ => lowest(accum),
    };
    let count: usize = reduced.iter().map(|&d| x.shape[d]).product();
    let outputs: usize = kept.iter().map(|&d| x.shape[d]).product();
    let reduced_sizes = reduced.iter().map(|&d| x.shape[d]);
    let reduced_strides = reduced.iter().map(|&d| strides[d]);
    let mut args = vec![
        element_arg(accum, init),
        u32_arg(kept.len() as u32),
        dims_arg(kept.iter().map(|&d| x.shape[d])),
        dims_arg(kept.iter().map(|&d| strides[d])),
        u32_arg(reduced.len() as u32),
    ];
    // pick a power-of-two lane count so each lane reads about LANE_BYTES, capped at one threadgroup
    let (kernel, grid) = if let Layout::Grouped = layout {
        let per_lane = (LANE_BYTES / x.dtype.size_of()).max(1);
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
    let args: Vec<Vec<u8>> = scalars.iter().cloned().chain(args).collect();
    launch(&kernel, &buffers, &args, grid, keep, name)
}

/// Threads per threadgroup; must match REDUCE_THREADS in kernels.metal.
pub(crate) const REDUCE_THREADS: usize = 256;
/// Bytes each lane of a grouped reduction should read.
const LANE_BYTES: usize = 32;

/// Returns the identity of max for `dtype`: -inf for floats, the minimum for signed ints, 0 otherwise.
fn lowest(dtype: DType) -> Scalar {
    match dtype {
        d if d.is_float() => Scalar::Float(f64::NEG_INFINITY),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            Scalar::Int(i64::MIN >> (64 - 8 * dtype.size_of()))
        }
        _ => Scalar::Int(0),
    }
}

/// A contiguous reduction viewed as `(a, count, b)` with the reduced axis split into `chunks` of `chunk` elements.
struct Split {
    a: usize,
    count: usize,
    b: usize,
    chunks: usize,
    chunk: usize,
}

/// Views `x` as `(a, count, b)` around the reduced block and splits `count` into chunks so the launch has
/// enough parallelism when the kept dims are small.
fn split(x: &TensorType, reduced: &[usize]) -> Split {
    let first = reduced.first().copied().unwrap_or(0);
    let end = reduced.last().map_or(first, |last| last + 1);
    let a: usize = x.shape[..first].iter().product();
    let count: usize = x.shape[first..end].iter().product();
    let b: usize = x.shape[end..].iter().product();
    // aim for about 256 threadgroups on rows and 65536 threads on columns, without chunks shorter than 4096/64;
    // an empty output has nothing to split
    let chunks = if a * b == 0 {
        1
    } else if b == 1 {
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

/// Reports whether reducing `axes` of `x` runs in two passes through scratch.
pub(crate) fn splits(x: &TensorType, axes: &[usize]) -> bool {
    scratch_bytes(x, axes, x.dtype) > 0
}

/// Returns the accumulator dtype: ReduceSum's declared one, otherwise the input dtype.
pub(crate) fn accum_dtype(op: &Primitive, dtype: DType) -> DType {
    match op {
        ReduceSum { accum_dtype, .. } => *accum_dtype,
        _ => dtype,
    }
}

/// Returns the kernel name suffix: the input dtype, plus `_accum` when accumulating in a different one.
pub(crate) fn kernel_dtype(x: DType, accum: DType) -> String {
    match x == accum {
        true => x.to_string(),
        false => format!("{x}_{accum}"),
    }
}

/// Returns the scratch bytes for the `(a, chunks, b)` partials of a split reduction, or 0 when it runs in one pass.
pub(crate) fn scratch_bytes(x: &TensorType, axes: &[usize], accum: DType) -> usize {
    let (x, reduced) = normalize(x, axes);
    if !reduced.windows(2).all(|w| w[1] == w[0] + 1) {
        return 0;
    }
    let Split { a, b, chunks, .. } = split(&x, &reduced);
    if chunks == 1 || a * b == 0 {
        return 0;
    }
    a * chunks * b * accum.size_of()
}

/// Launches the row or column kernel over the `(a, count, b)` view, in one pass or as a partials pass into
/// `scratch` followed by a pass reducing the partials.
#[allow(clippy::too_many_arguments)]
fn consecutive(
    step: &Step,
    op: &'static str,
    name: &'static str,
    x: &TensorType,
    accum: DType,
    reduced: &[usize],
    fused: Option<&str>,
    last: Option<&str>,
    launches: (&'static str, &'static str),
    inputs: &[*const u8],
    output: *mut u8,
    extra: &[*const u8],
    scalars: &[Vec<u8>],
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
    let (dtype, rows) = (kernel_dtype(x.dtype, accum), b == 1);
    if a * b == 0 {
        return Ok(());
    }
    let layout = if rows { "rows" } else { "cols" };
    // one launch over `count` elements in chunks of `chunk`; rows use a threadgroup per (chunk, row)
    let pass = |kernel: String,
                srcs: &[*const u8],
                dst: *const u8,
                extra: &[*const u8],
                scalars: &[Vec<u8>],
                count,
                chunk,
                chunks: usize,
                keep,
                name: &'static str| {
        let mut buffers = srcs.to_vec();
        buffers.push(dst);
        buffers.extend(extra);
        let args =
            |own: &[Vec<u8>]| -> Vec<Vec<u8>> { scalars.iter().chain(own).cloned().collect() };
        if rows {
            let args = args(&[u64_arg(count), u64_arg(chunk)]);
            launch(
                &kernel,
                &buffers,
                &args,
                Grid::Groups([chunks, a, 1]),
                keep,
                name,
            )
        } else {
            let args = args(&[
                u32_arg(b as u32),
                u64_arg(count),
                u64_arg(chunk),
                u32_arg(chunks as u32),
            ]);
            let grid = Grid::Threads([a * chunks * b, 1, 1]);
            launch(&kernel, &buffers, &args, grid, keep, name)
        }
    };
    let first = |own: String| fused.map_or(own, str::to_owned);
    if chunks == 1 {
        let kernel = first(format!("{op}_{layout}_{dtype}"));
        return pass(
            kernel, inputs, output, extra, scalars, count, count, 1, keep, name,
        );
    }
    if scratch.is_null() {
        return Err(format!(
            "{name}: a split reduction needs scratch for its partials: compile the graph for MPS (lumen.compile, Plan(graph, \"mps\"))"
        ));
    }
    let p = scratch.cast_const();
    let partials = TensorType::new(accum, &[a, chunks, b]);
    // the profiler sees the first pass writing the partials tensor instead of the step's output
    crate::profiler::next_launch(launches.0, || {
        let extra = step.extra_outputs.iter().map(|(_, ty)| ty.clone());
        let mut output_strides = step.output_strides();
        output_strides[0] = contiguous_strides(&partials.shape);
        crate::profiler::IoTypes {
            inputs: step.inputs.iter().map(|(_, ty)| ty.clone()).collect(),
            outputs: std::iter::once(partials.clone()).chain(extra).collect(),
            input_strides: step.input_strides(),
            output_strides,
        }
    });
    let kernel = first(format!("{op}_{layout}_{dtype}"));
    pass(
        kernel,
        inputs,
        p,
        extra,
        scalars,
        count,
        chunk,
        chunks,
        keep.clone(),
        name,
    )?;
    let kernel = last.map_or_else(|| format!("{op}_{layout}_{accum}"), str::to_owned);
    crate::profiler::next_launch(launches.1, || crate::profiler::IoTypes {
        inputs: vec![partials.clone()],
        outputs: vec![step.output.1.clone()],
        input_strides: vec![contiguous_strides(&partials.shape)],
        output_strides: step.output_strides()[..1].to_vec(),
    });
    pass(
        kernel,
        &[p],
        output,
        &[],
        &[],
        chunks,
        chunks,
        1,
        keep,
        name,
    )
}
