//! Plan steps as Metal kernels on MPS: [`encode`] picks a step's kernel
//! (in `lumen/ops/<op>/mps.metal`: elementwise, reduce, dot_general,
//! layout, factory) and computes its arguments (shapes, strides, op codes),
//! and `lumen/ops/mps.mm` launches it into lumen's MPS stream without
//! waiting (see [`crate::stream::mps`]). Values are contiguous, so a kernel only
//! needs strides where it reads in another order (broadcast, transpose,
//! reductions and contractions).

use std::ffi::{CString, c_char, c_void};

use super::plan::Step;
use super::primitive::free_dims;
use super::{Primitive, TensorType};
use crate::stream::mps::{self, Completion};
use crate::tensor::contiguous_strides;
use crate::tensor::dtype::dispatch_dtype;
use crate::tensor::storage::as_bytes;
use crate::{DType, Element, Scalar, Tensor, TensorOptions};

unsafe extern "C" {
    // In lumen/ops/mps.mm.
    fn lumen_mps_launch_kernel(
        name: *const c_char,
        buffers: *const *const u8,
        nbuffers: usize,
        args: *const *const u8,
        arg_lens: *const usize,
        nargs: usize,
        grid: *const usize,
        groups: i32,
        timed: i32,
        done: Completion,
        context: *mut c_void,
    ) -> i32;
}

fn u32_arg(v: u32) -> Vec<u8> {
    v.to_ne_bytes().to_vec()
}

fn u64_arg(v: usize) -> Vec<u8> {
    (v as u64).to_ne_bytes().to_vec()
}

/// A `constant ulong *` argument (never empty: Metal binds no zero-length
/// bytes, and the kernels read only the first `ndim`).
fn dims_arg(v: impl IntoIterator<Item = usize>) -> Vec<u8> {
    let mut v: Vec<u64> = v.into_iter().map(|d| d as u64).collect();
    if v.is_empty() {
        v.push(0);
    }
    as_bytes(&v).to_vec()
}

/// [`dims_arg`] as `constant uint *`, for kernels indexing in 32 bits
/// (their tensors have fewer than 2^32 elements).
fn dims32_arg(v: impl IntoIterator<Item = usize>) -> Vec<u8> {
    let mut v: Vec<u32> = v.into_iter().map(|d| d as u32).collect();
    if v.is_empty() {
        v.push(0);
    }
    as_bytes(&v).to_vec()
}

/// `value` as one element of `dtype`, converted as the reference does.
fn element_arg(dtype: DType, value: Scalar) -> Vec<u8> {
    dispatch_dtype!(dtype, T => as_bytes(std::slice::from_ref(&T::from_scalar(value))).to_vec())
}

/// Bytes of elements an elementwise kernel's thread takes
/// (`BYTES_PER_THREAD` in `lumen/ops/mps.metal`).
const BYTES_PER_THREAD: usize = 16;

/// The elements of `dtype` a thread of an elementwise kernel takes.
fn per_thread(dtype: DType) -> usize {
    (BYTES_PER_THREAD / dtype.size_of()).max(1)
}

/// `dims` (each a size and stride, outermost first) as one dimension of
/// the same elements, if they are laid out as one: its size and stride.
fn collapse(dims: impl IntoIterator<Item = (usize, usize)>) -> Option<(usize, usize)> {
    // Size-1 dimensions can have any stride.
    let dims: Vec<(usize, usize)> = dims.into_iter().filter(|d| d.0 != 1).collect();
    if dims.windows(2).any(|w| w[0].1 != w[1].0 * w[1].1) {
        return None;
    }
    Some((
        dims.iter().map(|d| d.0).product(),
        dims.last().map_or(0, |d| d.1),
    ))
}

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

/// Encode `step` reading `inputs` and writing `output` (null for a value
/// with no elements), keeping `keep` alive until the GPU has run it.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    use Primitive::*;
    let args: Vec<&TensorType> = step.inputs.iter().map(|(_, ty)| ty).collect();
    let out = &step.output.1;
    // A gather of `shape` from the input at `strides`: a thread per element,
    // on a grid of (innermost dimension, rows).
    let gather = |shape: &[usize], strides: Vec<usize>| {
        let (inner, inner_stride) = (
            shape.last().copied().unwrap_or(1),
            strides.last().copied().unwrap_or(0),
        );
        let outer = &shape[..shape.len().saturating_sub(1)];
        let args = vec![
            u32_arg(outer.len() as u32),
            dims32_arg(outer.iter().copied()),
            dims32_arg(strides[..outer.len()].iter().copied()),
            u32_arg(inner_stride as u32),
        ];
        let rows = outer.iter().product();
        (
            format!("gather_{}", out.dtype.size_of()),
            args,
            Grid::Threads([inner, rows, 1]),
        )
    };
    // Threadgroups for kernels that tile their output; others run a thread
    // per output element, or per PER_THREAD elements for elementwise ones.
    let n = out.numel();
    let mut grid = Grid::Threads([n, 1, 1]);
    // An elementwise kernel's grid, for elements of `dtype` a thread.
    let elementwise = |dtype| Grid::Threads([n.div_ceil(per_thread(dtype)), 1, 1]);
    if n > u32::MAX as usize {
        return Err(format!(
            "{}: {n} elements are more than MPS kernels index",
            step.primitive.name()
        ));
    }
    let (kernel, bytes) = match &step.primitive {
        // One kernel per op and dtype, named after the primitive.
        p @ (Add | Sub | Mul | Div | Max | Eq | Lt | Neg | Exp | Log | Rsqrt | Tanh | Logistic) => {
            grid = elementwise(args[0].dtype);
            (
                format!("{}_{}", p.name(), args[0].dtype),
                vec![u32_arg(n as u32)],
            )
        }
        ConvertElementType { .. } => {
            grid = elementwise(args[0].dtype);
            (
                format!("convert_{}_{}", args[0].dtype, out.dtype),
                vec![u32_arg(n as u32)],
            )
        }
        Select => {
            grid = elementwise(out.dtype);
            (
                format!("select_{}", out.dtype.size_of()),
                vec![u32_arg(n as u32)],
            )
        }
        ReduceSum { axes } | ReduceMax { axes } => {
            let x = args[0];
            let strides = contiguous_strides(&x.shape);
            let mut reduced = axes.clone();
            reduced.sort_unstable();
            if reduced.windows(2).all(|w| w[1] == w[0] + 1) {
                return reduce_consecutive(step, x, &reduced, inputs[0], output, keep);
            }
            // Other axes: a thread per output, summing in the reference's
            // (row-major) order.
            let kept: Vec<usize> = free_dims(x.shape.len(), &reduced).collect();
            let init = match step.primitive {
                ReduceSum { .. } => Scalar::Int(0),
                _ => lowest(x.dtype),
            };
            (
                format!("{}_{}", step.primitive.name(), x.dtype),
                vec![
                    element_arg(x.dtype, init),
                    u32_arg(kept.len() as u32),
                    dims_arg(kept.iter().map(|&d| x.shape[d])),
                    dims_arg(kept.iter().map(|&d| strides[d])),
                    u32_arg(reduced.len() as u32),
                    dims_arg(reduced.iter().map(|&d| x.shape[d])),
                    dims_arg(reduced.iter().map(|&d| strides[d])),
                    u64_arg(reduced.iter().map(|&d| x.shape[d]).product()),
                ],
            )
        }
        DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            rhs_batch,
        } => {
            let (lhs, rhs) = (args[0], args[1]);
            let (ls, rs) = (
                contiguous_strides(&lhs.shape),
                contiguous_strides(&rhs.shape),
            );
            // Each output dimension as (size, lhs stride, rhs stride).
            let mut dims: Vec<(usize, usize, usize)> = lhs_batch
                .iter()
                .zip(rhs_batch)
                .map(|(&l, &r)| (lhs.shape[l], ls[l], rs[r]))
                .collect();
            let lhs_used = [lhs_batch.as_slice(), lhs_contracting].concat();
            let rhs_used = [rhs_batch.as_slice(), rhs_contracting].concat();
            dims.extend(free_dims(lhs.shape.len(), &lhs_used).map(|d| (lhs.shape[d], ls[d], 0)));
            dims.extend(free_dims(rhs.shape.len(), &rhs_used).map(|d| (rhs.shape[d], 0, rs[d])));
            let contracting: Vec<(usize, usize, usize)> = lhs_contracting
                .iter()
                .zip(rhs_contracting)
                .map(|(&l, &r)| (lhs.shape[l], ls[l], rs[r]))
                .collect();
            // The tiled kernel, when the batch, lhs free, rhs free and
            // contracting dimensions each collapse into one: any matmul,
            // batched or with transposed operands.
            let (nb, nl) = (lhs_batch.len(), lhs.shape.len() - lhs_used.len());
            let one = |dims: &[(usize, usize, usize)],
                       pick: fn(&(usize, usize, usize)) -> usize| {
                collapse(dims.iter().map(|d| (d.0, pick(d))))
            };
            let matmul = (|| {
                let (b, lsb) = one(&dims[..nb], |d| d.1)?;
                let (_, rsb) = one(&dims[..nb], |d| d.2)?;
                let (m, lsm) = one(&dims[nb..nb + nl], |d| d.1)?;
                let (n, rsn) = one(&dims[nb + nl..], |d| d.2)?;
                let (k, lsk) = one(&contracting, |d| d.1)?;
                let (_, rsk) = one(&contracting, |d| d.2)?;
                Some((b, [m, n, k, lsb, lsm, lsk, rsb, rsk, rsn]))
            })();
            if let Some((b, p)) = matmul {
                grid = Grid::Groups([p[1].div_ceil(32), p[0].div_ceil(32), b]);
                (format!("matmul_{}", out.dtype), vec![dims_arg(p)])
            } else {
                (
                    format!("dot_{}", out.dtype),
                    vec![
                        u32_arg(dims.len() as u32),
                        dims_arg(dims.iter().map(|d| d.0)),
                        dims_arg(dims.iter().map(|d| d.1)),
                        dims_arg(dims.iter().map(|d| d.2)),
                        u32_arg(contracting.len() as u32),
                        dims_arg(contracting.iter().map(|d| d.0)),
                        dims_arg(contracting.iter().map(|d| d.1)),
                        dims_arg(contracting.iter().map(|d| d.2)),
                        u64_arg(contracting.iter().map(|d| d.0).product()),
                    ],
                )
            }
        }
        // A plan's reshapes are copies (the others alias their operand).
        Reshape { .. } => {
            let (kernel, bytes, g) = gather(&[out.numel()], vec![1]);
            grid = g;
            (kernel, bytes)
        }
        BroadcastInDim {
            broadcast_dimensions,
            ..
        } => {
            let x = args[0];
            let xs = contiguous_strides(&x.shape);
            let mut strides = vec![0; out.shape.len()];
            for (k, &d) in broadcast_dimensions.iter().enumerate() {
                if x.shape[k] != 1 {
                    strides[d] = xs[k];
                }
            }
            let (kernel, bytes, g) = gather(&out.shape, strides);
            grid = g;
            (kernel, bytes)
        }
        Transpose { permutation } => {
            let x = args[0];
            if let Some((batch, rows, cols)) = swapped_blocks(&x.shape, permutation) {
                grid = Grid::Groups([cols.div_ceil(32), rows.div_ceil(32), batch]);
                let bytes = vec![u32_arg(rows as u32), u32_arg(cols as u32)];
                (format!("transpose_{}", out.dtype.size_of()), bytes)
            } else {
                let xs = contiguous_strides(&x.shape);
                let (kernel, bytes, g) =
                    gather(&out.shape, permutation.iter().map(|&d| xs[d]).collect());
                grid = g;
                (kernel, bytes)
            }
        }
        Full { fill_value, .. } => {
            grid = elementwise(out.dtype);
            (
                format!("fill_{}", out.dtype.size_of()),
                vec![element_arg(out.dtype, *fill_value), u32_arg(n as u32)],
            )
        }
        Iota { dimension, .. } => {
            let (outer, inner) = (&out.shape[..*dimension], &out.shape[dimension + 1..]);
            let size = out.shape[*dimension];
            grid = Grid::Threads([inner.iter().product(), size, outer.iter().product()]);
            (format!("iota_{}", out.dtype), vec![])
        }
    };
    let mut buffers = inputs.to_vec();
    buffers.push(output.cast_const());
    launch(&kernel, &buffers, &bytes, grid, keep, step.primitive.name())
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

/// Reduce `x` over the consecutive axes `reduced`, viewed as [a, count, b]
/// (`lumen/ops/reduce/mps.metal`): rows when b = 1, columns otherwise. One
/// launch, or, with too few outputs to fill the GPU, two: `count` split
/// into chunks reduced in parallel into partials, then the partials.
fn reduce_consecutive(
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
                launch(
                    &kernel,
                    &[src, dst],
                    &args,
                    Grid::Threads([a * chunks * b, 1, 1]),
                    keep,
                    name,
                )
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
    pass(
        format!("{name}_{layout}_final_{dtype}"),
        p,
        output,
        chunks,
        chunks,
        1,
        keep,
    )
}

/// How a kernel's threads are laid out.
#[derive(Debug, Clone, Copy)]
enum Grid {
    /// A thread per index of an x-by-y-by-z grid (x varying fastest).
    Threads([usize; 3]),
    /// That many threadgroups of 16x16 threads.
    Groups([usize; 3]),
}

/// Launch `kernel` over `grid`.
fn launch(
    kernel: &str,
    buffers: &[*const u8],
    args: &[Vec<u8>],
    grid: Grid,
    keep: Vec<Tensor>,
    name: &'static str,
) -> Result<(), String> {
    let (sizes, groups) = match grid {
        Grid::Threads(sizes) => (sizes, false),
        Grid::Groups(sizes) => (sizes, true),
    };
    if sizes.contains(&0) {
        return Ok(());
    }
    let c_name = CString::new(kernel).expect("kernel names have no NUL");
    let pointers: Vec<*const u8> = args.iter().map(|a| a.as_ptr()).collect();
    let lens: Vec<usize> = args.iter().map(Vec::len).collect();
    let (context, done, timed) = mps::submit(keep, name);
    let status = unsafe {
        lumen_mps_launch_kernel(
            c_name.as_ptr(),
            buffers.as_ptr(),
            buffers.len(),
            pointers.as_ptr(),
            lens.as_ptr(),
            args.len(),
            sizes.as_ptr(),
            groups.into(),
            timed.into(),
            done,
            context,
        )
    };
    if status != 0 {
        mps::cancel(context);
        return Err(format!("the MPS kernel {kernel} could not run ({status})"));
    }
    Ok(())
}
