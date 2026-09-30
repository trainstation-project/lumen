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
use crate::{DType, Element, Scalar, Tensor};

unsafe extern "C" {
    // In lumen/ops/mps.mm.
    fn lumen_mps_launch_kernel(
        name: *const c_char,
        buffers: *const *const u8,
        nbuffers: usize,
        args: *const *const u8,
        arg_lens: *const usize,
        nargs: usize,
        threads: usize,
        groups: *const usize,
        group: *const usize,
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

/// `value` as one element of `dtype`, converted as the reference does.
fn element_arg(dtype: DType, value: Scalar) -> Vec<u8> {
    dispatch_dtype!(dtype, T => as_bytes(std::slice::from_ref(&T::from_scalar(value))).to_vec())
}

/// Elements an elementwise kernel's thread takes (`PER_THREAD` in
/// `lumen/ops/mps.metal`).
const PER_THREAD: usize = 4;

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
    let gather = |shape: &[usize], strides: Vec<usize>| {
        (
            format!("gather_{}", out.dtype.size_of()),
            vec![
                u32_arg(shape.len() as u32),
                dims_arg(shape.iter().copied()),
                dims_arg(strides),
            ],
        )
    };
    // Threadgroups for kernels that tile their output; others run a thread
    // per output element, or per PER_THREAD elements for elementwise ones.
    let n = out.numel();
    let mut groups = None;
    let mut threads = n;
    let elementwise = n.div_ceil(PER_THREAD);
    if n > u32::MAX as usize {
        return Err(format!(
            "{}: {n} elements are more than MPS kernels index",
            step.primitive.name()
        ));
    }
    let (kernel, bytes) = match &step.primitive {
        // One kernel per op and dtype, named after the primitive.
        p @ (Add | Sub | Mul | Div | Max | Eq | Lt | Neg | Exp | Log | Rsqrt | Tanh | Logistic) => {
            threads = elementwise;
            (
                format!("{}_{}", p.name(), args[0].dtype),
                vec![u32_arg(n as u32)],
            )
        }
        ConvertElementType { .. } => {
            threads = elementwise;
            (
                format!("convert_{}_{}", args[0].dtype, out.dtype),
                vec![u32_arg(n as u32)],
            )
        }
        Select => {
            threads = elementwise;
            (
                format!("select_{}", out.dtype.size_of()),
                vec![u32_arg(n as u32)],
            )
        }
        ReduceSum { axes } | ReduceMax { axes } => {
            let x = args[0];
            let strides = contiguous_strides(&x.shape);
            // In increasing order, so each output sums in the reference's
            // (row-major) order.
            let mut reduced = axes.clone();
            reduced.sort_unstable();
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
                groups = Some([p[1].div_ceil(32), p[0].div_ceil(32), b]);
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
        Reshape { .. } => gather(&[out.numel()], vec![1]),
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
            gather(&out.shape, strides)
        }
        Transpose { permutation } => {
            let xs = contiguous_strides(&args[0].shape);
            gather(&out.shape, permutation.iter().map(|&d| xs[d]).collect())
        }
        Full { fill_value, .. } => {
            threads = elementwise;
            (
                format!("fill_{}", out.dtype.size_of()),
                vec![element_arg(out.dtype, *fill_value), u32_arg(n as u32)],
            )
        }
        Iota { dimension, .. } => (
            format!("iota_{}", out.dtype),
            vec![
                u64_arg(out.shape[*dimension]),
                u64_arg(out.shape[dimension + 1..].iter().product()),
            ],
        ),
    };
    let mut buffers = inputs.to_vec();
    buffers.push(output.cast_const());
    launch(
        &kernel,
        &buffers,
        &bytes,
        threads,
        groups,
        keep,
        step.primitive.name(),
    )
}

/// Launch `kernel` over `threads` threads (one per output element), or
/// over `groups` of 16x16 threads.
fn launch(
    kernel: &str,
    buffers: &[*const u8],
    args: &[Vec<u8>],
    threads: usize,
    groups: Option<[usize; 3]>,
    keep: Vec<Tensor>,
    name: &'static str,
) -> Result<(), String> {
    if threads == 0 {
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
            threads,
            groups.as_ref().map_or(std::ptr::null(), |g| g.as_ptr()),
            [16, 16, 1].as_ptr(),
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
