//! Graph-plan steps as Metal kernels on MPS. [`encode`] hands each step to
//! its op (`lumen/ops/<op>/mps.rs`), which picks the kernel in its
//! `mps.metal` and computes the arguments (shapes, strides); [`launch`]
//! runs it through `mps.mm` into lumen's MPS stream without waiting (see
//! [`crate::stream::mps`]). Values are contiguous, so a kernel only needs
//! strides where it reads in another order (broadcast, transpose,
//! reductions and contractions).

use std::ffi::{CString, c_char, c_void};

use crate::graph::plan::Step;
use crate::graph::{Primitive, TensorType};
use crate::stream::mps::{self, Completion};
use crate::tensor::dtype::dispatch_dtype;
use crate::tensor::storage::as_bytes;
use crate::{DType, Element, Scalar, Tensor};

unsafe extern "C" {
    // In mps.mm.
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

/// Encode `step` reading `inputs` and writing `output` (null for a value
/// with no elements), with the workspace bytes the planner set aside for its
/// kernel at `scratch` (null if none; see [`scratch_bytes`]), keeping `keep`
/// alive until the GPU has run it.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    use Primitive::*;
    let n = step.output.1.numel();
    if n > u32::MAX as usize {
        return Err(format!(
            "{}: {n} elements are more than MPS kernels index",
            step.primitive.name()
        ));
    }

    if let ReduceSum { .. } | ReduceMax { .. } = step.primitive {
        return super::reduce::mps::encode(step, inputs, output, scratch, keep);
    }

    let encode = match step.primitive {
        Add
        | Sub
        | Mul
        | Div
        | Max
        | Eq
        | Lt
        | Neg
        | Exp
        | Log
        | Rsqrt
        | Tanh
        | Logistic
        | ConvertElementType { .. }
        | Select => super::elementwise::mps::encode,
        ReduceSum { .. } | ReduceMax { .. } => unreachable!("encoded above"),
        DotGeneral { .. } => super::dot_general::mps::encode,
        Reshape { .. } | BroadcastInDim { .. } | Transpose { .. } | Slice { .. } => {
            super::layout::mps::encode
        }
        Full { .. } | Iota { .. } => super::factory::mps::encode,
        Fusion { .. } => crate::compiler::mps::encode,
    };
    encode(step, inputs, output, keep)
}

/// The workspace bytes a step's MPS kernel needs while it runs, which the
/// planner sets aside ([`crate::graph::PlanOptions::scratch`]): a split
/// reduction's partials.
pub(crate) fn scratch_bytes(p: &Primitive, inputs: &[&TensorType], _output: &TensorType) -> usize {
    match p {
        Primitive::ReduceSum { axes } | Primitive::ReduceMax { axes } => {
            super::reduce::mps::scratch_bytes(inputs[0], axes)
        }
        _ => 0,
    }
}

pub(crate) fn u32_arg(v: u32) -> Vec<u8> {
    v.to_ne_bytes().to_vec()
}

pub(crate) fn u64_arg(v: usize) -> Vec<u8> {
    (v as u64).to_ne_bytes().to_vec()
}

/// A `constant ulong *` argument (never empty: Metal binds no zero-length
/// bytes, and the kernels read only the first `ndim`).
pub(crate) fn dims_arg(v: impl IntoIterator<Item = usize>) -> Vec<u8> {
    let mut v: Vec<u64> = v.into_iter().map(|d| d as u64).collect();
    if v.is_empty() {
        v.push(0);
    }
    as_bytes(&v).to_vec()
}

/// [`dims_arg`] as `constant uint *`, for kernels indexing in 32 bits
/// (their tensors have fewer than 2^32 elements).
pub(crate) fn dims32_arg(v: impl IntoIterator<Item = usize>) -> Vec<u8> {
    let mut v: Vec<u32> = v.into_iter().map(|d| d as u32).collect();
    if v.is_empty() {
        v.push(0);
    }
    as_bytes(&v).to_vec()
}

/// `value` as one element of `dtype`, converted as the reference does.
pub(crate) fn element_arg(dtype: DType, value: Scalar) -> Vec<u8> {
    dispatch_dtype!(dtype, T => as_bytes(std::slice::from_ref(&T::from_scalar(value))).to_vec())
}

/// Bytes of elements an elementwise kernel's thread takes
/// (`BYTES_PER_THREAD` in `mps.metal`).
pub(crate) const BYTES_PER_THREAD: usize = 16;

/// The grid of an elementwise kernel over `n` elements of `dtype`.
pub(crate) fn elementwise_grid(n: usize, dtype: DType) -> Grid {
    Grid::Threads([
        n.div_ceil((BYTES_PER_THREAD / dtype.size_of()).max(1)),
        1,
        1,
    ])
}

/// How a kernel's threads are laid out.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Grid {
    /// A thread per index of an x-by-y-by-z grid (x varying fastest).
    Threads([usize; 3]),
    /// That many threadgroups of 16x16 threads.
    Groups([usize; 3]),
}

/// Launch `kernel` for `step` over `grid`, on buffers `inputs` then
/// `output` and argument bytes `args`.
pub(crate) fn launch_step(
    step: &Step,
    kernel: &str,
    inputs: &[*const u8],
    output: *mut u8,
    args: &[Vec<u8>],
    grid: Grid,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let mut buffers = inputs.to_vec();
    buffers.push(output.cast_const());
    launch(kernel, &buffers, args, grid, keep, step.primitive.name())
}

/// Launch `kernel` over `grid`, recorded in the profiler as `name`.
pub(crate) fn launch(
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
    let (context, done, timed) = mps::submit(keep, name, kernel);
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
