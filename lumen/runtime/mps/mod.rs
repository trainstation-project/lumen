//! MPS kernel arguments: the argument packing and grids kernels launch with (`runtime::kernel::mps`
//! launches them), and the scratch a step's kernels need.

use crate::graph::{Primitive, TensorType};
use crate::tensor::dtype::dispatch_dtype;
use crate::tensor::storage::as_bytes;
use crate::{DType, Element, Scalar};

/// Scratch bytes a step's kernels need beyond its output, for primitives that use any.
pub(crate) fn scratch_bytes(p: &Primitive, inputs: &[&TensorType], _output: &TensorType) -> usize {
    match p {
        Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => {
            let accum = crate::codegen::reduction::mps::accum_dtype(p, inputs[0].dtype);
            crate::codegen::reduction::mps::scratch_bytes(inputs[0], axes, accum)
        }
        Primitive::Cumsum {
            axis, accum_dtype, ..
        } => crate::codegen::scan::mps::scratch_bytes(inputs[0], *axis, *accum_dtype),
        Primitive::Fusion { body, .. } => crate::graph_compiler::mps::fusion_scratch_bytes(body),
        Primitive::LinearCrossEntropy => {
            crate::codegen::dot_general::mps::linear_cross_entropy_scratch(inputs[0], inputs[1])
        }
        Primitive::ScatterAdd { axis } => {
            crate::codegen::scatter_add::mps::scratch_bytes(inputs[0], inputs[1], *axis)
        }
        _ => 0,
    }
}

/// Packs a `u32` kernel argument.
pub(crate) fn u32_arg(v: u32) -> Vec<u8> {
    v.to_ne_bytes().to_vec()
}

/// Packs a `usize` as a `u64` kernel argument.
pub(crate) fn u64_arg(v: usize) -> Vec<u8> {
    (v as u64).to_ne_bytes().to_vec()
}

/// Packs dimensions as a `u64` array argument; empty becomes `[0]` so the buffer is never zero-sized.
pub(crate) fn dims_arg(v: impl IntoIterator<Item = usize>) -> Vec<u8> {
    let mut v: Vec<u64> = v.into_iter().map(|d| d as u64).collect();

    if v.is_empty() {
        v.push(0);
    }

    as_bytes(&v).to_vec()
}

/// Packs dimensions as a `u32` array argument; empty becomes `[0]` so the buffer is never zero-sized.
pub(crate) fn dims32_arg(v: impl IntoIterator<Item = usize>) -> Vec<u8> {
    let mut v: Vec<u32> = v.into_iter().map(|d| d as u32).collect();

    if v.is_empty() {
        v.push(0);
    }

    as_bytes(&v).to_vec()
}

/// Packs `value` converted to `dtype` as a single-element argument.
pub(crate) fn element_arg(dtype: DType, value: Scalar) -> Vec<u8> {
    dispatch_dtype!(dtype, T => as_bytes(std::slice::from_ref(&T::from_scalar(value))).to_vec())
}

/// Bytes each elementwise thread handles; matches `BYTES_PER_THREAD` in `kernels.metal`.
pub(crate) const BYTES_PER_THREAD: usize = 16;
/// Thread count above which elementwise grids start giving threads several elements.
const GRID_THREADS: usize = 32 << 10;

/// One-dimensional grid for an elementwise kernel over `n` elements, giving each thread up to 16 bytes once `n` is large.
pub(crate) fn elementwise_grid(n: usize, dtype: DType) -> Grid {
    let most = (BYTES_PER_THREAD / dtype.size_of()).max(1);
    Grid::Threads([n.div_ceil((n / GRID_THREADS).clamp(1, most)), 1, 1])
}

pub(crate) use crate::custom_op::mps::Grid;
