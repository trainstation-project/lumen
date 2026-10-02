//! dot_general on MPS (`mps.metal`) as a tiled matmul: each operand's
//! batch, free and contracting dimensions collapse into one each (any
//! matmul, batched or with transposed operands), or, where they do not, the
//! operand is first gathered into a contiguous copy in that order.

use crate::graph::Primitive::DotGeneral;
use crate::graph::TensorType;
use crate::graph::plan::Step;
use crate::graph::primitive::free_dims;
use crate::ops::layout::mps::gather;
use crate::ops::mps::{Grid, dims_arg, launch};
use crate::tensor::contiguous_strides;
use crate::{Device, Tensor, TensorOptions};

/// The output tile of a threadgroup (`mps.metal`): 128 x 64 for the float
/// kernels and 64 x 64 for their `_small` variants, which matmuls with
/// fewer than `SMALL_TILES` of the large tiles (or whose M fills less than
/// half of the last) take; 64 x 64 (`MM_TILE`) for the 8- to 32-bit integer
/// kernels and 32 x 32 (`WIDE_TILE`) for the 64-bit ones.
const FLOAT_TILE: (usize, usize) = (128, 64);
const SMALL_TILE: (usize, usize) = (64, 64);
const INT_TILE: (usize, usize) = (64, 64);
const WIDE_TILE: (usize, usize) = (32, 32);
const SMALL_TILES: usize = 32;

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    mut keep: Vec<Tensor>,
) -> Result<(), String> {
    let DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
    } = &step.primitive
    else {
        unreachable!("a dot_general")
    };
    let (lhs, rhs, out) = (&step.inputs[0].1, &step.inputs[1].1, &step.output.1);
    let name = step.primitive.name();
    // The operands' dimensions in matmul order: lhs [batch, free,
    // contracting], rhs [batch, contracting, free].
    let lhs_used = [lhs_batch.as_slice(), lhs_contracting].concat();
    let rhs_used = [rhs_batch.as_slice(), rhs_contracting].concat();
    let lhs_free: Vec<usize> = free_dims(lhs.shape.len(), &lhs_used).collect();
    let rhs_free: Vec<usize> = free_dims(rhs.shape.len(), &rhs_used).collect();
    let lhs_order = [lhs_batch.as_slice(), &lhs_free, lhs_contracting].concat();
    let rhs_order = [rhs_batch.as_slice(), rhs_contracting, &rhs_free].concat();
    let (nb, nk) = (lhs_batch.len(), lhs_contracting.len());
    let size =
        |ty: &TensorType, dims: &[usize]| dims.iter().map(|&d| ty.shape[d]).product::<usize>();
    let (b, m, n, k) = (
        size(lhs, lhs_batch),
        size(lhs, &lhs_free),
        size(rhs, &rhs_free),
        size(lhs, lhs_contracting),
    );
    let (lp, [lsb, lsm, lsk]) = operand(
        lhs,
        &lhs_order,
        nb,
        nb + lhs_free.len(),
        inputs[0],
        &mut keep,
        name,
    )?;
    let (rp, [rsb, rsk, rsn]) = operand(rhs, &rhs_order, nb, nb + nk, inputs[1], &mut keep, name)?;
    let large = b * m.div_ceil(FLOAT_TILE.0) * n.div_ceil(FLOAT_TILE.1);
    let small = large < SMALL_TILES || matches!(m % FLOAT_TILE.0, 1..=64);
    let (kernel, (tm, tn)) = match (out.dtype.is_float(), small) {
        (true, false) => (format!("matmul_{}", out.dtype), FLOAT_TILE),
        (true, true) => (format!("matmul_small_{}", out.dtype), SMALL_TILE),
        (false, _) if out.dtype.size_of() == 8 => (format!("matmul_{}", out.dtype), WIDE_TILE),
        (false, _) => (format!("matmul_{}", out.dtype), INT_TILE),
    };
    let grid = Grid::Groups([n.div_ceil(tn), m.div_ceil(tm), b]);
    let p = [m, n, k, lsb, lsm, lsk, rsb, rsk, rsn];
    launch(
        &kernel,
        &[lp, rp, output.cast_const()],
        &[dims_arg(p)],
        grid,
        keep,
        name,
    )
}

/// The operand `ty` at `ptr` with its dimensions in `order`, as three
/// strided dimensions: `order[..i]`, `order[i..j]` and `order[j..]`, each
/// collapsed into one. Where they do not collapse, gathers the operand into
/// a contiguous copy in `order` (kept alive in `keep`) and returns that.
fn operand(
    ty: &TensorType,
    order: &[usize],
    i: usize,
    j: usize,
    ptr: *const u8,
    keep: &mut Vec<Tensor>,
    name: &'static str,
) -> Result<(*const u8, [usize; 3]), String> {
    let strides = contiguous_strides(&ty.shape);
    let groups = [&order[..i], &order[i..j], &order[j..]];
    let collapsed: Option<Vec<(usize, usize)>> = groups
        .iter()
        .map(|g| collapse(g.iter().map(|&d| (ty.shape[d], strides[d]))))
        .collect();
    if let Some(c) = collapsed {
        return Ok((ptr, [c[0].1, c[1].1, c[2].1]));
    }
    let options = TensorOptions::new().dtype(ty.dtype).device(Device::Mps);
    // SAFETY: the gather writes every element before the matmul reads it.
    let copy = unsafe { Tensor::empty(&[ty.numel()], options) };
    keep.push(copy.clone());
    let shape: Vec<usize> = order.iter().map(|&d| ty.shape[d]).collect();
    let from: Vec<usize> = order.iter().map(|&d| strides[d]).collect();
    gather(
        ty.dtype.size_of(),
        &shape,
        &from,
        ptr,
        copy.data_ptr(),
        keep.clone(),
        name,
    )?;
    let inner = |g: &[usize]| g.iter().map(|&d| ty.shape[d]).product::<usize>();
    let (s1, s2) = (inner(groups[1]), inner(groups[2]));
    Ok((copy.data_ptr().cast_const(), [s1 * s2, s2, 1]))
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
