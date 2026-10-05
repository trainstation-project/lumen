//! dot_general on MPS (`kernels.metal`) as a tiled matmul: each operand's
//! batch, free and contracting dimensions collapse into one each (any
//! matmul, batched or with transposed operands). The MPS compiler
//! ([`crate::compiler::mps`]) puts an operand whose dimensions do not into
//! that order with a transpose first, so this never copies.

use crate::Tensor;
use crate::graph::Primitive::{self, DotGeneral};
use crate::graph::TensorType;
use crate::graph::plan::Step;
use crate::graph::primitive::free_dims;
use crate::ops::mps::{Grid, dims_arg, launch};
use crate::tensor::contiguous_strides;

/// The output tile of a threadgroup (`kernels.metal`): 128 x 64 for the float
/// kernels and 64 x 64 for their `_small` variants, which matmuls with
/// fewer than `SMALL_TILES` of the large tiles (or whose M fills less than
/// half of the last) take; 64 x 64 (`MM_TILE`) for the 8- to 32-bit integer
/// kernels and 32 x 32 (`WIDE_TILE`) for the 64-bit ones.
const FLOAT_TILE: (usize, usize) = (128, 64);
pub(crate) const SMALL_TILE: (usize, usize) = (32, 32);
const INT_TILE: (usize, usize) = (64, 64);
const WIDE_TILE: (usize, usize) = (32, 32);
const SMALL_TILES: usize = 32;

/// A dot_general's operands' dimensions in matmul order, lhs [batch, free,
/// contracting] and rhs [batch, contracting, free], and where each splits
/// into those three groups.
pub(crate) struct MatmulOrder {
    pub lhs: Vec<usize>,
    pub rhs: Vec<usize>,
    pub lhs_split: (usize, usize),
    pub rhs_split: (usize, usize),
}

/// [`MatmulOrder`] of dot_general `p` on operands of ranks `lhs_rank` and
/// `rhs_rank`.
pub(crate) fn matmul_order(p: &Primitive, lhs_rank: usize, rhs_rank: usize) -> MatmulOrder {
    let DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
        ..
    } = p
    else {
        unreachable!("a dot_general")
    };
    let lhs_used = [lhs_batch.as_slice(), lhs_contracting].concat();
    let rhs_used = [rhs_batch.as_slice(), rhs_contracting].concat();
    let lhs_free: Vec<usize> = free_dims(lhs_rank, &lhs_used).collect();
    let rhs_free: Vec<usize> = free_dims(rhs_rank, &rhs_used).collect();
    let (nb, nk) = (lhs_batch.len(), lhs_contracting.len());
    MatmulOrder {
        lhs: [lhs_batch.as_slice(), &lhs_free, lhs_contracting].concat(),
        rhs: [rhs_batch.as_slice(), rhs_contracting, &rhs_free].concat(),
        lhs_split: (nb, nb + lhs_free.len()),
        rhs_split: (nb, nb + nk),
    }
}

/// The dimensions of `ty`, laid out at `strides` (in elements), in
/// `order`, split at `(i, j)`, as three strided dimensions (`order[..i]`,
/// `order[i..j]`, `order[j..]` each collapsed into one), if they collapse:
/// their strides.
pub(crate) fn collapsed(
    ty: &TensorType,
    strides: &[usize],
    order: &[usize],
    (i, j): (usize, usize),
) -> Option<[usize; 3]> {
    let mut out = [0; 3];
    for (k, group) in [&order[..i], &order[i..j], &order[j..]]
        .into_iter()
        .enumerate()
    {
        out[k] = collapse(group.iter().map(|&d| (ty.shape[d], strides[d])))?.1;
    }
    Some(out)
}

/// Whether dot_general `p` of operands of types `operands` can read its
/// operand `k` in place at `strides` (in elements): if its dimensions
/// collapse into matmul form at them.
pub(crate) fn reads_strided(
    p: &Primitive,
    operands: [&TensorType; 2],
    k: usize,
    strides: &[usize],
) -> bool {
    let order = matmul_order(p, operands[0].shape.len(), operands[1].shape.len());
    let (dims, split) = match k {
        0 => (&order.lhs, order.lhs_split),
        _ => (&order.rhs, order.rhs_split),
    };
    collapsed(operands[k], strides, dims, split).is_some()
}

/// How a dot's matmul launches: on the small tiles or not, its grid, and
/// its dimension arguments (`p` in `kernels.metal`).
pub(crate) struct MatmulLaunch {
    pub small: bool,
    pub grid: Grid,
    pub p: [usize; 9],
}

/// The launch of dot_general `prim` of `lhs` and `rhs`, read at
/// `lhs_strides` and `rhs_strides` (in elements), writing `out`.
pub(crate) fn plan_matmul(
    prim: &Primitive,
    lhs: &TensorType,
    rhs: &TensorType,
    out: &TensorType,
    lhs_strides: &[usize],
    rhs_strides: &[usize],
    name: &str,
) -> Result<MatmulLaunch, String> {
    let order = matmul_order(prim, lhs.shape.len(), rhs.shape.len());
    let not_matmul = || {
        format!(
            "{name}: an operand is not in matmul form: compile the graph for MPS (lumen.compile, Plan(graph, \"mps\"))"
        )
    };
    let [lsb, lsm, lsk] =
        collapsed(lhs, lhs_strides, &order.lhs, order.lhs_split).ok_or_else(not_matmul)?;
    let [rsb, rsk, rsn] =
        collapsed(rhs, rhs_strides, &order.rhs, order.rhs_split).ok_or_else(not_matmul)?;
    let size =
        |ty: &TensorType, dims: &[usize]| dims.iter().map(|&d| ty.shape[d]).product::<usize>();
    let (b, m) = (
        size(lhs, &order.lhs[..order.lhs_split.0]),
        size(lhs, &order.lhs[order.lhs_split.0..order.lhs_split.1]),
    );
    let (k, n) = (
        size(lhs, &order.lhs[order.lhs_split.1..]),
        size(rhs, &order.rhs[order.rhs_split.1..]),
    );
    let large = b * m.div_ceil(FLOAT_TILE.0) * n.div_ceil(FLOAT_TILE.1);
    let small = large < SMALL_TILES || matches!(m % FLOAT_TILE.0, 1..=64);
    let (tm, tn) = match (out.dtype.is_float(), small) {
        (true, false) => FLOAT_TILE,
        (true, true) => SMALL_TILE,
        (false, _) if out.dtype.size_of() == 8 => WIDE_TILE,
        (false, _) => INT_TILE,
    };
    Ok(MatmulLaunch {
        small,
        grid: Grid::Groups([n.div_ceil(tn), m.div_ceil(tm), b]),
        p: [m, n, k, lsb, lsm, lsk, rsb, rsk, rsn],
    })
}

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (lhs, rhs, out) = (&step.inputs[0].1, &step.inputs[1].1, &step.output.1);
    let name = step.label;
    // An operand read in place as a view of a larger buffer (a slice) has
    // its strides; any other is contiguous.
    let strides = |k: usize| {
        let ty = &step.inputs[k].1;
        step.views[k]
            .as_ref()
            .map_or_else(|| contiguous_strides(&ty.shape), |v| v.strides.clone())
    };
    let launch_ = plan_matmul(
        &step.primitive,
        lhs,
        rhs,
        out,
        &strides(0),
        &strides(1),
        name,
    )?;
    // Of the operands' dtype throughout (`matmul_<dtype>`), or accumulating
    // in float, writing the operands' or float (`matmul_bf16_f32_bf16`).
    let float = match (lhs.dtype, out.dtype) {
        (d, o) if d == o && o == accum_dtype(&step.primitive) => format!("{d}"),
        (d, o) => format!("{d}_{}_{o}", accum_dtype(&step.primitive)),
    };
    let kernel = match (out.dtype.is_float(), launch_.small) {
        (true, false) => format!("matmul_{float}"),
        (true, true) => format!("matmul_small_{float}"),
        (false, _) => format!("matmul_{}", out.dtype),
    };
    launch(
        &kernel,
        &[inputs[0], inputs[1], output.cast_const()],
        &[dims_arg(launch_.p)],
        launch_.grid,
        keep,
        name,
    )
}

/// The dtype dot_general `p` accumulates in.
fn accum_dtype(p: &Primitive) -> crate::DType {
    let DotGeneral { accum_dtype, .. } = p else {
        unreachable!("a dot_general")
    };
    *accum_dtype
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
