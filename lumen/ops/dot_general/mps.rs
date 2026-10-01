//! dot_general on MPS (`mps.metal`): a tiled kernel when the contraction
//! is a matmul (the batch, lhs free, rhs free and contracting dimensions
//! each collapse into one: any matmul, batched or with transposed
//! operands), a naive one, a thread per output, otherwise.

use crate::Tensor;
use crate::graph::Primitive::DotGeneral;
use crate::graph::plan::Step;
use crate::graph::primitive::free_dims;
use crate::ops::mps::{Grid, dims_arg, launch_step, u32_arg, u64_arg};
use crate::tensor::contiguous_strides;

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
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
    let (nb, nl) = (lhs_batch.len(), lhs.shape.len() - lhs_used.len());
    let one = |dims: &[(usize, usize, usize)], pick: fn(&(usize, usize, usize)) -> usize| {
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
        let grid = Grid::Groups([p[1].div_ceil(32), p[0].div_ceil(32), b]);
        let kernel = format!("matmul_{}", out.dtype);
        return launch_step(step, &kernel, inputs, output, &[dims_arg(p)], grid, keep);
    }
    let args = [
        u32_arg(dims.len() as u32),
        dims_arg(dims.iter().map(|d| d.0)),
        dims_arg(dims.iter().map(|d| d.1)),
        dims_arg(dims.iter().map(|d| d.2)),
        u32_arg(contracting.len() as u32),
        dims_arg(contracting.iter().map(|d| d.0)),
        dims_arg(contracting.iter().map(|d| d.1)),
        dims_arg(contracting.iter().map(|d| d.2)),
        u64_arg(contracting.iter().map(|d| d.0).product()),
    ];
    let grid = Grid::Threads([out.numel(), 1, 1]);
    launch_step(
        step,
        &format!("dot_{}", out.dtype),
        inputs,
        output,
        &args,
        grid,
        keep,
    )
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
