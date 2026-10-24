mod config;
mod linear_cross_entropy;
mod mpp;
mod skinny;
mod tiled;

pub(crate) use config::*;
pub(crate) use linear_cross_entropy::*;
pub(crate) use mpp::*;
pub(crate) use skinny::*;
pub(crate) use tiled::*;

use crate::Tensor;
use crate::graph::Primitive::{self, DotGeneral};
use crate::graph::TensorType;
use crate::graph::primitive::free_dims;
use crate::graph_compiler::plan::Step;
use crate::kernels::mps::{Grid, dims_arg, launch};
use crate::tensor::contiguous_strides;

const FLOAT_TILE: (usize, usize) = (128, 64);
const MID_TILE: (usize, usize) = (64, 64);
const SMALL_TILE: (usize, usize) = (32, 32);
const INT_TILE: (usize, usize) = (64, 64);
const WIDE_TILE: (usize, usize) = (32, 32);
const SMALL_TILES: usize = 32;

pub(crate) struct MatmulOrder {
    pub lhs: Vec<usize>,
    pub rhs: Vec<usize>,
    pub lhs_split: (usize, usize),
    pub rhs_split: (usize, usize),
}

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

fn float_tiles(b: usize, m: usize, n: usize) -> (bool, bool, (usize, usize)) {
    let large = b * m.div_ceil(FLOAT_TILE.0) * n.div_ceil(FLOAT_TILE.1);
    let small = large < SMALL_TILES || matches!(m % FLOAT_TILE.0, 1..=32);
    let mid = !small && matches!(m % FLOAT_TILE.0, 33..=64);
    let tile = match (small, mid) {
        (true, _) => SMALL_TILE,
        (_, true) => MID_TILE,
        _ => FLOAT_TILE,
    };
    (small, mid, tile)
}

pub(crate) fn float_groups(b: usize, m: usize, n: usize) -> usize {
    let (_, _, (tm, tn)) = float_tiles(b, m, n);
    b * m.div_ceil(tm) * n.div_ceil(tn)
}

pub(crate) struct MatmulLaunch {
    pub small: bool,
    pub mid: bool,
    pub grid: Grid,
    pub p: [usize; 9],
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_matmul(
    prim: &Primitive,
    lhs: &TensorType,
    rhs: &TensorType,
    out: &TensorType,
    lhs_strides: &[usize],
    rhs_strides: &[usize],
    name: &str,
    tile: Option<(usize, usize)>,
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
    let float = out.dtype.is_float();
    let (small, mid, tile) = match tile {
        None => float_tiles(b, m, n),
        Some(tile) => (false, false, tile),
    };
    let (small, mid) = (float && small, float && mid);
    let (tm, tn) = match float {
        true => tile,
        false if out.dtype.size_of() == 8 => WIDE_TILE,
        false => INT_TILE,
    };
    Ok(MatmulLaunch {
        small,
        mid,
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
        None,
    )?;
    launch(
        &format!("matmul_{}", out.dtype),
        &[inputs[0], inputs[1], output.cast_const()],
        &[dims_arg(launch_.p)],
        launch_.grid,
        keep,
        name,
    )
}

fn collapse(dims: impl IntoIterator<Item = (usize, usize)>) -> Option<(usize, usize)> {
    let dims: Vec<(usize, usize)> = dims.into_iter().filter(|d| d.0 != 1).collect();
    if dims.windows(2).any(|w| w[0].1 != w[1].0 * w[1].1) {
        return None;
    }
    Some((
        dims.iter().map(|d| d.0).product(),
        dims.last().map_or(0, |d| d.1),
    ))
}
