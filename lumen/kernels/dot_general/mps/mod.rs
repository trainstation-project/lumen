mod config;

pub(crate) use config::*;

use indoc::formatdoc;

use crate::Tensor;
use crate::codegen::mps::{THREAD_ARGS, formatted};
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
const SKINNY_ROWS: usize = 32;
const SKINNY_WORK: usize = 1 << 24;

pub(crate) const MPP_PREFIX: &str = "matmul_mpp_";

pub(crate) fn mpp_transposes(step: &Step) -> Option<(bool, bool)> {
    let (lhs, rhs, out) = (&step.inputs[0].1, &step.inputs[1].1, &step.output.1);
    let (ls, rs) = (
        contiguous_strides(&lhs.shape),
        contiguous_strides(&rhs.shape),
    );
    let launch_ = plan_matmul(&step.primitive, lhs, rhs, out, &ls, &rs, step.label, None).ok()?;
    let [m, n, k, _, sm, sk, _, rk, rn] = launch_.p;
    let fits = [lhs.numel(), rhs.numel(), m * n].iter().all(|&e| e <= i32::MAX as usize);
    let left = match (sk, sm) {
        (1, _) => false,
        (_, 1) if k > 1 => true,
        _ => return None,
    };
    let right = match (rn, rk) {
        (1, _) => false,
        (_, 1) if n > 1 => true,
        _ => return None,
    };
    fits.then_some((left, right))
}

pub(crate) fn skinny(step: &Step) -> bool {
    let (lhs, rhs, out) = (&step.inputs[0].1, &step.inputs[1].1, &step.output.1);
    let (ls, rs) = (
        contiguous_strides(&lhs.shape),
        contiguous_strides(&rhs.shape),
    );
    plan_matmul(&step.primitive, lhs, rhs, out, &ls, &rs, step.label, None)
        .is_ok_and(|launch_| {
            let [m, n, k, ..] = launch_.p;
            m <= SKINNY_ROWS && launch_.p[8] == 1 && m * n * k >= SKINNY_WORK
        })
}

pub(crate) fn matmul_source(
    t: &str,
    a: &str,
    o: &str,
    c: GemmConfig,
    (left, right): (bool, bool),
) -> (String, String) {
    let GemmConfig {
        bm,
        bn,
        bk,
        threads,
        cols,
        prefetch,
        algorithm,
    } = c;
    let simds = threads / 32;
    let signature = |name: &str| {
        formatdoc! {"
            [[max_total_threads_per_threadgroup({threads})]]
            kernel void {name}(device const {t} *lhs [[buffer(0)]],
            device const {t} *rhs [[buffer(1)]],
            device {o} *out [[buffer(2)]],
            constant ulong *p [[buffer(3)]],
            {THREAD_ARGS}) {{"}
    };
    if algorithm == Algorithm::Skinny {
        let name = format!("matmul_skinny_{t}_{a}_{o}_{bm}x{bn}x{bk}_{threads}");
        let source = formatdoc! {"
            {}
            threadgroup float partials[{threads} / ({bn} / {bk}) * ({bn} + 1)];
            matmul_skinny<{t}, {o}, {bm}, {threads}, {bn}, {bk}>(lhs, rhs, out, p, partials, group, tid.x);
            }}
            ", signature(&name)};
        return (name, formatted(&source));
    }
    if algorithm == Algorithm::Mpp {
        let name = format!(
            "{MPP_PREFIX}{t}_{a}_{o}_{bm}x{bn}x{bk}_{threads}_{}{}",
            u8::from(left),
            u8::from(right)
        );
        let source = formatdoc! {"
            {}
            matmul_mpp<{t}, {o}, {bm}, {bn}, {bk}, {simds}, {left}, {right}>(lhs, rhs, out, p, group);
            }}
            ", signature(&name)};
        return (name, formatted(&source));
    }
    let name = format!(
        "matmul_{t}_{a}_{o}_{bm}x{bn}x{bk}_{threads}_{cols}_{}",
        u8::from(prefetch)
    );
    let source = formatdoc! {"
        {}
        threadgroup {t} lt[SG_LT_OF({bm}, {bk}, {t}, {a}, {simds})], rt[{bk} * {bn}];
        matmul_sg_impl<{t}, {a}, {o}, {bm}, {bn}, {bk}, {o},
        Same, false, false, false, false, false, false,
        {threads}, {cols}, {prefetch}>(
        lhs, rhs, out, p,
        lt, rt,
        group, tid.y * 16 + tid.x, sg, lane);
        }}
        ", signature(&name)};
    let source = formatted(&source);
    (name, source)
}

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
    let float = match (lhs.dtype, out.dtype) {
        (d, o) if d == o && o == accum_dtype(&step.primitive) => format!("{d}"),
        (d, o) => format!("{d}_{}_{o}", accum_dtype(&step.primitive)),
    };
    let kernel = match (out.dtype.is_float(), launch_.small, launch_.mid) {
        (true, true, _) => format!("matmul_small_{float}"),
        (true, _, true) => format!("matmul_mid_{float}"),
        (true, ..) => format!("matmul_{float}"),
        (false, ..) => format!("matmul_{}", out.dtype),
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

pub(crate) fn encode_tuned(
    step: &Step,
    kernel: &str,
    config: GemmConfig,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (lhs, rhs, out) = (&step.inputs[0].1, &step.inputs[1].1, &step.output.1);
    let (ls, rs) = (
        contiguous_strides(&lhs.shape),
        contiguous_strides(&rhs.shape),
    );
    let tile = Some((config.bm, config.bn));
    let launch_ = plan_matmul(&step.primitive, lhs, rhs, out, &ls, &rs, step.label, tile)?;
    let Grid::Groups(groups) = launch_.grid else {
        unreachable!("a matmul's grid")
    };
    let grid = Grid::GroupsOf {
        groups,
        threads: config.threads,
    };
    let buffers = [inputs[0], inputs[1], output.cast_const()];
    launch(
        kernel,
        &buffers,
        &[dims_arg(launch_.p)],
        grid,
        keep,
        step.label,
    )
}

fn accum_dtype(p: &Primitive) -> crate::DType {
    let DotGeneral { accum_dtype, .. } = p else {
        unreachable!("a dot_general")
    };
    *accum_dtype
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

pub(crate) fn linear_cross_entropy_scratch(h: &TensorType, w: &TensorType) -> usize {
    let (b, v) = (h.shape[0], w.shape[0]);
    2 * b * v.div_ceil(float_tiles(1, b, v).2.1) * 4
}

pub(crate) fn encode_linear_cross_entropy(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (h, w, target) = (&step.inputs[0].1, &step.inputs[1].1, &step.inputs[2].1);
    let (b, d, v) = (h.shape[0], h.shape[1], w.shape[0]);
    if b == 0 || v == 0 {
        return Ok(());
    }
    if scratch.is_null() {
        return Err(format!(
            "{}: needs scratch for its partials: compile the graph for MPS (lumen.compile, Plan(graph, \"mps\"))",
            step.label
        ));
    }
    let (small, mid, (tm, tn)) = float_tiles(1, b, v);
    let kind = match (small, mid) {
        (true, _) => "small_",
        (_, true) => "mid_",
        _ => "",
    };
    let tiles = v.div_ceil(tn);
    let (maxima, sums) = (
        scratch.cast_const(),
        unsafe { scratch.add(b * tiles * 4) }.cast_const(),
    );
    let p = dims_arg([b, v, d, 0, d, 1, 0, 1, d]);
    let partials = TensorType::new(crate::DType::F32, &[2, b, tiles]);
    crate::profiler::next_launch("linear_cross_entropy (partials)", || {
        crate::profiler::IoTypes {
            inputs: step.inputs.iter().map(|(_, ty)| ty.clone()).collect(),
            outputs: vec![partials.clone()],
            input_strides: step.input_strides(),
            output_strides: vec![contiguous_strides(&partials.shape)],
        }
    });
    launch(
        &format!("lce_partials_{kind}{}_{}", h.dtype, target.dtype),
        &[
            inputs[0],
            inputs[1],
            inputs[2],
            output.cast_const(),
            maxima,
            sums,
        ],
        &[p],
        Grid::Groups([tiles, b.div_ceil(tm), 1]),
        keep.clone(),
        step.label,
    )?;
    crate::profiler::next_launch("linear_cross_entropy (combine)", || {
        crate::profiler::IoTypes {
            inputs: vec![partials.clone()],
            outputs: vec![step.output.1.clone()],
            input_strides: vec![contiguous_strides(&partials.shape)],
            output_strides: step.output_strides(),
        }
    });
    launch(
        "lce_combine",
        &[maxima, sums, output.cast_const()],
        &[dims_arg([b, tiles])],
        Grid::Groups([b.div_ceil(8), 1, 1]),
        keep,
        step.label,
    )
}
