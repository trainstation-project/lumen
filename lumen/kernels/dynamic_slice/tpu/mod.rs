use crate::DType;
use crate::device::tpu::layout::Layout;
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::kernels::dot_general::tpu::block;
use crate::kernels::tpu::{Launch, launch};
use crate::tensor::scalar::Scalar;

const LANES: usize = 128;
const STAGED: usize = 1 << 20;
const STREAMED: usize = 1 << 18;
const COPY_BYTES: usize = 2 << 20;
const DEPTH: usize = 8;

const RESIDENT: usize = 48 << 20;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let (operand, update) = (&step.inputs[0].1, &step.inputs[1].1);
    let failed = |what: &str| {
        Err(format!(
            "{what} does not run on TPU yet: {update} into {operand}"
        ))
    };
    let rank = operand.shape.len();
    let fixed: Option<Vec<i32>> = match &step.primitive {
        Primitive::Fusion { body, .. } => {
            let nodes = body.nodes();
            let value = |v| match nodes.iter().find(|n| n.output == v).map(|n| &n.primitive) {
                Some(Primitive::Full {
                    fill_value: Scalar::Int(i),
                    ..
                }) => *i,
                _ => unreachable!("a constant start"),
            };
            let starts = nodes
                .iter()
                .filter(|n| matches!(n.primitive, Primitive::DynamicUpdateSlice))
                .flat_map(|n| {
                    n.inputs[2..]
                        .iter()
                        .enumerate()
                        .map(|(d, &v)| (d, value(v)))
                })
                .map(|(d, s)| s.clamp(0, (operand.shape[d] - update.shape[d]) as i64) as i32)
                .collect();
            Some(starts)
        }
        _ => None,
    };
    let count = match fixed {
        Some(_) => step.inputs.len() - 1,
        None => 1,
    };
    if rank < 2 {
        return failed("a dynamic_update_slice of fewer than 2 dimensions");
    }
    if fixed.is_none() && step.inputs[2..].iter().any(|(_, t)| t.dtype != DType::I32) {
        return failed("a dynamic_update_slice at other than int32 indices");
    }
    let layout = Layout::new(operand.dtype, &operand.shape)?;
    if layout.stored.bits() < 16 {
        return failed("a dynamic_update_slice of 8-bit elements");
    }
    if update.numel() == 0 {
        return Ok(());
    }
    let tile = layout.rows;
    let (rows, cols) = (operand.shape[rank - 2], operand.shape[rank - 1]);
    let (ur, uc) = (update.shape[rank - 2], update.shape[rank - 1]);
    let (rp, cp) = (rows.div_ceil(tile) * tile, cols.div_ceil(LANES) * LANES);
    let (urp, ucp) = (ur.div_ceil(tile) * tile, uc.div_ceil(LANES) * LANES);
    let (staged, streamed) = (env("DUS_STAGED", STAGED), env("DUS_STREAMED", STREAMED));
    let bn = block(ucp, LANES, (staged / tile).max(LANES)).unwrap_or(LANES);
    let bm = block(urp, tile, (staged / bn / tile * tile).max(tile)).unwrap_or(tile);
    let single = update.shape[..rank - 2].iter().product::<usize>() == 1 && urp == bm && ucp == bn;
    let width = match uc == cols {
        true => ucp,
        false => (ucp + LANES).min(cp),
    };
    let gm = match single {
        true => urp,
        false => block(urp, tile, (streamed / width / tile * tile).max(tile)).unwrap_or(tile),
    };
    let whole = (ur.is_multiple_of(tile) || ur == rows) && (uc.is_multiple_of(LANES) || uc == cols);
    let windows: Vec<Vec<(usize, usize)>> = fixed
        .iter()
        .flat_map(|starts| starts.chunks(rank))
        .map(|at| {
            let at: Vec<usize> = at.iter().map(|&s| s as usize).collect();
            let (r0, c0) = (at[rank - 2], at[rank - 1]);
            let mut ranges: Vec<(usize, usize)> = (0..rank - 2)
                .map(|d| (at[d], at[d] + update.shape[d]))
                .collect();
            match whole && r0.is_multiple_of(tile) && c0.is_multiple_of(LANES) {
                true => ranges.extend([(r0, r0 + urp), (c0, c0 + ucp)]),
                false => {
                    let (a, b) = (r0 / tile * tile, (c0 / LANES * LANES).min(cp - width));
                    ranges.extend([(a, a + urp + tile), (b, b + width)]);
                }
            }
            ranges
        })
        .collect();
    let overlap = |p: &[(usize, usize)], q: &[(usize, usize)]| {
        p.iter().zip(q).all(|(x, y)| x.0 < y.1 && y.0 < x.1)
    };
    let ordered = windows
        .iter()
        .enumerate()
        .any(|(i, p)| windows[i + 1..].iter().any(|q| overlap(p, q)));
    let depth = env("DUS_DEPTH", DEPTH).max(2);
    let small = update.shape[..rank - 2].iter().product::<usize>() == 1 && gm == urp;
    let one_slab = windows
        .iter()
        .all(|w| w[..rank - 2] == windows[0][..rank - 2]);
    let starts_of: Vec<&[i32]> = fixed.iter().flat_map(|s| s.chunks(rank)).collect();
    let aligned = whole
        && starts_of.iter().all(|at| {
            (at[rank - 2] as usize).is_multiple_of(tile)
                && (at[rank - 1] as usize).is_multiple_of(LANES)
        });
    let direct = aligned && !ordered;
    let resident = match (ordered || direct) && small && one_slab {
        true => {
            let (top, bottom) = windows.iter().fold((usize::MAX, 0), |(t, b), w| {
                (t.min(w[rank - 2].0), b.max(w[rank - 2].1.min(rp)))
            });
            let (left, right) = windows.iter().fold((usize::MAX, 0), |(l, r), w| {
                (l.min(w[rank - 1].0), r.max(w[rank - 1].1))
            });
            let slab: usize = (0..rank - 2)
                .map(|d| windows[0][d].0 * operand.shape[d + 1..rank - 2].iter().product::<usize>())
                .sum();
            let bytes = (bottom - top) * (right - left) * layout.stored.size_of();
            let covered = direct && windows.len() * urp * ucp == (bottom - top) * (right - left);
            (bytes <= RESIDENT)
                .then(|| format!("[{top}, {bottom}, {left}, {right}, {slab}, {covered}, {direct}]"))
        }
        false => None,
    };
    let copy = inputs[0] != output.cast_const();
    let copy_rows = block(
        rp,
        tile,
        (COPY_BYTES / (cp * layout.stored.size_of()) / tile * tile).max(tile),
    )
    .unwrap_or(tile);
    let spec = format!(
        "\"shape\": {:?}, \"update\": {:?}, \"dtype\": \"{}\", \"tile\": {tile}, \"bm\": {bm}, \"bn\": {bn}, \"gm\": {gm}, \"width\": {width}, \"copy\": {copy}, \"copy_rows\": {copy_rows}, \"count\": {count}, \"static\": {}, \"depth\": {depth}, \"ordered\": {ordered}, \"resident\": {}",
        operand.shape,
        update.shape,
        operand.dtype.full_name(),
        fixed.is_some(),
        resident.as_deref().unwrap_or("null")
    );
    let n = step.inputs.len();
    let rows_of: Vec<*const u8> = inputs[..n]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .collect();
    let launched = Launch {
        module: "lumen.kernels.dynamic_slice.tpu",
        kernel: format!(
            "dynamic_update_slice_{count}x{}_{bm}x{bn}_{gm}",
            operand.dtype
        ),
        spec,
        rows: &rows_of,
        scalars: fixed.as_deref().unwrap_or(&[]),
        host: &[],
        sparse: None,
    };
    launch(index, launched).map_err(|e| format!("{}: {e}", step.label))
}
