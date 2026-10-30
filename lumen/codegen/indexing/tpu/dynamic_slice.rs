//! Plans the Pallas `dynamic_update_slice` kernel on TPU: block sizes, the windows each update
//! touches, and whether the touched region can stay resident in VMEM.
use crate::DType;
use crate::codegen::dot_general::tpu::block;
use crate::device::tpu::layout::Layout;
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::kernels::tpu::{Launch, launch};
use crate::tensor::scalar::Scalar;

/// Lane width of a TPU tile.
const LANES: usize = 128;
/// Element budget for one staged `bm` x `bn` piece of an aligned update.
const STAGED: usize = 1 << 20;
/// Element budget for one streamed `gm` x `width` window of an unaligned update.
const STREAMED: usize = 1 << 19;
/// Byte budget for one row block of the operand-to-output copy.
const COPY_BYTES: usize = 2 << 20;
/// Updates kept in flight when each update fits in one block.
const DEPTH: usize = 8;

/// Most bytes of the output region all updates touch that may be kept in VMEM.
const RESIDENT: usize = 48 << 20;

/// Launches the `dynamic_update_slice` kernel for `step`: one update at runtime int32 starts, or
/// a fusion of several updates at constant starts (clamped into range here).
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

    // A fusion carries its updates' starts as constants; read and clamp them at compile time.
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

    // Sizes padded to whole TPU tiles (`tile` rows by `LANES` columns).
    let tile = layout.rows;
    let (rows, cols) = (operand.shape[rank - 2], operand.shape[rank - 1]);
    let (ur, uc) = (update.shape[rank - 2], update.shape[rank - 1]);
    let (rp, cp) = (rows.div_ceil(tile) * tile, cols.div_ceil(LANES) * LANES);
    let (urp, ucp) = (ur.div_ceil(tile) * tile, uc.div_ceil(LANES) * LANES);
    let bn = block(ucp, LANES, (STAGED / tile).max(LANES)).unwrap_or(LANES);
    let bm = block(urp, tile, (STAGED / bn / tile * tile).max(tile)).unwrap_or(tile);
    let single = update.shape[..rank - 2].iter().product::<usize>() == 1 && urp == bm && ucp == bn;

    // An unaligned window spans one extra lane block unless the update covers every column.
    let width = match uc == cols {
        true => ucp,
        false => (ucp + LANES).min(cp),
    };

    let gm = match single {
        true => urp,
        false => block(urp, tile, (STREAMED / width / tile * tile).max(tile)).unwrap_or(tile),
    };

    // Whether the update's last two dims end on tile boundaries (or at the operand's edges).
    let whole = (ur.is_multiple_of(tile) || ur == rows) && (uc.is_multiple_of(LANES) || uc == cols);

    // Tile-aligned window each constant-start update writes, per dim as (start, end).
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

    // Overlapping windows must be applied in order rather than overlapped in flight.
    let ordered = windows
        .iter()
        .enumerate()
        .any(|(i, p)| windows[i + 1..].iter().any(|q| overlap(p, q)));

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

    let lone_update = update.shape[..rank - 2].iter().product::<usize>() == 1;
    let direct = fixed.is_some() && aligned && lone_update;

    // Keep the box covering all windows in VMEM when every update lands in the same leading slab.
    let resident = match fixed.is_some() && (direct || ordered && small) && one_slab {
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

            let covered =
                direct && !ordered && windows.len() * urp * ucp == (bottom - top) * (right - left);

            (bytes <= RESIDENT).then_some((top, bottom, left, right, slab, covered))
        }
        false => None,
    };

    let copy = inputs[0] != output.cast_const();

    let rows_block = |length: usize| {
        let cap = (COPY_BYTES / (cp * layout.stored.size_of()) / tile * tile).max(tile);
        block(length, tile, cap).unwrap_or(tile)
    };

    let lone = operand.shape[..rank - 2].iter().product::<usize>() == 1;

    // When the resident updates exactly cover full-width rows, only the rows outside them need copying.
    let ranges = match resident {
        Some((top, bottom, 0, right, _, true)) if right == cp && lone => {
            vec![(0, top), (bottom, rp - bottom)]
        }
        _ => vec![(0, rp)],
    };

    let copied_rows: Vec<String> = ranges
        .into_iter()
        .filter(|&(_, length)| length > 0)
        .map(|(first, length)| format!("[{first}, {length}, {}]", rows_block(length)))
        .collect();

    let resident = resident.map_or(
        "null".to_string(),
        |(top, bottom, left, right, slab, covered)| {
            format!("[{top}, {bottom}, {left}, {right}, {slab}, {covered}, {direct}]")
        },
    );

    let spec = format!(
        "\"shape\": {:?}, \"update\": {:?}, \"dtype\": \"{}\", \"tile\": {tile}, \"bm\": {bm}, \"bn\": {bn}, \"gm\": {gm}, \"width\": {width}, \"copy\": {copy}, \"copied_rows\": [{}], \"count\": {count}, \"static\": {}, \"depth\": {DEPTH}, \"ordered\": {ordered}, \"resident\": {resident}",
        operand.shape,
        update.shape,
        operand.dtype.full_name(),
        copied_rows.join(", "),
        fixed.is_some(),
    );

    let n = step.inputs.len();

    let rows_of: Vec<*const u8> = inputs[..n]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .collect();

    let launched = Launch {
        module: "lumen.codegen.indexing.tpu.dynamic_slice",
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
