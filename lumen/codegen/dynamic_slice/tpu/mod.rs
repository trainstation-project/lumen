//! Plans the Pallas `dynamic_update_slice` kernel on TPU: block sizes, the windows each update
//! touches, and whether the touched region can stay resident in VMEM.
use crate::DType;
use crate::codegen::dot_general::tpu::block;
use crate::codegen::dynamic_slice::DynamicSliceCodegen;
use crate::codegen::utils::tpu::Tpu;
use crate::device::tpu::layout::Layout;
use crate::graph::{Graph, Primitive, TensorType};
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

/// The constant starts of the updates in `updates` fusion `body`, clamped into range, each update's after
/// the one before.
fn starts(body: &Graph, operand: &TensorType, update: &TensorType) -> Vec<i32> {
    let nodes = body.nodes();

    let value = |v| match nodes.iter().find(|n| n.output == v).map(|n| &n.primitive) {
        Some(Primitive::Full {
            fill_value: Scalar::Int(i),
            ..
        }) => *i,
        _ => unreachable!("a constant start"),
    };

    nodes
        .iter()
        .filter(|n| matches!(n.primitive, Primitive::DynamicUpdateSlice))
        .flat_map(|n| {
            n.inputs[2..]
                .iter()
                .enumerate()
                .map(|(d, &v)| (d, value(v)))
        })
        .map(|(d, s)| s.clamp(0, (operand.shape[d] - update.shape[d]) as i64) as i32)
        .collect()
}

/// A TPU `dynamic_update_slice` program: its kernel name, its spec (but for whether the operand is copied,
/// which the launch knows) and the constant starts it reads as scalars.
pub(crate) struct Program {
    pub(crate) kernel: String,
    pub(crate) spec: String,
    pub(crate) scalars: Vec<i32>,
}

/// The program writing `count` updates of `update`'s type into `operand`: at the constant starts `fixed`
/// (each update's after the one before), or at runtime int32 starts when `None`.
fn program(
    operand: &TensorType,
    update: &TensorType,
    fixed: Option<Vec<i32>>,
    count: usize,
) -> Result<Program, String> {
    let failed = |what: &str| {
        Err(format!(
            "{what} does not run on TPU yet: {update} into {operand}"
        ))
    };

    let rank = operand.shape.len();

    if rank < 2 {
        return failed("a dynamic_update_slice of fewer than 2 dimensions");
    }

    let layout = Layout::new(operand.dtype, &operand.shape)?;

    if layout.stored.bits() < 16 {
        return failed("a dynamic_update_slice of 8-bit elements");
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
        "\"shape\": {:?}, \"update\": {:?}, \"dtype\": \"{}\", \"tile\": {tile}, \"bm\": {bm}, \"bn\": {bn}, \"gm\": {gm}, \"width\": {width}, \"copied_rows\": [{}], \"count\": {count}, \"static\": {}, \"depth\": {DEPTH}, \"ordered\": {ordered}, \"resident\": {resident}",
        operand.shape,
        update.shape,
        operand.dtype.full_name(),
        copied_rows.join(", "),
        fixed.is_some(),
    );

    Ok(Program {
        kernel: format!(
            "dynamic_update_slice_{count}x{}_{bm}x{bn}_{gm}",
            operand.dtype
        ),
        spec,
        scalars: fixed.unwrap_or_default(),
    })
}

/// TPU dynamic slices: the fixed `dynamic_update_slice` Pallas program, specialized by a spec of the operand,
/// the updates and their blocks.
impl DynamicSliceCodegen for Tpu {
    type Code = Program;
    type Options = ();

    /// The program writing the `updates` fusion `body`'s updates at their constant starts.
    fn updates(body: &Graph, _: &()) -> Result<Program, String> {
        let ty = |k: usize| body.type_of(body.inputs()[k]);
        let (operand, update) = (ty(0), ty(1));
        let fixed = starts(body, operand, update);

        program(operand, update, Some(fixed), body.inputs().len() - 1)
    }

    /// The program writing `update` into `operand` at runtime int32 starts.
    fn update(
        operand: &TensorType,
        update: &TensorType,
        _: &[bool],
        _: &(),
    ) -> Result<Program, String> {
        program(operand, update, None, 1)
    }
}

/// Launches the `dynamic_update_slice` kernel for `step`: one update at runtime int32 starts, or
/// a fusion of several updates at constant starts (clamped into range here).
pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let (operand, update) = (&step.inputs[0].1, &step.inputs[1].1);
    let fixed = matches!(step.primitive, Primitive::Fusion { .. });

    let failed = |what: &str| {
        Err(format!(
            "{what} does not run on TPU yet: {update} into {operand}"
        ))
    };

    if !fixed && step.inputs[2..].iter().any(|(_, t)| t.dtype != DType::I32) {
        return failed("a dynamic_update_slice at other than int32 indices");
    }

    if update.numel() == 0 {
        return Ok(());
    }

    let p = match &step.primitive {
        Primitive::Fusion { body, .. } => Tpu::updates(body, &())?,
        _ => Tpu::update(operand, update, &[], &())?,
    };

    let copy = inputs[0] != output.cast_const();
    let spec = format!("{}, \"copy\": {copy}", p.spec);
    let n = step.inputs.len();

    let rows_of: Vec<*const u8> = inputs[..n]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .collect();

    let launched = Launch {
        module: "lumen.codegen.dynamic_slice.tpu.kernel",
        kernel: p.kernel,
        spec,
        rows: &rows_of,
        scalars: &p.scalars,
        host: &[],
        sparse: None,
    };

    launch(index, launched).map_err(|e| format!("{}: {e}", step.label))
}
