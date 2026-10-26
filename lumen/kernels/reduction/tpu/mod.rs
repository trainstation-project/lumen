//! Block sizes for the TPU Pallas reduction kernels.
use crate::DType;
use crate::codegen::tpu::reduction::Layout;
use crate::device::tpu::layout::Layout as Tiles;
use crate::kernels::dot_general::tpu::block;

/// Target bytes of one operand block in VMEM.
const BLOCK_BYTES: usize = 1 << 21;

/// Returns the most sublane rows a TPU tile needs among `dtypes`, at least 8.
fn tile_rows(dtypes: &[DType]) -> Result<usize, String> {
    dtypes
        .iter()
        .map(|&d| Tiles::new(d, &[1, 1]).map(|l| l.rows))
        .try_fold(8, |most, rows| rows.map(|r| most.max(r)))
}

/// Picks block sizes for a reduction `layout` that cover whole tiles of the `ins` and `outs` dtypes:
/// `[bh, bm, bn]` for rows, `[bm, bn]` for columns and 2-D totals, `[b]` for flat totals.
pub(crate) fn blocks(layout: &Layout, ins: &[DType], outs: &[DType]) -> Result<Vec<usize>, String> {
    let failed = |what: &str| format!("{what} has no block of whole tiles");
    let rows_in = tile_rows(ins)?;
    // lane-axis blocks are multiples of 128 dividing `n`, or all of `n` when none fits
    let lanes = |n: usize, cap: usize| block(n, 128, cap).unwrap_or(n);
    match *layout {
        Layout::Rows { rank, h, m, n, .. } => {
            let bh = match rank {
                2 => 1,
                _ => {
                    let unit = tile_rows(outs)?;
                    block(h, unit, unit).ok_or_else(|| failed("its rows' rows"))?
                }
            };
            let bn = lanes(n, 512);
            let unit_m = (if rank == 2 { 1024 } else { 128 }).max(rows_in);
            let cap = 1024.min((BLOCK_BYTES / (bh * bn)).max(unit_m));
            let bm = block(m, unit_m, cap).ok_or_else(|| failed("its rows"))?;
            Ok(vec![bh, bm, bn])
        }
        Layout::Columns { m, n, .. } => {
            let bn = block(n, 1024, 2048).unwrap_or(n);
            // budget 4 bytes per element: the operand is read as float32 or wider in the accumulator
            let cap = (BLOCK_BYTES / (4 * bn)).max(rows_in);
            let bm = block(m, rows_in, cap).ok_or_else(|| failed("its rows"))?;
            Ok(vec![bm, bn])
        }
        Layout::Total { n, flat: true, .. } => Ok(vec![block(n, 1024, 64 * 1024).unwrap_or(n)]),
        Layout::Total { m, n, .. } => {
            let bn = lanes(n, 2048);
            let cap = (BLOCK_BYTES / (4 * bn)).max(rows_in);
            let bm = block(m, rows_in, cap).ok_or_else(|| failed("its rows"))?;
            Ok(vec![bm, bn])
        }
    }
}
