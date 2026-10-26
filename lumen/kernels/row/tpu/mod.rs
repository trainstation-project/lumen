//! Block sizes for the TPU row-fusion kernels.
use crate::codegen::tpu::row::ROW_BYTES;
use crate::kernels::dot_general::tpu::block;

/// Picks how many of `rows` rows of `n` elements a step holds: a multiple of `unit` dividing `rows` within the
/// ROW_BYTES budget (counted at 4 bytes an element), or all rows when there are fewer than `unit`.
pub(crate) fn block_rows(rows: usize, n: usize, unit: usize) -> Result<usize, String> {
    if rows < unit {
        return Ok(rows);
    }
    let cap = (ROW_BYTES / (n * 4)).max(unit);
    block(rows, unit, cap).ok_or_else(|| "its rows have no block of whole tiles".into())
}
