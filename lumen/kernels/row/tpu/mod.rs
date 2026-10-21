use crate::codegen::tpu::row::ROW_BYTES;
use crate::kernels::dot_general::tpu::block;

pub(crate) fn block_rows(rows: usize, n: usize, unit: usize) -> Result<usize, String> {
    if rows < unit {
        return Ok(rows);
    }
    let cap = (ROW_BYTES / (n * 4)).max(unit);
    block(rows, unit, cap).ok_or_else(|| "its rows have no block of whole tiles".into())
}
