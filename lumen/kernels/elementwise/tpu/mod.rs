use crate::DType;
use crate::device::tpu::layout::Layout;
use crate::kernels::dot_general::tpu::block;

const BLOCK_BYTES: usize = 1 << 20;

const STREAM_BYTES: usize = 32 << 20;

pub(crate) fn blocks(shape: &[usize], dtypes: &[DType]) -> Result<Vec<usize>, String> {
    let failed = |what: &str| format!("elementwise on TPU: {what} has no block of whole tiles");
    if shape.len() < 2 {
        let n = shape.iter().product::<usize>().max(1);
        return Ok(vec![
            block(n, 1024, 64 * 1024).ok_or_else(|| failed("its length"))?,
        ]);
    }
    let rows = dtypes
        .iter()
        .map(|&d| Layout::new(d, &[1, 1]).map(|l| l.rows))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(8);
    let (m, n) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let mut bm = block(m, rows, 512).ok_or_else(|| failed("its rows"))?;
    let mut bn = block(n, 128, 2048).ok_or_else(|| failed("its columns"))?;
    let lead: usize = shape[..shape.len() - 2].iter().product();
    let width = dtypes.iter().map(|d| d.size_of()).max().unwrap_or(4);
    let budget = STREAM_BYTES / dtypes.len();
    while bm * bn * width > budget {
        let halves = |b: usize, unit: usize, of: usize| {
            (b / 2).is_multiple_of(unit) && of.is_multiple_of(b / 2)
        };
        if halves(bm, rows, m) {
            bm /= 2;
        } else if halves(bn, 128, n) {
            bn /= 2;
        } else {
            break;
        }
    }
    let tile = bm.div_ceil(rows) * rows * bn.div_ceil(128) * 128 * width;
    let bl = (1..=lead)
        .rev()
        .find(|&b| lead.is_multiple_of(b) && b * tile <= BLOCK_BYTES)
        .unwrap_or(1);
    Ok(vec![bl, bm, bn])
}

pub(crate) fn gcd(a: usize, b: usize) -> usize {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}
