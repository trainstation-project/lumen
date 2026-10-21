use crate::DType;
use crate::device::tpu::layout::Layout;
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::kernels::dot_general::tpu::block;
use crate::kernels::tpu::{Launch, launch};

const LANES: usize = 128;

const BLOCK_BYTES: usize = 8 << 20;

const WINDOW: usize = 8192;

const ROWS: usize = 64;

pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let (table, ids) = (&step.inputs[0].1, &step.inputs[1].1);
    let failed = |what: &str| Err(format!("{what} does not run on TPU yet: {table} at {ids}"));
    let Primitive::Gather { axis: 0 } = step.primitive else {
        return failed("a gather along other than axis 0");
    };
    if table.shape.len() != 2 || ids.shape.is_empty() || ids.dtype != DType::I32 {
        return failed("a gather of other than a [vocab, width] table's rows at int32 indices");
    }
    if !matches!(table.dtype, DType::F32 | DType::BF16 | DType::I32) {
        return failed("a table of other than float32, bfloat16 or int32");
    }
    let (vocab, width) = (table.shape[0], table.shape[1]);
    let tile = Layout::new(table.dtype, &[1, 1])?.rows;
    let rank = ids.shape.len();
    let s = ids.shape[rank - 1];
    let r = if rank >= 2 { ids.shape[rank - 2] } else { 1 };
    let indices: Vec<usize> = match rank {
        1 => vec![s],
        _ => vec![ids.shape[..rank - 2].iter().product(), r, s],
    };
    let rw = match r {
        _ if rank == 1 => 1,
        r if r.is_multiple_of(8) => 8,
        r if r < 8 => r,
        _ => return failed("indices of rows not a multiple of 8 (or fewer)"),
    };
    let cw = match s {
        s if rw * s <= WINDOW => s,
        s if s.is_multiple_of(LANES) => LANES,
        _ => return failed("a long last dimension of indices not a multiple of 128"),
    };
    let whole = |b: &usize| s.is_multiple_of(*b) && cw.is_multiple_of(*b);
    let bm = match (1..=ROWS / tile).rev().map(|k| k * tile).find(whole) {
        Some(bm) => bm,
        None if s < tile => s,
        None => {
            return failed("a last dimension of indices not a multiple of the table's tile rows");
        }
    };
    let cap = (BLOCK_BYTES / (bm * tile * 4)).max(LANES);
    let bn = block(width, LANES, cap)
        .filter(|&b| b == width || b.is_multiple_of(LANES))
        .ok_or_else(|| format!("gather on TPU: {width} columns have no block of whole tiles"))?;
    let spec = format!(
        "\"vocab\": {vocab}, \"width\": {width}, \"dtype\": \"{}\", \"tile\": {tile}, \"indices\": {indices:?}, \"bm\": {bm}, \"bn\": {bn}, \"rw\": {rw}, \"cw\": {cw}",
        table.dtype.full_name()
    );
    let rows = [inputs[0], inputs[1], output.cast_const()];
    let launched = Launch {
        module: "lumen.kernels.gather.tpu",
        kernel: format!("gather_{}_{bm}x{bn}", table.dtype),
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
    };
    launch(index, launched).map_err(|e| format!("{}: {e}", step.label))
}
