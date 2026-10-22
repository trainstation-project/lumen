use crate::DType;
use crate::device::tpu::layout::Layout;
use crate::graph_compiler::plan::Step;
use crate::kernels::dot_general::tpu::block;
use crate::kernels::tpu::{Launch, launch};

const LANES: usize = 128;

const BLOCK: usize = 1 << 19;

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
    if rank < 2 {
        return failed("a dynamic_update_slice of fewer than 2 dimensions");
    }
    if step.inputs[2..].iter().any(|(_, t)| t.dtype != DType::I32) {
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
    let (ur, uc) = (update.shape[rank - 2], update.shape[rank - 1]);
    let (urp, ucp) = (ur.div_ceil(tile) * tile, uc.div_ceil(LANES) * LANES);
    let bn = block(ucp, LANES, 1024).unwrap_or(LANES);
    let bm = block(urp, tile, (BLOCK / bn / tile * tile).max(tile)).unwrap_or(tile);
    let copy = inputs[0] != output.cast_const();
    let spec = format!(
        "\"shape\": {:?}, \"update\": {:?}, \"dtype\": \"{}\", \"tile\": {tile}, \"bm\": {bm}, \"bn\": {bn}, \"copy\": {copy}",
        operand.shape,
        update.shape,
        operand.dtype.full_name()
    );
    let n = step.inputs.len();
    let rows: Vec<*const u8> = inputs[..n]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .collect();
    let launched = Launch {
        module: "lumen.kernels.dynamic_slice.tpu",
        kernel: format!("dynamic_update_slice_{}_{bm}x{bn}", operand.dtype),
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
        sparse: None,
    };
    launch(index, launched).map_err(|e| format!("{}: {e}", step.label))
}
