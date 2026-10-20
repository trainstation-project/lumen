use crate::DType;
use crate::device::tpu::layout::Layout;
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::ops::dot_general::tpu::block;
use crate::ops::tpu::{Launch, launch};

pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let Primitive::ReduceSum { axes, .. } = &step.primitive else {
        unreachable!("a reduce_sum")
    };
    let (x, out) = (&step.inputs[0].1, &step.output.1);
    if out.numel() == 0 {
        return Ok(());
    }
    let rank = x.shape.len();
    if rank < 2 || axes.as_slice() != [rank - 1] {
        return Err(format!(
            "reduce_sum on TPU: over the last dimension of two or more, got axes {axes:?} of {x}"
        ));
    }
    let failed = |what: &str| format!("reduce_sum on TPU: {what} has no block of whole tiles");
    let rows = |d: DType| Layout::new(d, &[1, 1]).map(|l| l.rows);
    let (m, n) = (x.shape[rank - 2], x.shape[rank - 1]);
    let (h, bh) = match rank {
        2 => (1, 1),
        _ => {
            let h = x.shape[rank - 3];
            let unit = rows(out.dtype)?;
            (
                h,
                block(h, unit, unit).ok_or_else(|| failed("its rows' rows"))?,
            )
        }
    };
    let bn = block(n, 128, 512).unwrap_or(n);
    let unit_m = (if rank == 2 { 1024 } else { 128 }).max(rows(x.dtype)?);
    let cap = (1024).min(((1 << 21) / (bh * bn)).max(unit_m));
    let bm = block(m, unit_m, cap).ok_or_else(|| failed("its rows"))?;
    let lead: usize = x.shape[..rank.saturating_sub(3)].iter().product();
    let spec = format!(
        "\"rank\": {rank}, \"lead\": {lead}, \"h\": {h}, \"m\": {m}, \"n\": {n}, \"input\": \"{}\", \
         \"output\": \"{}\", \"blocks\": [{bh}, {bm}, {bn}]",
        x.dtype.full_name(),
        out.dtype.full_name(),
    );
    let rows = [inputs[0], output.cast_const()];
    let launched = Launch {
        module: "lumen.ops.reduce.tpu",
        kernel: format!("reduce_sum_rows_{}_{}_{bm}x{bn}", x.dtype, out.dtype),
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
    };
    launch(index, launched)
}
