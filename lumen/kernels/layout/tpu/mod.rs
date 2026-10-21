use crate::device::tpu::layout::Layout;
use crate::graph_compiler::plan::Step;
use crate::kernels::dot_general::tpu::block;
use crate::kernels::tpu::{Launch, launch};

pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let (from, to) = (&step.inputs[0].1, &step.output.1);
    let (f, t) = (&from.shape, &to.shape);
    let failed = || {
        Err(format!(
            "reshape on TPU: {from} to {to} moves elements between tiles other than by splitting or merging its last dimension"
        ))
    };
    let split = t.len() == f.len() + 1;
    let (flat, tiled) = match split {
        true => (f, t),
        false => (t, f),
    };
    if tiled.len() != flat.len() + 1 || tiled.len() < 3 {
        return failed();
    }
    let r = flat.len();
    let (a, w) = (tiled[r - 1], tiled[r]);
    if flat[..r - 1] != tiled[..r - 1] || flat[r - 1] != a * w {
        return failed();
    }
    let m = flat[r - 2];
    let lead: usize = flat[..r - 2].iter().product();
    let rows = Layout::new(from.dtype, &[1, 1])?.rows;
    let bm = block(m, rows, 256)
        .ok_or_else(|| format!("reshape on TPU: {m} rows have no block of whole tiles"))?;
    let spec = format!(
        "\"split\": {split}, \"lead\": {lead}, \"m\": {m}, \"a\": {a}, \"w\": {w}, \"bm\": {bm}, \"dtype\": \"{}\"",
        from.dtype.full_name()
    );
    let rows = [inputs[0], output.cast_const()];
    let launched = Launch {
        module: "lumen.kernels.layout.tpu",
        kernel: format!("reshape_{}_{a}x{w}", if split { "split" } else { "merge" }),
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
    };
    launch(index, launched)
}
