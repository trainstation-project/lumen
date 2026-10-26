//! Launches the Pallas flash attention forward and backward kernels on TPU.

#[cfg(lumen_tpu_linked)]
use crate::graph::Primitive;
#[cfg(lumen_tpu_linked)]
use crate::graph_compiler::plan::Step;
#[cfg(lumen_tpu_linked)]
use crate::kernels::tpu::{Launch, launch};

/// TPU lane width; attention block sizes are multiples of it.
const LANES: usize = 128;

/// Largest query block.
const QUERIES: usize = 1024;

/// Largest key block.
const KEYS: usize = 4096;

/// Largest multiple of `LANES` up to `cap` that divides `n`, or `n` itself when none does.
fn block(n: usize, cap: usize) -> usize {
    (1..=cap / LANES)
        .rev()
        .map(|k| k * LANES)
        .find(|&b| n.is_multiple_of(b))
        .unwrap_or(n)
}

/// Query and key block sizes for sequence lengths `sq` and `sk`.
pub(crate) fn blocks(sq: usize, sk: usize) -> (usize, usize) {
    (block(sq, QUERIES), block(sk, KEYS))
}

/// Launches the Python attention kernel `kind` (`forward` or `backward`) with buffer pointers `rows`.
#[cfg(lumen_tpu_linked)]
fn run(index: usize, kind: &str, spec: String, rows: &[*const u8]) -> Result<(), String> {
    let module = match kind {
        "forward" => "lumen.kernels.attention.tpu.forward",
        _ => "lumen.kernels.attention.tpu.backward",
    };
    let launched = Launch {
        module,
        kernel: format!("flash_attention_{kind}"),
        spec,
        rows,
        scalars: &[],
        host: &[],
        sparse: None,
    };
    launch(index, launched)
}

/// Encodes an attention fusion step, binding its inputs, then its output, then any extra outputs.
#[cfg(lumen_tpu_linked)]
pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let Primitive::Fusion { name, body, .. } = &step.primitive else {
        unreachable!("an attention fusion")
    };
    let n = step.inputs.len();
    let rows: Vec<*const u8> = inputs[..n]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(inputs[n..].iter().copied())
        .collect();
    let (kind, spec) = match name.as_str() {
        "flash_attention" => crate::codegen::tpu::attention::forward(body)?,
        _ => crate::codegen::tpu::attention::backward(body)?,
    };
    run(index, kind, spec, &rows)
}
