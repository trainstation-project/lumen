//! Launches the Pallas flash attention forward and backward kernels on TPU.

#[cfg(lumen_tpu_linked)]
use crate::graph::Primitive;
#[cfg(lumen_tpu_linked)]
use crate::graph_compiler::plan::Step;
#[cfg(lumen_tpu_linked)]
use crate::kernels::tpu::{Launch, launch};

mod config;

pub(crate) use config::blocks;

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
