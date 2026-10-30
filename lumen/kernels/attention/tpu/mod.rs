//! Launches the Pallas flash attention forward and backward kernels on TPU.

#[cfg(lumen_tpu_linked)]
use crate::graph::Primitive;
#[cfg(lumen_tpu_linked)]
use crate::graph_compiler::plan::Step;
#[cfg(lumen_tpu_linked)]
use crate::kernels::tpu::{Launch, launch};
#[cfg(lumen_tpu_linked)]
use crate::tensor::contiguous_strides;

/// Launches the Python attention kernel `kind` (`forward` or `backward`) with buffer pointers `rows`.
#[cfg(lumen_tpu_linked)]
fn run(index: usize, kind: &str, spec: String, rows: &[*const u8]) -> Result<(), String> {
    let module = match kind {
        "forward" => "lumen.codegen.attention.tpu.forward",
        _ => "lumen.codegen.attention.tpu.backward",
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
    // The kernels read each operand's heads and head dim (its last two dims) contiguous.
    for (strides, (_, ty)) in step.input_strides().iter().zip(&step.inputs) {
        let rank = ty.shape.len();
        let contiguous = contiguous_strides(&ty.shape);
        if rank > 0 && strides[rank.saturating_sub(2)..] != contiguous[rank.saturating_sub(2)..] {
            return Err(format!(
                "{}: flash attention on TPU reads its operands' last two dims (heads, head dim) \
                 contiguous; one has strides {strides:?}",
                step.label
            ));
        }
    }
    let n = step.inputs.len();
    let rows: Vec<*const u8> = inputs[..n]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(inputs[n..].iter().copied())
        .collect();
    let (kind, spec) = crate::codegen::attention::tpu::kernel(name, body)?;
    run(index, kind, spec, &rows)
}
