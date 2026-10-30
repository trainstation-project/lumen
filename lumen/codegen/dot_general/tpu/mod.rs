//! TPU `dot_general` (with a fused elementwise epilogue): block sizes, and launching the Pallas program.

use crate::DType;
use crate::codegen::dot_general::GemmCodegen;
pub(crate) mod codegen;
pub(crate) mod row;

use crate::codegen::utils::tpu::Tpu;
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::kernels::tpu::{Launch, launch};

/// The largest block dividing `n` that is a multiple of `unit` and at most `cap`; `n` itself when it fits.
pub(crate) fn block(n: usize, unit: usize, cap: usize) -> Option<usize> {
    if n <= cap {
        return Some(n);
    }

    (1..=cap / unit)
        .rev()
        .map(|i| i * unit)
        .find(|&b| n.is_multiple_of(b))
}

/// Starting `(m, n, k)` block caps for `dtype` operands with contraction length `k`.
pub(crate) fn caps(dtype: DType, k: usize) -> (usize, usize, usize) {
    match dtype {
        DType::F32 => (2048, 1024, 1024),
        _ if k <= 4096 => (2048, 1024, k),
        _ => (2048, 2048, 2048),
    }
}

/// Launches the dot (or dot-plus-epilogue fusion) of a plan step on TPU device `index`.
pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let out = &step.output.1;

    if out.numel() == 0 {
        return Ok(());
    }

    let (kernel, spec) = match &step.primitive {
        Primitive::Fusion { body, .. } => {
            let dot = body
                .nodes()
                .iter()
                .find(|n| matches!(n.primitive, Primitive::DotGeneral { .. }))
                .expect("a dot");

            Tpu::gemm(body, dot, &[], step)?
        }
        p => codegen::spec(
            step,
            p,
            &step.inputs[0].1,
            &step.inputs[1].1,
            out.dtype,
            None,
        )?,
    };

    let n = step.inputs.len();

    let rows: Vec<*const u8> = inputs[..n]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(inputs[n..].iter().copied())
        .collect();

    let launched = Launch {
        module: "lumen.codegen.dot_general.tpu",
        kernel,
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
        sparse: None,
    };

    launch(index, launched)
}
