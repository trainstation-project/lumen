//! TPU `dot_general` (with a fused elementwise epilogue): block sizes, and launching the Pallas program.

use crate::DType;
use crate::codegen::dot_general::GemmCodegen;
pub(crate) mod codegen;
pub(crate) mod row;

use crate::codegen::tpu::{Kernel, Rows};
use crate::codegen::utils::tpu::Tpu;
use crate::graph::Primitive;
use crate::plan::StepBase;

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

/// The kernel of a plan step's dot (or dot-plus-epilogue fusion); `None` when its output is empty.
pub(crate) fn kernel(step: &StepBase) -> Result<Option<Kernel>, String> {
    let out = &step.output.1;

    if out.numel() == 0 {
        return Ok(None);
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

    Ok(Some(Kernel::new(
        "lumen.codegen.dot_general.tpu.gemm_kernel",
        kernel,
        spec,
        Rows::Split,
    )))
}
