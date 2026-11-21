//! TPU `dot_general` (with a fused elementwise epilogue): block sizes, and launching the Pallas program.

#[cfg(lumen_tpu_linked)]
use crate::DType;
#[cfg(lumen_tpu_linked)]
use crate::codegen::dot_general::GemmCodegen;
#[cfg(lumen_tpu_linked)]
pub(crate) mod codegen;
#[cfg(lumen_tpu_linked)]
pub(crate) mod row;

use crate::codegen::devices::tpu::Tpu;
#[cfg(lumen_tpu_linked)]
use crate::codegen::devices::tpu::{Kernel, Rows};
#[cfg(lumen_tpu_linked)]
use crate::graph::Primitive;
#[cfg(lumen_tpu_linked)]
use crate::plan::StepBase;

/// The largest block dividing `n` that is a multiple of `unit` and at most `cap`; `n` itself when it fits.
#[cfg(lumen_tpu_linked)]
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
#[cfg(lumen_tpu_linked)]
pub(crate) fn caps(dtype: DType, k: usize) -> (usize, usize, usize) {
    match dtype {
        DType::F32 => (2048, 1024, 1024),
        _ if k <= 4096 => (2048, 1024, k),
        _ => (2048, 2048, 2048),
    }
}

/// The kernel of a plan step's dot (or dot-plus-epilogue fusion); `None` when its output is empty.
#[cfg(lumen_tpu_linked)]
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

/// TPU merges without `fuse`, late (once the graph is simplified), packing whole vector lanes: everything read in place,
/// no views, `LANES` lanes, 1 column.
impl crate::codegen::dot_general::MergesDots for Tpu {
    const WITH_FUSE: bool = false;
    const LATE: bool = true;

    /// Everything read in place; no views; `LANES` lanes; 1 column.
    fn target() -> crate::graph_compiler::passes::merge_dots::Target {
        crate::graph_compiler::passes::merge_dots::Target {
            reads: |_, _| true,
            views: false,
            lanes: crate::codegen::devices::tpu::LANES,
            columns: 1,
        }
    }
}

/// TPU runs batched inner products as elementwise and reduction fusions.
impl crate::codegen::dot_general::InnerProducts for Tpu {}

/// TPU's matmul kernel takes dots with their operands' reshapes folded in.
impl crate::codegen::dot_general::FoldsDotReshapes for Tpu {}

/// TPU's matmul kernel reads its batch dims leading, the lhs as batch, free dims then one contracting one (or batch,
/// contracting, free), the rhs as batch, then one contracting and one free dim in either order.
impl crate::codegen::dot_general::DotLayouts for Tpu {
    /// Whether the TPU matmul kernel reads operand `k` of `dot` over `operands` as laid out.
    fn readable(
        dot: &crate::graph::Primitive,
        operands: [&crate::graph::TensorType; 2],
        k: usize,
    ) -> bool {
        let crate::graph::Primitive::DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            rhs_batch,
            ..
        } = dot
        else {
            unreachable!("a dot_general")
        };

        let leads = |batch: &[usize]| batch.iter().enumerate().all(|(i, &d)| d == i);
        let nb = lhs_batch.len();
        let rank = operands[k].shape.len();

        match k {
            0 => {
                let last = *lhs_contracting == [rank - 1];
                let transposed = rank == nb + 2 && *lhs_contracting == [nb];
                leads(lhs_batch) && rank >= nb + 2 && (last || transposed)
            }
            _ => leads(rhs_batch) && rank == nb + 2 && rhs_contracting.len() == 1,
        }
    }
}

/// TPU fuses matmuls with their slices, stacked lhs and epilogues, in whole vector lanes, a producer writing a
/// concatenation's slot as `fusion::tpu::slot_writes` accepts.
impl crate::codegen::dot_general::FusesDots for Tpu {
    const LANES: usize = crate::codegen::devices::tpu::LANES;

    /// `fusion::tpu::slot_writes`.
    fn slot_writes(
        graph: &crate::graph::Graph,
        concat: &crate::graph::Node,
        k: usize,
        producer: &crate::graph::Node,
    ) -> bool {
        crate::codegen::fusion::tpu::slot_writes(graph, concat, k, producer)
    }
}
