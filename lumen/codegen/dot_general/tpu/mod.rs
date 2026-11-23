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
            [&step.inputs[0].1, &step.inputs[1].1],
            [Some(0), Some(1)],
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

/// How the TPU GEMM reads an operand that is a slice of a value: the `[slabs, rows, columns]` view of the value its
/// blocks step through, from row `row` and column `column`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SliceView {
    /// Rows of each slab of the view: the value's, or for a matrix the ones up to the slice's end.
    pub(crate) rows: usize,
    /// Columns of the value.
    pub(crate) columns: usize,
    /// The slice's first row.
    pub(crate) row: usize,
    /// The slice's first column.
    pub(crate) column: usize,
}

/// How the TPU GEMM reads `operand`, the slice from `starts` of a value laid out with contiguous `strides`, in
/// place: when its leading dims are the value's (but for fewer of the first), its columns are the value's or
/// start, span and lie in whole 128-lane tiles, and its rows are the value's or start and span whole tiles of rows
/// (a matrix's may also be its first rows); else `None`.
pub(crate) fn slice_view(
    operand: &crate::graph::TensorType,
    strides: &[usize],
    starts: &[usize],
) -> Option<SliceView> {
    let r = operand.shape.len();

    if r < 2 || strides.len() != r || starts.len() != r || strides[r - 1] != 1 {
        return None;
    }

    let tile = crate::device::tpu::layout::Layout::new(operand.dtype, &[1, 1])
        .ok()?
        .rows;

    // The value's extent along dim `d` past the first.
    let extent = |d: usize| strides[d - 1] / strides[d];
    let (m, n) = (operand.shape[r - 2], operand.shape[r - 1]);
    let (row, column, columns) = (starts[r - 2], starts[r - 1], strides[r - 2]);

    let whole = |start: usize, size: usize, of: usize, unit: usize| {
        (start == 0 && size == of) || [start, size, of].iter().all(|v| v.is_multiple_of(unit))
    };

    if !whole(column, n, columns, crate::codegen::devices::tpu::LANES) {
        return None;
    }

    let rows = match r {
        // A matrix is viewed down to the slice's last row.
        2 if row == 0 || (row.is_multiple_of(tile) && m.is_multiple_of(tile)) => row + m,
        2 => return None,
        _ => {
            let leading = starts[..r - 2].iter().all(|&s| s == 0)
                && (1..r - 2).all(|d| operand.shape[d] == extent(d));

            if !(leading && whole(row, m, extent(r - 2), tile)) {
                return None;
            }

            extent(r - 2)
        }
    };

    Some(SliceView {
        rows,
        columns,
        row,
        column,
    })
}

/// The TPU GEMM reads slices `slice_view` takes, in a step of its own or a `dot_general` fusion.
impl crate::codegen::dot_general::StridedOperands for Tpu {
    /// None: it reads a slice in place only knowing where the slice starts (`reads_slice`).
    fn reads_strided(
        _: &crate::graph::Primitive,
        _: [&crate::graph::TensorType; 2],
        _: usize,
        _: &[usize],
    ) -> bool {
        false
    }

    /// Whether `kernel` is the GEMM's (`dot_general`) and `slice_view` takes the slice.
    fn reads_slice(
        kernel: &str,
        _: &crate::graph::Primitive,
        operands: [&crate::graph::TensorType; 2],
        k: usize,
        strides: &[usize],
        starts: &[usize],
    ) -> bool {
        kernel == "dot_general" && slice_view(operands[k], strides, starts).is_some()
    }
}
