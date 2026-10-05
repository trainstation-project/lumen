//! gather and scatter_add on MPS (`kernels.metal`): a gather a thread an
//! element of its result; a scatter_add a thread a column of its value,
//! adding its updates in index order (deterministic), or, of few rows and
//! updates, a thread an element, scanning the indices in order. A scatter_add whose
//! value the plan put in its operand's buffer (`graph/plan.rs`: nothing
//! reads the operand after it) adds the updates there, in place; otherwise
//! it copies the operand first.

use crate::graph::Primitive;
use crate::graph::plan::Step;
use crate::ops::mps::{Grid, launch, u64_arg};
use crate::{DType, Tensor};

/// Threads a grid row takes: past it, rows (Metal grids are 32-bit a
/// dimension).
const ROW: usize = 1 << 20;

/// The most indices a scatter_add's rows by its updates (n * m) for a
/// thread an element, each scanning all m indices.
const SCAN: usize = 1 << 16;

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (x, indices) = (&step.inputs[0].1, &step.inputs[1].1);
    let axis = match step.primitive {
        Primitive::Gather { axis } | Primitive::ScatterAdd { axis } => axis,
        _ => unreachable!("a gather or scatter_add"),
    };
    let index = match indices.dtype {
        DType::I32 => "i32",
        _ => "i64",
    };
    let (n, m) = (x.shape[axis], indices.numel());
    let inner: usize = x.shape[axis + 1..].iter().product();
    let scatter = matches!(step.primitive, Primitive::ScatterAdd { .. });
    // A gather's result's elements; a scatter's columns (outer * inner),
    // or its elements when each scans the indices.
    let scan = scatter && n * m <= SCAN;
    let total = match scatter && !scan {
        true => step.output.1.numel() / n.max(1),
        false => step.output.1.numel(),
    };
    let (kernel, buffers) = match scatter {
        false => (
            format!("gather_{}_{index}", x.dtype.size_of()),
            vec![inputs[0], inputs[1], output.cast_const()],
        ),
        true => {
            if x.dtype == DType::Bool {
                return Err(format!(
                    "{}: booleans have no sum to scatter-add",
                    step.label
                ));
            }
            // The operand's value, if not already in the output's buffer.
            if output.cast_const() != inputs[0] {
                let name = step.label;
                crate::ops::dynamic_slice::mps::copy(inputs[0], output, x, keep.clone(), name)?;
            }
            let kernel = match scan {
                true => "scatter_add_scan",
                false => "scatter_add",
            };
            (
                format!("{kernel}_{}_{index}", x.dtype.name()),
                vec![inputs[2], inputs[1], output.cast_const()],
            )
        }
    };
    if total == 0 || n == 0 {
        crate::profiler::clear_next_launch();
        return Ok(());
    }
    let args = [
        u64_arg(n),
        u64_arg(m),
        u64_arg(inner.max(1)),
        u64_arg(total),
    ];
    // A scan's threadgroups of 256 (`SCAN_BLOCK`), each element a thread.
    let grid = match scan {
        true => Grid::Groups([total.div_ceil(256), 1, 1]),
        false => Grid::Threads([total.min(ROW), total.div_ceil(ROW), 1]),
    };
    launch(&kernel, &buffers, &args, grid, keep, step.label)
}
