//! Metal launch of `scatter_add`: copies the operand into the output, then adds the updates in.

use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::kernels::mps::{Grid, launch, u64_arg};
use crate::{DType, Tensor};

/// Most threads per grid row; larger launches spill into more rows.
const ROW: usize = 1 << 20;

/// Largest `n * m` (axis length times index count) for which each output element scans all indices.
const SCAN: usize = 1 << 16;

/// Encodes `out = x` with `updates` rows added at `indices` along the step's axis; inputs are
/// `[x, indices, updates]`.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (x, indices) = (&step.inputs[0].1, &step.inputs[1].1);

    let Primitive::ScatterAdd { axis } = step.primitive else {
        unreachable!("a scatter_add")
    };

    let index = match indices.dtype {
        DType::I32 => "i32",
        _ => "i64",
    };

    let (n, m) = (x.shape[axis], indices.numel());
    let inner: usize = x.shape[axis + 1..].iter().product();
    // The scan kernel runs a thread per output element; the serial one a thread per position
    // off the axis, looping over the indices in order.
    let scan = n * m <= SCAN;

    let total = match scan {
        true => step.output.1.numel(),
        false => step.output.1.numel() / n.max(1),
    };

    if x.dtype == DType::Bool {
        return Err(format!(
            "{}: booleans have no sum to scatter-add",
            step.label
        ));
    }

    // Copy x into the output first unless they already share a buffer.
    if output.cast_const() != inputs[0] {
        let name = step.label;

        crate::codegen::indexing::mps::dynamic_slice::copy(
            inputs[0],
            output,
            x,
            keep.clone(),
            name,
        )?;
    }

    let kernel = match scan {
        true => "scatter_add_scan",
        false => "scatter_add",
    };

    let kernel = format!("{kernel}_{}_{index}", x.dtype.name());
    let buffers = vec![inputs[2], inputs[1], output.cast_const()];

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

    let grid = match scan {
        true => Grid::Groups([total.div_ceil(256), 1, 1]),
        false => Grid::Threads([total.min(ROW), total.div_ceil(ROW), 1]),
    };

    launch(&kernel, &buffers, &args, grid, keep, step.label)
}
