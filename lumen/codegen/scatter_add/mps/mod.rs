//! Metal scatter-adds: the fixed `scatter_add` kernels (`kernel.metal`) and their launch, which copies the
//! operand into the output, then adds the updates in.

use crate::graph::{Primitive, TensorType};
use crate::graph_compiler::plan::Step;
use crate::runtime::mps::{Grid, element_arg, elementwise_grid, launch, u32_arg, u64_arg};
use crate::{DType, Scalar, Tensor};

/// Threads of the row kernel's threadgroup, one a column (`SCATTER_THREADS` in `kernel.metal`).
const THREADS: usize = 256;
/// Threads of the atomic kernel's threadgroup: on an M4 Pro, 16M float32 updates add in 872 µs in
/// groups of 1024 and 1709 µs in groups of 256.
const ATOMIC_THREADS: usize = 1024;

/// Scratch bytes of a scatter-add of `indices` along an axis `n` long: each position's update count and
/// start (`2n + 1` u32s) and the updates' order (`m` u32s).
pub(crate) fn scratch_bytes(x: &TensorType, indices: &TensorType, axis: usize) -> usize {
    (2 * x.shape[axis] + 1 + indices.numel()) * 4
}

/// Encodes `out = x` with `updates` rows added at `indices` along the step's axis; inputs are
/// `[x, indices, updates]`. Unless `mps.deterministic`, float32, float16, bfloat16, int32 and uint32
/// updates are added in by atomics, a thread an element (torch's `index_add`); otherwise they are
/// sorted by position (a count, its prefix sum and a placement, in `scratch`), then a threadgroup per
/// position with updates adds them in update order, a thread a column.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
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
    let outer: usize = x.shape[..axis].iter().product();
    let total = step.output.1.numel();

    if x.dtype == DType::Bool {
        return Err(format!(
            "{}: booleans have no sum to scatter-add",
            step.label
        ));
    }

    // Copy x into the output first unless they already share a buffer.
    if output.cast_const() != inputs[0] {
        let name = step.label;

        crate::codegen::dynamic_slice::mps::copy(inputs[0], output, x, keep.clone(), name)?;
    }

    if total == 0 || n == 0 || m == 0 {
        crate::profiler::clear_next_launch();
        return Ok(());
    }

    let (updates, index_buffer, out) = (inputs[2], inputs[1], output.cast_const());
    let label = step.label;

    // Unless deterministic, torch's `index_add`: a thread per update element, added in atomically.
    let count = outer.max(1) * m * inner.max(1);
    let atomic = matches!(
        x.dtype,
        DType::F32 | DType::F16 | DType::BF16 | DType::I32 | DType::U32
    );

    if atomic
        && !crate::graph_compiler::config::configs().mps.deterministic
        && count <= u32::MAX as usize
    {
        let name = format!("scatter_add_atomic_{}_{index}", x.dtype.name());
        let args = [n, m, inner.max(1), count].map(|v| u32_arg(v as u32));

        let grid = Grid::GroupsOf {
            groups: [count.div_ceil(ATOMIC_THREADS), 1, 1],
            threads: ATOMIC_THREADS,
        };

        return launch(
            &name,
            &[updates, index_buffer, out],
            &args,
            grid,
            keep,
            label,
        );
    }

    // Each position's update count (then the place kernel's cursor) and start, and the updates' order.
    let at = |words: usize| scratch.wrapping_add(words * 4).cast_const();
    let (counts, starts, order) = (at(0), at(n), at(2 * n + 1));

    let zero = [
        element_arg(DType::F32, Scalar::Float(0.0)),
        u32_arg(n as u32),
    ];
    let grid = elementwise_grid(n, DType::F32);
    launch("fill_4", &[counts], &zero, grid, Vec::new(), label)?;

    let count = format!("scatter_add_count_{index}");
    let grid = Grid::Threads([m, 1, 1]);
    launch(
        &count,
        &[index_buffer, counts],
        &[u64_arg(m)],
        grid,
        Vec::new(),
        label,
    )?;

    let grid = Grid::GroupsOf {
        groups: [1, 1, 1],
        threads: 1024,
    };
    launch(
        "scatter_add_starts",
        &[counts, starts],
        &[u64_arg(n)],
        grid,
        Vec::new(),
        label,
    )?;

    let place = format!("scatter_add_place_{index}");
    let buffers = [index_buffer, starts, counts, order];
    launch(
        &place,
        &buffers,
        &[u64_arg(m)],
        Grid::Threads([m, 1, 1]),
        Vec::new(),
        label,
    )?;

    let rows = format!("scatter_add_rows_{}_{index}", x.dtype.name());
    let buffers = [updates, index_buffer, out, starts, order];
    let args = [u64_arg(n), u64_arg(m), u64_arg(inner.max(1))];

    let grid = Grid::GroupsOf {
        groups: [n, inner.max(1).div_ceil(THREADS), outer.max(1)],
        threads: THREADS,
    };

    launch(&rows, &buffers, &args, grid, keep, label)
}
