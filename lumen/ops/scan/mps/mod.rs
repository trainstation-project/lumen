//! cumsum on MPS (`kernels.metal`): the operand viewed as [a, n, b], each of
//! its lines of n elements along the axis scanned. A row (b = 1) of more
//! than [`ROW_THREADS`] elements takes threadgroups (`cumsum_rows`, blocks
//! of 1024 elements scanned in parallel, each carried to the next); any
//! other line (a column, a short row) threads (`cumsum_lines`). With too
//! few lines to fill the GPU, each is split into chunks, scanned in three
//! launches (reduce-then-scan): each chunk's sum into partials (in the
//! scratch the planner set aside), the partials scanned in place, then each
//! chunk from the sum of those before it. Every output is computed in a
//! fixed order: deterministic.

use crate::graph::plan::Step;
use crate::graph::{Primitive, TensorType};
use crate::ops::mps::{Grid, launch, u32_arg, u64_arg};
use crate::ops::reduce::mps::kernel_dtype;
use crate::{DType, Tensor};

/// Rows of more elements take a threadgroup; shorter ones, a thread.
const ROW_THREADS: usize = 64;
/// Rows are split into chunks while fewer than this many threadgroups would
/// scan them, each chunk of at least `ROW_CHUNK` elements.
const ROW_GROUPS: usize = 256;
const ROW_CHUNK: usize = 4096;
/// Lines a thread each: split while fewer than this many threads would scan
/// them, each chunk of at least `LINE_CHUNK` elements.
const LINE_THREADS: usize = 16384;
const LINE_CHUNK: usize = 64;

/// How a cumsum of `x` along `axis` runs: [a, n, b], whether a row kernel
/// scans it, and its chunks of `chunk` elements.
struct Scan {
    a: usize,
    n: usize,
    b: usize,
    rows: bool,
    chunks: usize,
    chunk: usize,
}

fn scan(x: &TensorType, axis: usize) -> Scan {
    let a: usize = x.shape[..axis].iter().product();
    let (n, b): (usize, usize) = (x.shape[axis], x.shape[axis + 1..].iter().product());
    let rows = b == 1 && n > ROW_THREADS;
    let (lines, target, least) = match rows {
        true => (a, ROW_GROUPS, ROW_CHUNK),
        false => (a * b, LINE_THREADS, LINE_CHUNK),
    };
    let chunks = match lines {
        0 => 1,
        _ => target.div_ceil(lines).min(n / least).max(1),
    };
    let chunk = n.div_ceil(chunks).max(1);
    Scan {
        a,
        n,
        b,
        rows,
        chunks: n.div_ceil(chunk).max(1),
        chunk,
    }
}

/// The scratch a cumsum of `x` along `axis`, accumulating in `accum`,
/// needs: a split scan's partials (none for the others).
pub(crate) fn scratch_bytes(x: &TensorType, axis: usize, accum: DType) -> usize {
    let s = scan(x, axis);
    match s.chunks {
        1 => 0,
        chunks => s.a * s.b * chunks * accum.size_of(),
    }
}

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let Primitive::Cumsum {
        axis,
        reverse,
        accum_dtype,
    } = &step.primitive
    else {
        unreachable!("a cumsum")
    };
    let x = &step.inputs[0].1;
    let Scan {
        a,
        n,
        b,
        rows,
        chunks,
        chunk,
    } = scan(x, *axis);
    if a * b * chunks > u32::MAX as usize {
        return Err(format!(
            "{}: {} lines are more than MPS kernels index",
            step.label,
            a * b
        ));
    }
    if chunks > 1 && scratch.is_null() {
        return Err(format!(
            "{}: a split scan needs scratch for its partials: compile the graph for MPS (lumen.compile, Plan(graph, \"mps\"))",
            step.label
        ));
    }
    let kernel = match rows {
        true => format!("cumsum_rows_{}", kernel_dtype(x.dtype, *accum_dtype)),
        false => format!("cumsum_lines_{}", kernel_dtype(x.dtype, *accum_dtype)),
    };
    let grid = match rows {
        true => Grid::Groups([chunks, a, 1]),
        false => Grid::Threads([a * b * chunks, 1, 1]),
    };
    // Unsplit, the partials are never read: any buffer.
    let partials = match chunks {
        1 => output.cast_const(),
        _ => scratch.cast_const(),
    };
    let buffers = [inputs[0], output.cast_const(), partials];
    let pass = |sum_only: bool, keep: Vec<Tensor>| {
        let mut args = vec![
            u64_arg(n),
            u32_arg(u32::from(*reverse)),
            u64_arg(chunk),
            u32_arg(chunks as u32),
            u32_arg(u32::from(sum_only)),
        ];
        if !rows {
            args.extend([u32_arg(b as u32), u32_arg((a * b) as u32)]);
        }
        launch(&kernel, &buffers, &args, grid, keep, step.label)
    };
    if chunks > 1 {
        // Each launch's writes are read by the next alone.
        pass(true, keep.clone())?;
        let lines = a * b;
        let args = [
            u64_arg(chunks),
            u32_arg(0),
            u64_arg(chunks),
            u32_arg(1),
            u32_arg(0),
            u32_arg(1),
            u32_arg(lines as u32),
        ];
        let kernel = format!("cumsum_lines_{accum_dtype}");
        let grid = Grid::Threads([lines, 1, 1]);
        launch(
            &kernel,
            &[partials, partials, partials],
            &args,
            grid,
            keep.clone(),
            step.label,
        )?;
    }
    pass(false, keep)
}
