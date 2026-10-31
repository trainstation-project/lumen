//! Metal encoding of `Cumsum`, splitting long scans into chunks scanned in three passes.
use crate::codegen::reduction::mps::kernel_dtype;
use crate::codegen::scan::ScanCodegen;
use crate::codegen::utils::mps::Mps;
use crate::graph::{Primitive, TensorType};
use crate::graph_compiler::plan::Step;
use crate::kernels::mps::{Grid, launch, u32_arg, u64_arg};
use crate::{DType, Tensor};

/// Scan length above which a contiguous axis uses the threadgroup row kernel.
const ROW_THREADS: usize = 64;
/// Target number of row threadgroups when splitting rows into chunks.
const ROW_GROUPS: usize = 256;
/// Smallest chunk a split row scan gives a threadgroup.
const ROW_CHUNK: usize = 4096;
/// Target number of threads when splitting strided lines into chunks.
const LINE_THREADS: usize = 16384;
/// Smallest chunk a split line scan gives a thread.
const LINE_CHUNK: usize = 64;

/// How a cumsum over one axis is laid out and split.
struct Scan {
    /// Product of the dims before the axis.
    a: usize,
    /// Length of the scanned axis.
    n: usize,
    /// Product of the dims after the axis (the scan stride).
    b: usize,
    /// Whether the axis is contiguous and long enough for the threadgroup row kernel.
    rows: bool,
    /// Number of chunks each line is split into.
    chunks: usize,
    /// Elements per chunk.
    chunk: usize,
}

/// Picks the kernel and chunking for a cumsum of `x` along `axis`, splitting lines so there is enough parallel work.
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

/// Metal runs a cumsum with its fixed kernel (`encode`): there is no code to generate.
impl ScanCodegen for Mps {
    /// Nothing: every scan fails.
    type Code = ();

    /// Fails: `x`'s cumsum runs Metal's fixed kernel.
    fn scan(x: &TensorType, _: usize, _: bool, _: DType) -> Result<(), String> {
        Err(format!("a cumsum of {x} runs Metal's fixed kernel"))
    }
}

/// Bytes of scratch for per-chunk partial sums; zero when the scan is not split.
pub(crate) fn scratch_bytes(x: &TensorType, axis: usize, accum: DType) -> usize {
    let s = scan(x, axis);

    match s.chunks {
        1 => 0,
        chunks => s.a * s.b * chunks * accum.size_of(),
    }
}

/// Encodes a cumsum step. A split scan runs three passes: per-chunk totals into scratch,
/// an in-place scan of those totals, then the chunked scan carrying each chunk's prefix.
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
        pass(true, keep.clone())?;
        // Scans each line's chunk totals in place, as a contiguous line of length `chunks`.
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
