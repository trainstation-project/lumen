//! Launches the Metal dynamic-slice kernels.
use crate::Tensor;
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;
use crate::kernels::mps::{Grid, dims_arg, elementwise_grid, launch, u32_arg, u64_arg};

/// Copies the `ty`-shaped tensor at `src` to `dst` on the GPU.
pub(crate) fn copy(
    src: *const u8,
    dst: *mut u8,
    ty: &crate::graph::TensorType,
    keep: Vec<Tensor>,
    name: &'static str,
) -> Result<(), String> {
    let n = ty.numel();
    let kernel = format!("dynamic_slice_copy_{}", ty.dtype.size_of());
    let grid = elementwise_grid(n, ty.dtype);

    launch(
        &kernel,
        &[src, dst.cast_const()],
        &[u32_arg(n as u32)],
        grid,
        keep,
        name,
    )
}

/// Highest operand rank the Metal kernels take.
const MAX_RANK: usize = 8;

/// Encodes a `DynamicSlice` or `DynamicUpdateSlice` step; an out-of-place update first copies
/// the operand into the output.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let x = &step.inputs[0].1;
    let rank = x.shape.len();

    if rank > MAX_RANK {
        return Err(format!(
            "{}: {rank} dimensions are more than its MPS kernels take ({MAX_RANK})",
            step.label
        ));
    }

    let bytes = x.dtype.size_of();

    let (source, block, first) = match &step.primitive {
        Primitive::DynamicSlice { slice_sizes } => (inputs[0], slice_sizes.clone(), 1),
        Primitive::DynamicUpdateSlice => (inputs[1], step.inputs[1].1.shape.clone(), 2),
        _ => unreachable!("a dynamic slice"),
    };

    let update = matches!(step.primitive, Primitive::DynamicUpdateSlice);

    if update && output.cast_const() != inputs[0] {
        copy(inputs[0], output, x, keep.clone(), step.label)?;
    }

    let index_dtype = step
        .inputs
        .get(first)
        .map_or("i32".into(), |(_, t)| t.dtype.to_string());

    // Unused index slots point at the source buffer; the kernel only reads the first `rank`.
    let mut buffers = vec![source, output.cast_const()];
    buffers.extend((0..MAX_RANK).map(|d| *inputs.get(first + d).unwrap_or(&source)));
    let count: usize = block.iter().product();

    let args = [
        dims_arg(x.shape.clone()),
        dims_arg(block.clone()),
        u32_arg(rank as u32),
        u64_arg(count),
    ];

    // Use 32-bit offsets when the operand fits, 64-bit otherwise.
    let width = match x.numel() <= u32::MAX as usize {
        true => 32,
        false => 64,
    };

    let kernel = match update {
        true => format!("dynamic_update_slice_{bytes}_{index_dtype}_{width}"),
        false => format!("dynamic_slice_{bytes}_{index_dtype}_{width}"),
    };

    let grid = Grid::Threads([count.div_ceil(16 / bytes), 1, 1]);
    launch(&kernel, &buffers, &args, grid, keep, step.label)
}
