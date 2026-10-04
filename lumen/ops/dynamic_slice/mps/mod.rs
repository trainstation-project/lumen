//! dynamic_slice and dynamic_update_slice on MPS (`kernels.metal`): a thread
//! an element of the block, its start indices read from their device
//! buffers when the kernel runs. A dynamic_update_slice whose value the plan
//! put in its operand's buffer (`graph/plan.rs`: nothing reads the operand
//! after it) writes the update alone, in place; otherwise it copies the
//! operand first.

use crate::Tensor;
use crate::graph::Primitive;
use crate::graph::plan::Step;
use crate::ops::mps::{Grid, dims_arg, elementwise_grid, launch, u32_arg};

/// The most dimensions the kernels take (`DS_MAX_RANK`).
const MAX_RANK: usize = 8;

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
    // The operand's value, if not already in the output's buffer.
    if update && output.cast_const() != inputs[0] {
        let n = x.numel();
        let args = [u32_arg(n as u32)];
        let kernel = format!("dynamic_slice_copy_{bytes}");
        let buffers = [inputs[0], output.cast_const()];
        let grid = elementwise_grid(n, x.dtype);
        launch(&kernel, &buffers, &args, grid, keep.clone(), step.label)?;
    }
    // Each dimension's start index, the unused slots any buffer.
    let index_dtype = step
        .inputs
        .get(first)
        .map_or("i32".into(), |(_, t)| t.dtype.to_string());
    let mut buffers = vec![source, output.cast_const()];
    buffers.extend((0..MAX_RANK).map(|d| *inputs.get(first + d).unwrap_or(&source)));
    let args = [
        dims_arg(x.shape.clone()),
        dims_arg(block.clone()),
        u32_arg(rank as u32),
    ];
    let kernel = match update {
        true => format!("dynamic_update_slice_{bytes}_{index_dtype}"),
        false => format!("dynamic_slice_{bytes}_{index_dtype}"),
    };
    let grid = Grid::Threads([block.iter().product(), 1, 1]);
    launch(&kernel, &buffers, &args, grid, keep, step.label)
}
