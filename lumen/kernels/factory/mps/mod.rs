//! Metal encoding of factory primitives: fill and iota (random bits are codegen/random's).
use crate::Tensor;
use crate::graph::Primitive::*;
use crate::graph_compiler::plan::Step;
use crate::kernels::mps::{Grid, element_arg, elementwise_grid, launch_step, u32_arg};

/// Encodes one factory `step` writing into `output`, keeping `keep` alive until the command buffer finishes.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let out = &step.output.1;

    match &step.primitive {
        Full { fill_value, .. } => {
            let n = out.numel();
            let args = [element_arg(out.dtype, *fill_value), u32_arg(n as u32)];
            let kernel = format!("fill_{}", out.dtype.size_of());
            let grid = elementwise_grid(n, out.dtype);
            launch_step(step, &kernel, inputs, output, &args, grid, keep)
        }
        Iota { dimension, .. } => {
            let (outer, inner) = (&out.shape[..*dimension], &out.shape[dimension + 1..]);

            // Grid is [inner, iota axis, outer] so the kernel reads its index from gid.y.
            let grid = Grid::Threads([
                inner.iter().product(),
                out.shape[*dimension],
                outer.iter().product(),
            ]);

            launch_step(
                step,
                &format!("iota_{}", out.dtype),
                inputs,
                output,
                &[],
                grid,
                keep,
            )
        }
        _ => unreachable!("a factory primitive"),
    }
}
