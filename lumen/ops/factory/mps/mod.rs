//! Factory primitives on MPS (`kernels.metal`): full, a dtype-agnostic fill a
//! few elements a thread, and iota, a thread per element on an inner x size
//! x outer grid whose y coordinate is the value.

use crate::Tensor;
use crate::graph::Primitive::*;
use crate::graph::plan::Step;
use crate::ops::mps::{Grid, element_arg, elementwise_grid, launch_step, u32_arg};

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
