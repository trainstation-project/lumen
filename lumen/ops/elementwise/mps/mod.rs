//! Elementwise primitives on MPS (`kernels.metal`): one kernel per op and
//! dtype, named after the primitive, each thread taking a few elements.

use crate::Tensor;
use crate::graph::Primitive::*;
use crate::graph::plan::Step;
use crate::ops::mps::{elementwise_grid, launch_step, u32_arg};

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (x, out) = (&step.inputs[0].1, &step.output.1);
    let n = out.numel();
    // Threads take elements of the input (for select, of the cases).
    let (kernel, grid) = match &step.primitive {
        Cast { .. } => (
            format!("convert_{}_{}", x.dtype, out.dtype),
            elementwise_grid(n, x.dtype),
        ),
        Select => (
            format!("select_{}", out.dtype.size_of()),
            elementwise_grid(n, out.dtype),
        ),
        p => (
            format!("{}_{}", p.name(), x.dtype),
            elementwise_grid(n, x.dtype),
        ),
    };
    launch_step(
        step,
        &kernel,
        inputs,
        output,
        &[u32_arg(n as u32)],
        grid,
        keep,
    )
}
