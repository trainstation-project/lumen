//! Metal encoding of factory primitives: fill and iota (random bits are codegen/random's).
use crate::Tensor;
use crate::codegen::utils::mps::Mps;
use crate::graph::Primitive::*;
use crate::plan::Step;
use crate::runtime::kernel::Launch as _;
use crate::runtime::kernel::mps::MpsLaunch;
use crate::runtime::mps::{Grid, element_arg, elementwise_grid, u32_arg};

/// Encodes one factory `step` writing into `output`, keeping `keep` alive until the command buffer finishes.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let out = &step.output.1;

    // The kernel binds the step's inputs, then its output.
    let mut buffers = inputs.to_vec();
    buffers.push(output.cast_const());

    match &step.primitive {
        Full { fill_value, .. } => {
            let n = out.numel();
            let args = [element_arg(out.dtype, *fill_value), u32_arg(n as u32)];
            let kernel = format!("fill_{}", out.dtype.size_of());
            let grid = elementwise_grid(n, out.dtype);
            Mps::launch(MpsLaunch {
                kernel: &kernel,
                buffers: &buffers,
                args: &args,
                grid,
                keep,
                name: step.label,
            })
        }
        Iota { dimension, .. } => {
            let (outer, inner) = (&out.shape[..*dimension], &out.shape[dimension + 1..]);

            // Grid is [inner, iota axis, outer] so the kernel reads its index from gid.y.
            let grid = Grid::Threads([
                inner.iter().product(),
                out.shape[*dimension],
                outer.iter().product(),
            ]);

            Mps::launch(MpsLaunch {
                kernel: &format!("iota_{}", out.dtype),
                buffers: &buffers,
                args: &[],
                grid,
                keep,
                name: step.label,
            })
        }
        _ => unreachable!("a factory primitive"),
    }
}
