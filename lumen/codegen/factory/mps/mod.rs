//! Metal fills and iotas: `kernel.metal`'s kernels and their launches.

use crate::codegen::devices::mps::Mps;
use crate::graph::Primitive::{Full, Iota};
use crate::runtime::kernel::Launch as _;
use crate::runtime::kernel::mps::{Grid, MpsLaunch, element_arg, elementwise_grid, u32_arg};

/// The Metal fill and iota kernels of a lone `full` or `iota`.
pub(crate) struct Factory;

impl crate::codegen::kernel::Kernel for Factory {
    type Step = crate::plan::MpsStep;
    type Buffers = crate::plan::step::mps::Encoding;
    type Config = ();

    /// Launches the fill or iota over the output.
    fn run(
        &self,
        step: &crate::plan::MpsStep,
        _: Option<&()>,
        e: &crate::plan::step::mps::Encoding,
    ) -> Result<(), String> {
        let (inputs, output, scratch, keep) =
            (e.inputs.as_slice(), e.output, e.scratch, e.keep.clone());
        let _ = scratch;

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
}
