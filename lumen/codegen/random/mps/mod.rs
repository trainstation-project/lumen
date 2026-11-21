//! Metal random bits: the `philox_bits` function every kernel is compiled with (`philox.metal`), and the
//! standalone `random_bits` kernel (`kernel.metal`) and its launch.

use super::RandomCodegen;
use crate::codegen::devices::mps::Mps;
use crate::graph::Primitive;
use crate::runtime::kernel::Launch as _;
use crate::runtime::kernel::mps::{MpsLaunch, elementwise_grid, u32_arg, u64_arg};

/// Metal random bits: a call to the prelude's `philox_bits`.
impl RandomCodegen for Mps {
    /// `philox_bits` of the 64-bit counter `start + offset + index`.
    fn bits(seed: &str, start: &str, offset: u64, index: &str) -> String {
        format!("philox_bits({seed}, {start} + {offset}ul + ulong({index}))")
    }
}

/// The Metal `random_bits` kernel of a lone `random_bits`, writing Philox words.
pub(crate) struct RandomBits;

impl crate::codegen::kernel::Kernel for RandomBits {
    type Step = crate::plan::MpsStep;
    type Buffers = crate::plan::step::mps::Encoding;
    type Config = ();

    /// Launches the kernel over the output's words.
    fn run(
        &self,
        step: &crate::plan::MpsStep,
        _: Option<&()>,
        e: &crate::plan::step::mps::Encoding,
    ) -> Result<(), String> {
        let (inputs, output, scratch, keep) =
            (e.inputs.as_slice(), e.output, e.scratch, e.keep.clone());
        let _ = scratch;

        let Primitive::RandomBits { offset, .. } = step.primitive else {
            unreachable!("a random_bits step")
        };

        let n = step.output.1.numel();
        let args = [u64_arg(offset as usize), u32_arg(n as u32)];
        let grid = elementwise_grid(n, step.output.1.dtype);

        // The kernel binds the step's inputs, then its output.
        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());

        Mps::launch(MpsLaunch {
            kernel: "random_bits",
            buffers: &buffers,
            args: &args,
            grid,
            keep,
            name: step.label,
        })
    }
}
