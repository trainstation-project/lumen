//! CUDA's (CuTe DSL) passes: it merges dots, fuses attention and rotary pair swaps, then fuses elementwise work and
//! GEMM epilogues, the fusing passes only with `fuse`.

use super::{Passes, Pipeline};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cuda, Device, DeviceModule};
use crate::graph_compiler::passes::alone::Alone;
use crate::graph_compiler::passes::attention::FlashAttention;
use crate::graph_compiler::passes::fusion::Fusion;
use crate::graph_compiler::passes::merge_dots::{MergeDots, MergeDotsBackward, Target};
use crate::graph_compiler::passes::simplify::Simplify;
use crate::graph_compiler::passes::swap_pairs::SwapPairs;
use crate::graph_compiler::passes::take_along_axis::TakeAlongAxis;
use crate::graph_compiler::passes::unsupported::Unsupported;

impl Passes for Cuda {
    /// CUDA's passes, in order. Merged dots pack for the GEMM epilogue's 16-byte vectors (8 columns of 16-bit
    /// products), their slices read in place by what CUDA fusion fuses.
    fn passes<'a>(config: &CompilerConfig) -> Pipeline<DeviceModule<'a, Cuda>> {
        let fuse = config.fuse;
        let target = Target {
            reads: crate::codegen::fusion::cuda::fusible,
            views: true,
            lanes: 1,
            columns: 8,
        };
        let mut pipeline = Pipeline::new(Cuda::NAME);

        pipeline
            .pass::<Unsupported>(config)
            .pass::<TakeAlongAxis>(config)
            .pass_if::<MergeDotsBackward>(fuse, config)
            .maybe(MergeDots::configured_with(config, target).filter(|_| fuse))
            .pass_if::<FlashAttention>(fuse, config)
            .add(Simplify { matched: true })
            .pass::<SwapPairs>(config)
            .pass_if::<Fusion>(fuse, config)
            .pass::<Alone>(config);

        pipeline
    }
}
