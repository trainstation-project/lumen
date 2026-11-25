//! CUDA's (CuTe DSL) passes: it merges dots, fuses attention and rotary pair swaps, then fuses elementwise work and
//! GEMM epilogues, the fusing passes only with `fuse`.

use super::{Pass, Passes};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cuda, DeviceModule};
use crate::graph_compiler::passes::alone::Alone;
use crate::graph_compiler::passes::attention::FlashAttention;
use crate::graph_compiler::passes::cse::Cse;
use crate::graph_compiler::passes::fusion::Fusion;
use crate::graph_compiler::passes::merge_dots::{MergeDots, MergeDotsBackward, Target};
use crate::graph_compiler::passes::preserve_precision::PreservePrecision;
use crate::graph_compiler::passes::simplify::Simplify;
use crate::graph_compiler::passes::swap_pairs::SwapPairs;
use crate::graph_compiler::passes::take_along_axis::TakeAlongAxis;
use crate::graph_compiler::passes::unsupported::Unsupported;

impl Passes for Cuda {
    /// CUDA's passes, in order. Merged dots pack for the GEMM epilogue's 16-byte vectors (8 columns of 16-bit
    /// products), their slices read in place by what CUDA fusion fuses.
    fn passes<'a>(config: &CompilerConfig) -> Vec<Option<Pass<DeviceModule<'a, Cuda>>>> {
        let fuse = config.fuse;
        let target = Target {
            reads: crate::codegen::fusion::cuda::fusible,
            views: true,
            lanes: 1,
            columns: 8,
        };

        vec![
            Self::pass::<PreservePrecision>(true, config),
            Self::add(Cse),
            Self::add(Simplify { matched: false }),
            Self::pass::<Unsupported>(true, config),
            Self::pass::<TakeAlongAxis>(true, config),
            Self::pass::<MergeDotsBackward>(fuse, config),
            Self::maybe(MergeDots::configured_with(config, target).filter(|_| fuse)),
            Self::pass::<FlashAttention>(fuse, config),
            Self::add(Simplify { matched: true }),
            Self::pass::<SwapPairs>(true, config),
            Self::pass::<Fusion>(fuse, config),
            Self::pass::<Alone>(true, config),
        ]
    }
}
