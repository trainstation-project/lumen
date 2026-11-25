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
            Self::add(config.preserve_precision, PreservePrecision),
            Self::add(true, Cse),
            Self::add(true, Simplify { matched: false }),
            Self::add(true, Unsupported::of::<Self>()),
            Self::add(true, TakeAlongAxis),
            Self::add(fuse && config.merge_dots, MergeDotsBackward),
            Self::add(fuse && config.merge_dots, MergeDots { target }),
            Self::add(fuse && config.flash_attention, FlashAttention),
            Self::add(true, Simplify { matched: true }),
            Self::add(true, SwapPairs),
            Self::add(fuse, Fusion),
            Self::add(true, Alone),
        ]
    }
}
