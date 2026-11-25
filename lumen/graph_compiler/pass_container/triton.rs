//! Triton's passes: MPS's GPU pass list (see `mps.rs`) without the Neural Engine or a canonical dot layout, plus
//! partial row reductions in GEMMs and horizontal fusion.

use super::{Pass, Passes};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::alone::Alone;
use crate::graph_compiler::passes::attention::{FlashAttention, FlashAttentionBackward};
use crate::graph_compiler::passes::chunked_cross_entropy::ChunkedCrossEntropy;
use crate::graph_compiler::passes::cse::Cse;
use crate::graph_compiler::passes::diamonds::Diamonds;
use crate::graph_compiler::passes::dot::{
    split_k::SplitK, strength::DotStrength, wide_rows::WideRows,
};
use crate::graph_compiler::passes::fusion::Fusion;
use crate::graph_compiler::passes::horizontal::HorizontalFusion;
use crate::graph_compiler::passes::linear_cross_entropy::LinearCrossEntropy;
use crate::graph_compiler::passes::linear_reordering::LinearReordering;
use crate::graph_compiler::passes::merge_dots::{MergeDots, MergeDotsBackward, Target};
use crate::graph_compiler::passes::place_concats::PlaceConcats;
use crate::graph_compiler::passes::place_views::PlaceViews;
use crate::graph_compiler::passes::preserve_precision::PreservePrecision;
use crate::graph_compiler::passes::simplify::Simplify;
use crate::graph_compiler::passes::swap_pairs::SwapPairs;
use crate::graph_compiler::passes::take_along_axis::TakeAlongAxis;
use crate::graph_compiler::passes::unsupported::Unsupported;
use crate::graph_compiler::passes::updates::Updates;

impl Passes for Triton {
    /// Triton's passes, in order. Merged dots pack for 8-column GEMM tiles, their slices read in place by what
    /// Triton fusion fuses. Update chains of 2 or more fuse: a kernel binds any number of buffers.
    fn passes<'a>(config: &CompilerConfig) -> Vec<Option<Pass<DeviceModule<'a, Triton>>>> {
        let fuse = config.fuse;
        let target = Target {
            reads: crate::codegen::fusion::triton::fusible,
            views: true,
            lanes: 1,
            columns: 8,
        };
        let row_tile = crate::codegen::dot_general::triton::ROW_TILE;
        let updates = Updates {
            least: 2,
            most: usize::MAX,
        };

        vec![
            Self::pass::<PreservePrecision>(true, config),
            Self::add(true, Cse),
            Self::add(true, Simplify { matched: false }),
            Self::pass::<Unsupported>(true, config),
            Self::pass::<TakeAlongAxis>(true, config),
            Self::pass::<ChunkedCrossEntropy>(true, config),
            Self::pass::<MergeDotsBackward>(fuse, config),
            Self::add(fuse && config.merge_dots, MergeDots { target }),
            Self::pass::<FlashAttention>(fuse, config),
            Self::pass::<FlashAttentionBackward>(fuse, config),
            Self::pass::<LinearCrossEntropy>(true, config),
            Self::add(true, Simplify { matched: true }),
            Self::pass::<SwapPairs>(true, config),
            Self::pass::<DotStrength>(true, config),
            Self::add(
                config.linear_reordering,
                LinearReordering {
                    tile: Some(row_tile),
                },
            ),
            Self::pass::<SplitK>(true, config),
            Self::add(fuse, updates),
            Self::pass::<Diamonds>(true, config),
            Self::pass::<PlaceConcats>(fuse, config),
            Self::pass::<Fusion>(fuse, config),
            Self::pass::<WideRows>(true, config),
            Self::pass::<PlaceViews>(true, config),
            Self::pass::<Alone>(true, config),
            Self::pass::<HorizontalFusion>(fuse, config),
        ]
    }
}
