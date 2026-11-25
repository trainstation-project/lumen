//! MPS's passes: the GPU pass list Triton shares (dots merged, attention and cross entropies fused, rows and
//! epilogues fused, the fusing passes only with `fuse`), plus the Neural Engine and concatenations fusion could not
//! place.

use super::{Pass, Passes};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::alone::Alone;
use crate::graph_compiler::passes::apple_neural_engine::NeuralEngine;
use crate::graph_compiler::passes::attention::{FlashAttention, FlashAttentionBackward};
use crate::graph_compiler::passes::chunked_cross_entropy::ChunkedCrossEntropy;
use crate::graph_compiler::passes::cse::Cse;
use crate::graph_compiler::passes::diamonds::Diamonds;
use crate::graph_compiler::passes::dot::{
    canonicalize::CanonicalizeDots, split_k::SplitK, strength::DotStrength,
};
use crate::graph_compiler::passes::fusion::Fusion;
use crate::graph_compiler::passes::linear_cross_entropy::LinearCrossEntropy;
use crate::graph_compiler::passes::linear_reordering::LinearReordering;
use crate::graph_compiler::passes::merge_dots::{MergeDots, MergeDotsBackward, Target};
use crate::graph_compiler::passes::place_concats::{Concats, PlaceConcats};
use crate::graph_compiler::passes::place_views::PlaceViews;
use crate::graph_compiler::passes::preserve_precision::PreservePrecision;
use crate::graph_compiler::passes::simplify::Simplify;
use crate::graph_compiler::passes::swap_pairs::SwapPairs;
use crate::graph_compiler::passes::take_along_axis::TakeAlongAxis;
use crate::graph_compiler::passes::unsupported::Unsupported;
use crate::graph_compiler::passes::updates::Updates;

impl Passes for Mps {
    /// MPS's passes, in order. Its backward dot merge also needs `contraction_epilogues`: the merged gradients are
    /// sliced in a GEMM epilogue. Merged dots pack for the GEMM epilogue's row columns, their slices read in place
    /// by what MPS fusion fuses. Update chains of 2 to 29 fuse: Metal binds 31 buffers, two of them the operand and
    /// output.
    fn passes<'a>(config: &CompilerConfig) -> Vec<Option<Pass<DeviceModule<'a, Mps>>>> {
        let fuse = config.fuse;
        let target = Target {
            reads: crate::codegen::fusion::mps::fusible,
            views: true,
            lanes: 1,
            columns: crate::codegen::regions::ROW_COLUMNS,
        };
        let row_tile = crate::codegen::dot_general::mps::row::ROW_TILE;

        vec![
            Self::pass::<PreservePrecision>(true, config),
            Self::add(true, Cse),
            Self::add(true, Simplify { matched: false }),
            Self::pass::<Unsupported>(true, config),
            Self::pass::<TakeAlongAxis>(true, config),
            Self::pass::<ChunkedCrossEntropy>(true, config),
            Self::pass::<MergeDotsBackward>(fuse && config.contraction_epilogues, config),
            Self::add(fuse && config.merge_dots, MergeDots { target }),
            Self::pass::<FlashAttention>(fuse, config),
            Self::pass::<FlashAttentionBackward>(fuse, config),
            Self::pass::<LinearCrossEntropy>(true, config),
            Self::add(true, Simplify { matched: true }),
            Self::pass::<SwapPairs>(true, config),
            Self::pass::<DotStrength>(true, config),
            Self::pass::<NeuralEngine>(true, config),
            Self::add(
                config.linear_reordering,
                LinearReordering {
                    tile: Some(row_tile),
                },
            ),
            Self::pass::<SplitK>(true, config),
            Self::pass::<CanonicalizeDots>(true, config),
            Self::add(fuse, Updates { least: 2, most: 29 }),
            Self::pass::<Diamonds>(true, config),
            Self::pass::<PlaceConcats>(fuse, config),
            Self::pass::<Fusion>(fuse, config),
            Self::pass::<PlaceViews>(true, config),
            Self::pass::<Alone>(true, config),
            Self::pass::<Concats>(true, config),
        ]
    }
}
