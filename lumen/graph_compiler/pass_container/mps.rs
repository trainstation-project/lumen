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
            Self::add(config.preserve_precision, PreservePrecision),
            Self::add(true, Cse),
            Self::add(true, Simplify { matched: false }),
            Self::add(true, Unsupported::of::<Self>()),
            Self::add(true, TakeAlongAxis),
            Self::add(
                fuse && config.fused_linear_cross_entropy,
                ChunkedCrossEntropy,
            ),
            Self::add(
                fuse && config.contraction_epilogues && config.merge_dots,
                MergeDotsBackward,
            ),
            Self::add(fuse && config.merge_dots, MergeDots { target }),
            Self::add(fuse && config.flash_attention, FlashAttention),
            Self::add(fuse && config.flash_attention, FlashAttentionBackward),
            Self::add(
                fuse && config.fused_linear_cross_entropy,
                LinearCrossEntropy,
            ),
            Self::add(true, Simplify { matched: true }),
            Self::add(true, SwapPairs),
            Self::add(true, DotStrength),
            Self::add(config.neural_engine, NeuralEngine),
            Self::add(
                config.linear_reordering,
                LinearReordering {
                    tile: Some(row_tile),
                },
            ),
            Self::add(config.split_k, SplitK),
            Self::add(true, CanonicalizeDots::of::<Self>()),
            Self::add(fuse, Updates { least: 2, most: 29 }),
            Self::add(
                fuse && config.normalization_diamonds,
                Diamonds::of::<Self>(),
            ),
            Self::add(fuse, PlaceConcats),
            Self::add(fuse, Fusion),
            Self::add(true, PlaceViews::of::<Self>()),
            Self::add(true, Alone),
            Self::add(true, Concats),
        ]
    }
}
