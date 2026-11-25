//! MPS's passes: the GPU pass list Triton shares (dots merged, attention and cross entropies fused, rows and
//! epilogues fused, the fusing passes only with `fuse`), plus the Neural Engine and concatenations fusion could not
//! place.

use super::{Passes, Pipeline};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Device, DeviceModule, Mps};
use crate::graph_compiler::passes::alone::Alone;
use crate::graph_compiler::passes::apple_neural_engine::NeuralEngine;
use crate::graph_compiler::passes::attention::{FlashAttention, FlashAttentionBackward};
use crate::graph_compiler::passes::chunked_cross_entropy::ChunkedCrossEntropy;
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
    fn passes<'a>(config: &CompilerConfig) -> Pipeline<DeviceModule<'a, Mps>> {
        let fuse = config.fuse;
        let target = Target {
            reads: crate::codegen::fusion::mps::fusible,
            views: true,
            lanes: 1,
            columns: crate::codegen::regions::ROW_COLUMNS,
        };
        let row_tile = crate::codegen::dot_general::mps::row::ROW_TILE;
        let mut pipeline = Pipeline::new(Mps::NAME);

        pipeline
            .pass::<Unsupported>(config)
            .pass::<TakeAlongAxis>(config)
            .pass::<ChunkedCrossEntropy>(config)
            .pass_if::<MergeDotsBackward>(fuse && config.contraction_epilogues, config)
            .maybe(MergeDots::configured_with(config, target).filter(|_| fuse))
            .pass_if::<FlashAttention>(fuse, config)
            .pass_if::<FlashAttentionBackward>(fuse, config)
            .pass::<LinearCrossEntropy>(config)
            .add(Simplify { matched: true })
            .pass::<SwapPairs>(config)
            .pass::<DotStrength>(config)
            .pass::<NeuralEngine>(config)
            .maybe(LinearReordering::configured_with(config, row_tile))
            .pass::<SplitK>(config)
            .pass::<CanonicalizeDots>(config)
            .maybe(fuse.then_some(Updates { least: 2, most: 29 }))
            .pass::<Diamonds>(config)
            .pass_if::<PlaceConcats>(fuse, config)
            .pass_if::<Fusion>(fuse, config)
            .pass::<PlaceViews>(config)
            .pass::<Alone>(config)
            .pass::<Concats>(config);

        pipeline
    }
}
