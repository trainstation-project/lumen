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
    fn passes<'a>(config: &CompilerConfig) -> Vec<Pass<DeviceModule<'a, Mps>>> {
        let fuse = config.fuse;
        let target = Target {
            reads: crate::codegen::fusion::mps::fusible,
            views: true,
            lanes: 1,
            columns: crate::codegen::regions::ROW_COLUMNS,
        };
        let row_tile = crate::codegen::dot_general::mps::row::ROW_TILE;

        let mut passes: Vec<Pass<DeviceModule<'a, Mps>>> = Vec::new();

        if config.preserve_precision {
            passes.push(Box::new(PreservePrecision));
        }

        passes.push(Box::new(Cse));
        passes.push(Box::new(Simplify { matched: false }));
        passes.push(Box::new(Unsupported::of::<Self>()));
        passes.push(Box::new(TakeAlongAxis));

        if fuse && config.fused_linear_cross_entropy {
            passes.push(Box::new(ChunkedCrossEntropy));
        }

        if fuse && config.contraction_epilogues && config.merge_dots {
            passes.push(Box::new(MergeDotsBackward));
        }

        if fuse && config.merge_dots {
            passes.push(Box::new(MergeDots { target }));
        }

        if fuse && config.flash_attention {
            passes.push(Box::new(FlashAttention));
            passes.push(Box::new(FlashAttentionBackward));
        }

        if fuse && config.fused_linear_cross_entropy {
            passes.push(Box::new(LinearCrossEntropy));
        }

        passes.push(Box::new(Simplify { matched: true }));
        passes.push(Box::new(SwapPairs));
        passes.push(Box::new(DotStrength));

        if config.neural_engine {
            passes.push(Box::new(NeuralEngine));
        }

        if config.linear_reordering {
            passes.push(Box::new(LinearReordering {
                tile: Some(row_tile),
            }));
        }

        if config.split_k {
            passes.push(Box::new(SplitK));
        }

        passes.push(Box::new(CanonicalizeDots::of::<Self>()));

        if fuse {
            passes.push(Box::new(Updates { least: 2, most: 29 }));
        }

        if fuse && config.normalization_diamonds {
            passes.push(Box::new(Diamonds::of::<Self>()));
        }

        if fuse {
            passes.push(Box::new(PlaceConcats));
            passes.push(Box::new(Fusion));
        }

        passes.push(Box::new(PlaceViews::of::<Self>()));
        passes.push(Box::new(Alone));
        passes.push(Box::new(Concats));

        passes
    }
}
