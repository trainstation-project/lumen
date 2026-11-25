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
    fn passes<'a>(config: &CompilerConfig) -> Vec<Pass<DeviceModule<'a, Triton>>> {
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

        let mut passes: Vec<Pass<DeviceModule<'a, Triton>>> = Vec::new();

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

        if fuse && config.merge_dots {
            passes.push(Box::new(MergeDotsBackward));
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

        if config.linear_reordering {
            passes.push(Box::new(LinearReordering {
                tile: Some(row_tile),
            }));
        }

        if config.split_k {
            passes.push(Box::new(SplitK));
        }

        if fuse {
            passes.push(Box::new(updates));
        }

        if fuse && config.normalization_diamonds {
            passes.push(Box::new(Diamonds::of::<Self>()));
        }

        if fuse {
            passes.push(Box::new(PlaceConcats));
            passes.push(Box::new(Fusion));
        }

        if fuse && config.normalization_diamonds && config.linear_reordering {
            passes.push(Box::new(WideRows));
        }

        passes.push(Box::new(PlaceViews::of::<Self>()));
        passes.push(Box::new(Alone));

        if fuse && config.horizontal_fusion {
            passes.push(Box::new(HorizontalFusion::of::<Self>()));
        }

        passes
    }
}
