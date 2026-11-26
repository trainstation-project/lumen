//! TPU's passes: every pass whether or not `fuse` is set (its kernels are its fusions), attention and matmuls fitted
//! to its tiled layout, dots merged once the graph is simplified, rows and matmuls fused into its own kernels.

use super::{Pass, Passes};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::alone::Alone;
use crate::graph_compiler::passes::attention::layouts::AttentionLayouts;
use crate::graph_compiler::passes::attention::{FlashAttention, FlashAttentionBackward};
use crate::graph_compiler::passes::cse::Cse;
use crate::graph_compiler::passes::diamonds::FusedDiamonds;
use crate::graph_compiler::passes::dot::{
    canonicalize::CanonicalizeDots, fold_reshapes::FoldDotReshapes, fuse::FuseDots, rows::DotRows,
    strength::DotStrength,
};
use crate::graph_compiler::passes::fusion::Fusion;
use crate::graph_compiler::passes::gather_rows::GatherRows;
use crate::graph_compiler::passes::horizontal::HorizontalFusion;
use crate::graph_compiler::passes::merge_dots::{MergeDots, MergeDotsBackward, Target};
use crate::graph_compiler::passes::place_concats::{Concats, PlaceConcats};
use crate::graph_compiler::passes::place_views::PlaceViews;
use crate::graph_compiler::passes::preserve_precision::PreservePrecision;
use crate::graph_compiler::passes::simplify::Simplify;
use crate::graph_compiler::passes::sink_reshapes::SinkReshapes;
use crate::graph_compiler::passes::swap_pairs::SwapPairs;
use crate::graph_compiler::passes::unsupported::Unsupported;
use crate::graph_compiler::passes::updates::Updates;

impl Passes for Tpu {
    /// TPU's passes, in order. Forward dots merge late, once the graph is simplified, packing whole vector lanes
    /// (everything read in place, no views, one column). Update chains of 1 or more fuse: even one update runs as an
    /// `updates` kernel.
    fn passes<'a>(config: &CompilerConfig) -> Vec<Pass<DeviceModule<'a, Tpu>>> {
        let target = Target {
            reads: |_, _| true,
            views: false,
            lanes: crate::codegen::devices::tpu::LANES,
            columns: 1,
        };
        let updates = Updates {
            least: 1,
            most: usize::MAX,
        };

        let mut passes: Vec<Pass<DeviceModule<'a, Tpu>>> = Vec::new();

        if config.preserve_precision {
            passes.push(Box::new(PreservePrecision));
        }

        passes.push(Box::new(Cse));
        passes.push(Box::new(Simplify { matched: false }));
        passes.push(Box::new(Unsupported::of::<Self>()));

        if config.merge_dots {
            passes.push(Box::new(MergeDotsBackward));
        }

        if config.flash_attention {
            passes.push(Box::new(FlashAttention));
            passes.push(Box::new(FlashAttentionBackward));
            passes.push(Box::new(AttentionLayouts));
        }

        passes.push(Box::new(FoldDotReshapes));
        passes.push(Box::new(Simplify { matched: true }));
        passes.push(Box::new(SwapPairs));
        passes.push(Box::new(SinkReshapes));
        passes.push(Box::new(DotStrength));

        if config.merge_dots {
            passes.push(Box::new(MergeDots { target }));
        }

        passes.push(Box::new(CanonicalizeDots::of::<Self>()));

        if config.normalization_diamonds {
            passes.push(Box::new(FusedDiamonds::of::<Self>()));
            passes.push(Box::new(DotRows));
            passes.push(Box::new(GatherRows {
                fits: crate::codegen::row::tpu::fits,
            }));
        }

        passes.push(Box::new(updates));
        passes.push(Box::new(FuseDots));
        passes.push(Box::new(PlaceConcats));
        passes.push(Box::new(Fusion));
        passes.push(Box::new(PlaceViews::of::<Self>()));
        passes.push(Box::new(Alone));
        passes.push(Box::new(Concats));

        if config.horizontal_fusion {
            passes.push(Box::new(HorizontalFusion::of::<Self>()));
        }

        passes
    }
}
