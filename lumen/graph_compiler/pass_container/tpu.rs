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
    fn passes<'a>(config: &CompilerConfig) -> Vec<Option<Pass<DeviceModule<'a, Tpu>>>> {
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

        vec![
            Self::add(config.preserve_precision, PreservePrecision),
            Self::add(true, Cse),
            Self::add(true, Simplify { matched: false }),
            Self::add(true, Unsupported::of::<Self>()),
            Self::add(config.merge_dots, MergeDotsBackward),
            Self::add(config.flash_attention, FlashAttention),
            Self::add(config.flash_attention, FlashAttentionBackward),
            Self::add(config.flash_attention, AttentionLayouts),
            Self::add(true, FoldDotReshapes),
            Self::add(true, Simplify { matched: true }),
            Self::add(true, SwapPairs),
            Self::add(true, SinkReshapes),
            Self::add(true, DotStrength),
            Self::add(config.merge_dots, MergeDots { target }),
            Self::add(true, CanonicalizeDots::of::<Self>()),
            Self::add(config.normalization_diamonds, FusedDiamonds::of::<Self>()),
            Self::add(config.normalization_diamonds, DotRows),
            Self::add(true, updates),
            Self::add(true, FuseDots),
            Self::add(true, PlaceConcats),
            Self::add(true, Fusion),
            Self::add(true, PlaceViews::of::<Self>()),
            Self::add(true, Alone),
            Self::add(true, Concats),
            Self::add(config.horizontal_fusion, HorizontalFusion::of::<Self>()),
        ]
    }
}
