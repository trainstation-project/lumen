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
            Self::pass::<PreservePrecision>(true, config),
            Self::add(Cse),
            Self::add(Simplify { matched: false }),
            Self::pass::<Unsupported>(true, config),
            Self::pass::<MergeDotsBackward>(true, config),
            Self::pass::<FlashAttention>(true, config),
            Self::pass::<FlashAttentionBackward>(true, config),
            Self::pass::<AttentionLayouts>(true, config),
            Self::pass::<FoldDotReshapes>(true, config),
            Self::add(Simplify { matched: true }),
            Self::pass::<SwapPairs>(true, config),
            Self::pass::<SinkReshapes>(true, config),
            Self::pass::<DotStrength>(true, config),
            Self::maybe(MergeDots::configured_with(config, target)),
            Self::pass::<CanonicalizeDots>(true, config),
            Self::pass::<FusedDiamonds>(true, config),
            Self::pass::<DotRows>(true, config),
            Self::add(updates),
            Self::pass::<FuseDots>(true, config),
            Self::pass::<PlaceConcats>(true, config),
            Self::pass::<Fusion>(true, config),
            Self::pass::<PlaceViews>(true, config),
            Self::pass::<Alone>(true, config),
            Self::pass::<Concats>(true, config),
            Self::pass::<HorizontalFusion>(true, config),
        ]
    }
}
