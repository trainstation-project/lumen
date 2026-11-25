//! TPU's passes: every pass whether or not `fuse` is set (its kernels are its fusions), attention and matmuls fitted
//! to its tiled layout, dots merged once the graph is simplified, rows and matmuls fused into its own kernels.

use super::{Passes, Pipeline};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Device, DeviceModule, Tpu};
use crate::graph_compiler::passes::alone::Alone;
use crate::graph_compiler::passes::attention::layouts::AttentionLayouts;
use crate::graph_compiler::passes::attention::{FlashAttention, FlashAttentionBackward};
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
use crate::graph_compiler::passes::simplify::Simplify;
use crate::graph_compiler::passes::sink_reshapes::SinkReshapes;
use crate::graph_compiler::passes::swap_pairs::SwapPairs;
use crate::graph_compiler::passes::unsupported::Unsupported;
use crate::graph_compiler::passes::updates::Updates;

impl Passes for Tpu {
    /// TPU's passes, in order. Forward dots merge late, once the graph is simplified, packing whole vector lanes
    /// (everything read in place, no views, one column). Update chains of 1 or more fuse: even one update runs as an
    /// `updates` kernel.
    fn passes<'a>(config: &CompilerConfig) -> Pipeline<DeviceModule<'a, Tpu>> {
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
        let mut pipeline = Pipeline::new(Tpu::NAME);

        pipeline
            .pass::<Unsupported>(config)
            .pass::<MergeDotsBackward>(config)
            .pass::<FlashAttention>(config)
            .pass::<FlashAttentionBackward>(config)
            .pass::<AttentionLayouts>(config)
            .pass::<FoldDotReshapes>(config)
            .add(Simplify { matched: true })
            .pass::<SwapPairs>(config)
            .pass::<SinkReshapes>(config)
            .pass::<DotStrength>(config)
            .maybe(MergeDots::configured_with(config, target))
            .pass::<CanonicalizeDots>(config)
            .pass::<FusedDiamonds>(config)
            .pass::<DotRows>(config)
            .add(updates)
            .pass::<FuseDots>(config)
            .pass::<PlaceConcats>(config)
            .pass::<Fusion>(config)
            .pass::<PlaceViews>(config)
            .pass::<Alone>(config)
            .pass::<Concats>(config)
            .pass::<HorizontalFusion>(config);

        pipeline
    }
}
