//! Triton's passes: MPS's GPU pass list (see `mps.rs`) without the Neural Engine or a canonical dot layout, plus
//! partial row reductions in GEMMs and horizontal fusion.

use super::{Passes, Pipeline};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Device, DeviceModule, Triton};
use crate::graph_compiler::passes::alone::Alone;
use crate::graph_compiler::passes::attention::{FlashAttention, FlashAttentionBackward};
use crate::graph_compiler::passes::chunked_cross_entropy::ChunkedCrossEntropy;
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
use crate::graph_compiler::passes::simplify::Simplify;
use crate::graph_compiler::passes::swap_pairs::SwapPairs;
use crate::graph_compiler::passes::take_along_axis::TakeAlongAxis;
use crate::graph_compiler::passes::unsupported::Unsupported;
use crate::graph_compiler::passes::updates::Updates;

impl Passes for Triton {
    /// Triton's passes, in order. Merged dots pack for 8-column GEMM tiles, their slices read in place by what
    /// Triton fusion fuses. Update chains of 2 or more fuse: a kernel binds any number of buffers.
    fn passes<'a>(config: &CompilerConfig) -> Pipeline<DeviceModule<'a, Triton>> {
        let fuse = config.fuse;
        let target = Target {
            reads: crate::codegen::fusion::triton::fusible,
            views: true,
            lanes: 1,
            columns: 8,
        };
        let updates = Updates {
            least: 2,
            most: usize::MAX,
        };
        let mut pipeline = Pipeline::new(Triton::NAME);

        pipeline
            .pass::<Unsupported>(config)
            .pass::<TakeAlongAxis>(config)
            .pass::<ChunkedCrossEntropy>(config)
            .pass_if::<MergeDotsBackward>(fuse, config)
            .maybe(MergeDots::configured_with(config, target).filter(|_| fuse))
            .pass_if::<FlashAttention>(fuse, config)
            .pass_if::<FlashAttentionBackward>(fuse, config)
            .pass::<LinearCrossEntropy>(config)
            .add(Simplify { matched: true })
            .pass::<SwapPairs>(config)
            .pass::<DotStrength>(config)
            .maybe(LinearReordering::configured_with(
                config,
                crate::codegen::dot_general::triton::ROW_TILE,
            ))
            .pass::<SplitK>(config)
            .maybe(fuse.then_some(updates))
            .pass::<Diamonds>(config)
            .pass_if::<PlaceConcats>(fuse, config)
            .pass_if::<Fusion>(fuse, config)
            .pass::<WideRows>(config)
            .pass::<PlaceViews>(config)
            .pass::<Alone>(config)
            .pass_if::<HorizontalFusion>(fuse, config);

        pipeline
    }
}
