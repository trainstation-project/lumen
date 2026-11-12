//! What Triton runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::fusion::triton as fusion;
use crate::graph_compiler::passes::horizontal::{Grouping, HorizontalFusion};
use crate::graph_compiler::pipeline::{Pass, when};

/// With `fuse` and `horizontal_fusion`.
impl DevicePass<Triton> for HorizontalFusion {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
        let horizontal = HorizontalFusion {
            candidate: fusion::horizontal_candidate,
            grouping: Grouping::Tiling,
            buffers: fusion::HORIZONTAL_BUFFERS,
            name: "elementwise",
        };

        when(config.fuse && config.horizontal_fusion, horizontal)
    }
}
