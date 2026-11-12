//! What Triton runs for this pass.

use super::HorizontalFusionDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::fusion::triton as fusion;
use crate::graph_compiler::passes::horizontal::{Grouping, HorizontalFusion};
use crate::graph_compiler::pipeline::{Pass, when};

/// Triton's answers to the pass.
impl HorizontalFusionDevice for Triton {
    /// With `fuse` and `horizontal_fusion`.
    fn horizontal_fusion<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        let horizontal = HorizontalFusion {
            candidate: fusion::horizontal_candidate,
            grouping: Grouping::Tiling,
            buffers: fusion::HORIZONTAL_BUFFERS,
            name: "elementwise",
        };

        when(config.fuse && config.horizontal_fusion, horizontal)
    }
}
