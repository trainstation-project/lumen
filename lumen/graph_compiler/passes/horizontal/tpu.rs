//! What TPU runs for this pass.

use super::HorizontalFusionDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::fusion::tpu as fusion;
use crate::graph_compiler::passes::horizontal::{Grouping, HorizontalFusion};
use crate::graph_compiler::pipeline::{Pass, when};

/// TPU's answers to the pass.
impl HorizontalFusionDevice for Tpu {
    /// With `horizontal_fusion`.
    fn horizontal_fusion<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        let horizontal = HorizontalFusion {
            candidate: fusion::horizontal_candidate,
            grouping: Grouping::Shape,
            buffers: fusion::HORIZONTAL_BUFFERS,
            name: "elementwise",
        };

        when(config.horizontal_fusion, horizontal)
    }
}
