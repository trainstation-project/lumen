//! What TPU runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::fusion::tpu as fusion;
use crate::graph_compiler::passes::horizontal::{Grouping, HorizontalFusion};
use crate::graph_compiler::pipeline::{Pass, when};

/// With `horizontal_fusion`.
impl DevicePass<Tpu> for HorizontalFusion {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Tpu>> {
        let horizontal = HorizontalFusion {
            candidate: fusion::horizontal_candidate,
            grouping: Grouping::Shape,
            buffers: fusion::HORIZONTAL_BUFFERS,
            name: "elementwise",
        };

        when(config.horizontal_fusion, horizontal)
    }
}
