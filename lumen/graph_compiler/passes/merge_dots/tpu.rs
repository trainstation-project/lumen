//! What TPU runs for this pass.

use crate::codegen::utils::tpu::LANES;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::merge_dots::{MergeDots, MergeDotsBackward, Target};
use crate::graph_compiler::pipeline::{Pass, when};

/// With `merge_dots`.
impl DevicePass<Tpu> for MergeDotsBackward {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Tpu>> {
        when(config.merge_dots, MergeDotsBackward)
    }
}

/// With `merge_dots`, packing whole vector lanes.
impl DevicePass<Tpu> for super::MergeDotsLate {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Tpu>> {
        let merge = MergeDots {
            target: Target {
                reads: |_, _| true,
                views: false,
                lanes: LANES,
                columns: 1,
            },
        };

        when(config.merge_dots, merge)
    }
}
