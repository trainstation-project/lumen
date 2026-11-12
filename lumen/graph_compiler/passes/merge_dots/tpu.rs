//! What TPU runs for this pass.

use super::MergeDotsDevice;
use crate::codegen::utils::tpu::LANES;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::merge_dots::{MergeDots, MergeDotsBackward, Target};
use crate::graph_compiler::pipeline::{Pass, when};

/// TPU's answers to the pass.
impl MergeDotsDevice for Tpu {
    /// With `merge_dots`.
    fn merge_dots_backward<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        when(config.merge_dots, MergeDotsBackward)
    }

    /// With `merge_dots`, packing whole vector lanes.
    fn merge_dots_late<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
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
