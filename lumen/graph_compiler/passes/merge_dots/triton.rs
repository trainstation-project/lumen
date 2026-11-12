//! What Triton runs for this pass.

use super::MergeDotsDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::fusion::triton as fusion;
use crate::graph_compiler::passes::merge_dots::{MergeDots, MergeDotsBackward, Target};
use crate::graph_compiler::pipeline::{Pass, when};

/// Triton's answers to the pass.
impl MergeDotsDevice for Triton {
    /// With `fuse` and `merge_dots`.
    fn merge_dots_backward<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        when(config.fuse && config.merge_dots, MergeDotsBackward)
    }

    /// With `fuse` and `merge_dots`.
    fn merge_dots<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        let merge = MergeDots {
            target: Target {
                reads: fusion::fusible,
                views: true,
                lanes: 1,
                columns: 8,
            },
        };

        when(config.fuse && config.merge_dots, merge)
    }
}
