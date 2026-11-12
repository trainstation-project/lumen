//! What Triton runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::fusion::triton as fusion;
use crate::graph_compiler::passes::merge_dots::{MergeDots, MergeDotsBackward, Target};
use crate::graph_compiler::pipeline::{Pass, when};

/// With `fuse` and `merge_dots`.
impl DevicePass<Triton> for MergeDotsBackward {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
        when(config.fuse && config.merge_dots, MergeDotsBackward)
    }
}

/// With `fuse` and `merge_dots`.
impl DevicePass<Triton> for MergeDots {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
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
