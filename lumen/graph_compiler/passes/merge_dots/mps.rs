//! What MPS runs for this pass.

use super::MergeDotsDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::fusion::mps as fusion;
use crate::graph_compiler::passes::merge_dots::{MergeDots, MergeDotsBackward, Target};
use crate::graph_compiler::pipeline::{Pass, when};

/// MPS's answers to the pass.
impl MergeDotsDevice for Mps {
    /// With `fuse`, `merge_dots` and `contraction_epilogues`.
    fn merge_dots_backward<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        let on = config.fuse && config.merge_dots && config.contraction_epilogues;
        when(on, MergeDotsBackward)
    }

    /// With `fuse` and `merge_dots`, packing for the GEMM epilogue's row columns.
    fn merge_dots<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        let merge = MergeDots {
            target: Target {
                reads: fusion::fusible,
                views: true,
                lanes: 1,
                columns: crate::codegen::regions::ROW_COLUMNS,
            },
        };

        when(config.fuse && config.merge_dots, merge)
    }
}
