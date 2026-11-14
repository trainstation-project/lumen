//! What CUDA (CuTe DSL) runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cuda, DeviceModule};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::fusion::cuda as fusion;
use crate::graph_compiler::passes::merge_dots::{MergeDots, MergeDotsBackward, Target};
use crate::graph_compiler::pipeline::{Pass, when};

/// With `fuse` and `merge_dots`.
impl DevicePass<Cuda> for MergeDotsBackward {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Cuda>> {
        when(config.fuse && config.merge_dots, MergeDotsBackward)
    }
}

/// With `fuse` and `merge_dots`, packing for the GEMM epilogue's 16-byte vectors (8 columns of 16-bit products).
impl DevicePass<Cuda> for MergeDots {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Cuda>> {
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
