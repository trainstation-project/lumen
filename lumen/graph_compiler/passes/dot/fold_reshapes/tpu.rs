//! What TPU runs for this pass.

use super::FoldDotReshapesDevice;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::dot::fold_reshapes::FoldDotReshapes;
use crate::graph_compiler::pipeline::Pass;

/// TPU's answers to the pass.
impl FoldDotReshapesDevice for Tpu {
    /// Always.
    fn fold_dot_reshapes<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(FoldDotReshapes)
    }
}
