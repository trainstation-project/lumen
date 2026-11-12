//! What TPU runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::dot::fold_reshapes::FoldDotReshapes;
use crate::graph_compiler::pipeline::Pass;

/// Always.
impl DevicePass<Tpu> for FoldDotReshapes {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Tpu>> {
        Box::new(FoldDotReshapes)
    }
}
