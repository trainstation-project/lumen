//! What CUDA (CuTe DSL) runs for this pass.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cuda, DeviceModule};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::attention::FlashAttention;
use crate::graph_compiler::pipeline::{Pass, when};

/// With `fuse` and `flash_attention`: the forwards its SM80 kernel runs (`codegen::attention::cuda::supports`).
impl DevicePass<Cuda> for FlashAttention {
    fn pass<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Cuda>> {
        when(config.fuse && config.flash_attention, FlashAttention)
    }
}
