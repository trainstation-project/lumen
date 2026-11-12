//! How the CUDA (CuTe DSL) backend fuses forward attention: its answers to the forward `FlashAttention` pass.

use crate::graph::Var;
use crate::graph_compiler::device::{Cuda, DeviceModule};
use crate::graph_compiler::passes::attention::FORWARD;

use super::{Attentions, Supports};

/// Flash-attention matching for CUDA: no epilogues, a static causal diagonal, and only the attentions its SM80
/// kernel runs (the rest stay unfused, and fail compiling as any op the backend has no kernel for).
impl Attentions for DeviceModule<'_, Cuda> {
    /// No: the kernel writes the attention alone.
    fn contraction_epilogues(&self) -> bool {
        false
    }

    /// None: the CUDA backend passes runtime scalars as 0-d device tensors.
    fn scalars(&self) -> Vec<Var> {
        Vec::new()
    }

    /// Every forward attention runs on the one forward kernel.
    fn kernel(&mut self) -> String {
        FORWARD.into()
    }

    /// No: the kernel's causal diagonal is fixed at compile time.
    fn runtime_offsets(&self) -> bool {
        false
    }

    /// The attentions `codegen::attention::cuda::supports` accepts.
    fn supports(&self) -> Supports {
        crate::codegen::attention::cuda::supports
    }
}
