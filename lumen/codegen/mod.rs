//! Code generators for each backend, plus GEMM epilogue region detection shared between them.
pub(crate) mod kinds;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod regions;
pub(crate) mod tpu;
pub(crate) mod triton;

use crate::graph::Graph;
use crate::graph_compiler::common::attention::{Attention, Backward, Strides};

/// A backend's flash-attention code generator: the code its kernels run for a fused attention body,
/// given the attention matched in it and how the body's inputs and outputs are laid out.
pub(crate) trait AttentionCodegen {
    /// The generated code: a Metal kernel's name and source, Triton jit helpers by parameter, or a TPU
    /// kernel's kind and spec.
    type Code;
    /// Code for forward attention `a`, matched in `body` laid out as `strides`.
    fn forward(body: &Graph, a: &Attention, strides: &Strides) -> Result<Self::Code, String>;
    /// Code for attention backward `b`, matched in `body` laid out as `strides`.
    fn backward(body: &Graph, b: &Backward, strides: &Strides) -> Result<Self::Code, String>;
}
