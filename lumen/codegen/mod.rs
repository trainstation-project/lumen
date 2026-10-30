//! Code generators for each backend, plus GEMM epilogue region detection shared between them.
pub(crate) mod kinds;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod regions;
pub(crate) mod tpu;
pub(crate) mod triton;

use std::collections::HashMap;

use crate::DType;
use crate::graph::Graph;
use crate::graph_compiler::common::attention::{Attention, Backward, Dropout, Score, Strides};

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
    /// Code applying the traced score ops `scores` (rounding casts and scalings) to the float32 `Q K^T`
    /// products, for the forward and backward code to call.
    fn score(scores: &[Score]) -> String;
    /// Code applying dropout `d` to each head's `sq` x `sk` probabilities, computed in `dtype`, for the
    /// forward and backward code to call. Fails on a backend whose kernels have no dropout yet.
    fn dropout(d: &Dropout, dtype: DType, sq: usize, sk: usize) -> Result<String, String>;
}

/// A backend's loop-fusion code generator: the code computing every output of the elementwise `body`, the
/// inputs flagged in `by_value` passed as scalars and output `k` written at `views[k]` (its element offset
/// and strides) when it has one.
pub(crate) trait ElementwiseCodegen {
    /// The generated code: a Metal kernel's name and source, a Triton kernel's, or a TPU program's kernel
    /// name and spec.
    type Code;
    /// How the loop's elements are split across threads when the autotuner chooses it: MPS's `Tuning`; `()`
    /// where it is chosen elsewhere (Triton at launch, TPU from the shapes).
    type Schedule;
    /// Code for the loop `body`; fails for a loop the backend has no kernel for.
    fn elementwise(
        body: &Graph,
        by_value: &[bool],
        views: &HashMap<usize, (usize, Vec<usize>)>,
        schedule: &Self::Schedule,
    ) -> Result<Self::Code, String>;
}
