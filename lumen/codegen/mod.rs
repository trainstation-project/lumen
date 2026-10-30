//! Code generators for each backend, plus GEMM epilogue region detection shared between them.
pub(crate) mod attention;
pub(crate) mod kinds;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod regions;
pub(crate) mod tpu;
pub(crate) mod triton;

use std::collections::HashMap;

use crate::graph::Graph;

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
