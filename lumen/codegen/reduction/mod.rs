//! Reduction code generation: the trait each backend implements, and each backend's generator.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod tpu;
pub(crate) mod triton;

use crate::graph::{Graph, Node};

/// A backend's reduction code generator: the code computing a reduction fusion `body`, its one reduction
/// `root` with the elementwise prologue feeding it and the epilogue after it, the inputs flagged in
/// `by_value` passed as scalars.
pub(crate) trait ReductionCodegen {
    /// The generated code: a Metal kernel's name and source, a Triton kernel with its launch shape, or a TPU
    /// kernel's name and spec.
    type Code;
    /// Code for the reduction fusion `body` reducing at `root`; fails for one the backend has no kernel for.
    fn reduction(body: &Graph, by_value: &[bool], root: &Node) -> Result<Self::Code, String>;
}
