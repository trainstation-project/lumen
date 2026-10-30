//! GEMM code generation: the trait each backend implements, and each backend's generator.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
pub(crate) mod triton;

use crate::graph::{Graph, Node};

/// A backend's GEMM code generator: the code computing a GEMM fusion `body`, its `dot` with the elementwise
/// epilogue after it, the inputs flagged in `by_value` passed as scalars.
pub(crate) trait GemmCodegen {
    /// The generated code: a Metal kernel's name and source, a Triton kernel's prologues and epilogue, or a
    /// TPU kernel's spec.
    type Code;
    /// What the backend generates the GEMM from beyond its body: MPS's tile configs, Triton's operand and
    /// output layouts, TPU's output windows.
    type Options: ?Sized;
    /// Code for the GEMM fusion `body` computing `dot`; fails for one the backend has no kernel for.
    fn gemm(
        body: &Graph,
        dot: &Node,
        by_value: &[bool],
        options: &Self::Options,
    ) -> Result<Self::Code, String>;
}
