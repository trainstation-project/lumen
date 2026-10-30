//! Indexing code generation and kernels: gathers and scatter-adds, each backend's beside the others.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
pub(crate) mod triton;

use crate::graph::TensorType;

/// A backend's indexing code generator: the code adding indexed rows into an operand. Backends that
/// run a fixed kernel for an op instead fail on it.
pub(crate) trait IndexingCodegen {
    /// The generated code: a Metal kernel's name and source, a Triton kernel's name and source, or a TPU
    /// program's kernel, spec and scalars.
    type Code;

    /// Code for a scatter-add of `picks` rows into `operand` along `axis`.
    fn scatter_add(operand: &TensorType, axis: usize, picks: usize) -> Result<Self::Code, String>;
}
