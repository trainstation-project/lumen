//! Indexing code generation and kernels: gathers, dynamic slices and updates, and scatter-adds, each
//! backend's beside the others.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
pub(crate) mod triton;

use crate::graph::{Graph, TensorType};

/// A backend's indexing code generator: the code writing indexed updates into an operand. Backends that
/// run a fixed kernel for an op instead fail on it.
pub(crate) trait IndexingCodegen {
    /// The generated code: a Metal kernel's name and source, a Triton kernel's name and source, or a TPU
    /// program's kernel, spec and scalars.
    type Code;

    /// Code for the `updates` fusion `body`: a chain of dynamic-update-slices at constant starts, each
    /// update (inputs `1..`) written into operand input `0`.
    fn updates(body: &Graph) -> Result<Self::Code, String>;

    /// Code for one dynamic-update-slice of `update` into `operand` at runtime starts, the start inputs
    /// flagged in `by_value` passed as scalars.
    fn update(
        operand: &TensorType,
        update: &TensorType,
        by_value: &[bool],
    ) -> Result<Self::Code, String>;

    /// Code for a scatter-add of `picks` rows into `operand` along `axis`.
    fn scatter_add(operand: &TensorType, axis: usize, picks: usize) -> Result<Self::Code, String>;
}
