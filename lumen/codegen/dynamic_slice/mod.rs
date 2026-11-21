//! Dynamic-slice code generation and kernels: slices and updates at runtime or constant starts, each
//! backend's beside the others.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod tpu;
pub(crate) mod triton;

use crate::graph::{Graph, TensorType};

/// A backend whose `updates` kernel writes a chain of constant-offset `dynamic_update_slice`s at once, so the graph
/// compiler's `updates` pass fuses chains into it: the chain lengths one kernel takes, and whether the pass waits
/// for the `fuse` config flag.
pub(crate) trait UpdateChains {
    /// Fewest updates a fused chain holds.
    const LEAST: usize;
    /// Most updates one kernel takes (the buffers it can bind).
    const MOST: usize;
    /// Whether the pass runs only with `fuse` (else always).
    const WITH_FUSE: bool = true;
}

/// A backend's dynamic-update-slice code generator: the code writing updates into an operand at their
/// starts. Backends that run a fixed kernel for an op instead fail on it.
pub(crate) trait DynamicSliceCodegen {
    /// The generated code: a Metal kernel's name and source, a Triton `BODY` function's, or a TPU program's
    /// kernel, spec and scalars.
    type Code;
    /// What the code is generated for beyond the op: the tensors' layouts Triton's loads and stores are
    /// generated for, or `()`.
    type Options: ?Sized;

    /// Code for the `updates` fusion `body`: a chain of dynamic-update-slices at constant starts, each
    /// update (inputs `1..`) written into operand input `0`.
    fn updates(body: &Graph, options: &Self::Options) -> Result<Self::Code, String>;

    /// Code for one dynamic-update-slice of `update` into `operand` at runtime starts, the start inputs
    /// flagged in `by_value` passed as scalars.
    fn update(
        operand: &TensorType,
        update: &TensorType,
        by_value: &[bool],
        options: &Self::Options,
    ) -> Result<Self::Code, String>;
}
