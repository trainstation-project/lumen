//! Layout code generation and kernels: copies between strided and contiguous forms, reshapes of tiled
//! tensors, and transposes fused with elementwise ops.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
pub(crate) mod triton;

use std::collections::HashMap;

use crate::graph::{Graph, TensorType};

/// A backend's layout code generator: the code moving elements between layouts. Backends whose layout
/// ops run as views or fixed kernels instead fail on them.
pub(crate) trait LayoutCodegen {
    /// The generated code: a Metal kernel's name and source, or a TPU program's kernel and spec.
    type Code;

    /// Code for the fusion `body` whose output is computed elementwise from transposes, staged through
    /// tiles; the inputs flagged in `by_value` are passed as scalars and the outputs in `views` written at
    /// their (offset, strides).
    fn transpose(
        body: &Graph,
        by_value: &[bool],
        views: &HashMap<usize, (usize, Vec<usize>)>,
    ) -> Result<Self::Code, String>;

    /// Code for a reshape of `from` into `to`.
    #[cfg_attr(not(lumen_tpu_linked), allow(dead_code))]
    fn reshape(from: &TensorType, to: &TensorType) -> Result<Self::Code, String>;
}
