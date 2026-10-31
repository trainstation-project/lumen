//! Scan kernels (cumulative sums along one axis): Metal's a fixed kernel split into chunks, Triton's
//! generated for each shape.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod triton;

use crate::DType;
use crate::graph::TensorType;

/// A backend's scan code generator: the code computing an inclusive cumulative sum. Backends that run a
/// fixed kernel for it instead fail on it.
pub(crate) trait ScanCodegen {
    /// The generated code: a Triton kernel's name and source with its lanes.
    type Code;

    /// Code for an inclusive cumulative sum of `x` along `axis` accumulating in `accum_dtype`, walking the
    /// axis backwards when `reverse`.
    fn scan(
        x: &TensorType,
        axis: usize,
        reverse: bool,
        accum_dtype: DType,
    ) -> Result<Self::Code, String>;
}
