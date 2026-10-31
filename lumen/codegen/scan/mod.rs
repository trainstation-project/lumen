//! Scan kernels (cumulative sums along one axis), each backend's a fixed kernel: Metal's split into
//! chunks, Triton's tiled for each shape.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod triton;

use crate::DType;
use crate::graph::TensorType;

/// A backend's scan code generator: how its kernel computes an inclusive cumulative sum. Backends whose
/// fixed kernel needs nothing generated fail on it.
pub(crate) trait ScanCodegen {
    /// The generated code: the tiling of Triton's fixed `scan_kernel`.
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
