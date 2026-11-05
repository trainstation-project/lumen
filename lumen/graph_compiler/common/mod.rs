//! Device-independent graph passes shared by the backend compilers.

pub(crate) mod alone;
pub(crate) mod cse;
pub(crate) mod diamonds;
pub(crate) mod dot;
pub(crate) mod elementwise;
pub(crate) mod fusion;
pub(crate) mod horizontal;
pub(crate) mod linear_reordering;
pub(crate) mod merge_dots;
pub(crate) mod multi_output;
pub(crate) mod place_concats;
pub(crate) mod place_views;
pub(crate) mod preserve_precision;
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) mod rms_norm;
pub(crate) mod simplify;
pub(crate) mod updates;

#[cfg_attr(not(all(feature = "python", lumen_mps_linked)), allow(dead_code))]
pub(crate) mod attention;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod chunked_cross_entropy;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod clamp_indices;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod linear_cross_entropy;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod take_along_axis;
