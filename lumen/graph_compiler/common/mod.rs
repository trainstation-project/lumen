//! Device-independent graph passes shared by the backend compilers.

#[cfg_attr(not(all(feature = "python", lumen_mps_linked)), allow(dead_code))]
pub(crate) mod attention;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod autotune;
pub(crate) mod cse;
pub(crate) mod diamonds;
pub(crate) mod dot_rows;
pub(crate) mod fold_dot_reshapes;
pub(crate) mod horizontal;
pub(crate) mod loops;
pub(crate) mod multi_output;
pub(crate) mod simplify;
pub(crate) mod split_k;

#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod chunked_cross_entropy;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod clamp_indices;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod dot_strength;
pub(crate) mod flash_attention;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod linear_cross_entropy;
pub(crate) mod merge_dots;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod merge_dots_backward;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod take_along_axis;
