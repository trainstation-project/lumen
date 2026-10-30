//! Metal indexing: dynamic-slice and dynamic-update-slice kernels, scatter-add kernels, and the source of a
//! fused chain of dynamic-update-slices.

pub(crate) mod dynamic_slice;
pub(crate) mod scatter_add;
mod updates;

pub(crate) use updates::updates_kernel;
