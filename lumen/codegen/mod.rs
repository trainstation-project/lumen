//! Code generators for each backend, plus GEMM epilogue region detection shared between them.
pub(crate) mod attention;
pub(crate) mod dot_general;
pub(crate) mod dynamic_slice;
pub(crate) mod elementwise;
pub(crate) mod emitter;
pub(crate) mod gather;
pub(crate) mod kinds;
pub(crate) mod layout;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod random;
pub(crate) mod reduction;
pub(crate) mod regions;
pub(crate) mod row;
pub(crate) mod scatter_add;
pub(crate) mod store;
pub(crate) mod triton;
pub(crate) mod utils;
