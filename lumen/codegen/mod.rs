//! Code generators for each backend, plus GEMM epilogue region detection shared between them.
pub(crate) mod attention;
pub(crate) mod elementwise;
pub(crate) mod kinds;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod reduction;
pub(crate) mod regions;
pub(crate) mod tpu;
pub(crate) mod triton;
