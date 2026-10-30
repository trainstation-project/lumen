//! Indexing code generation and kernels: gathers, dynamic slices and updates, and scatter-adds, each
//! backend's beside the others.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
pub(crate) mod triton;
