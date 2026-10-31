//! Scan kernels (cumulative sums along one axis): Metal's a fixed kernel split into chunks, Triton's
//! generated for each shape.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod triton;
