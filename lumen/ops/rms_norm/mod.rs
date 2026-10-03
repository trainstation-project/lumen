//! rms_norm: `x * rsqrt(mean(x^2) + epsilon)` over the last dimension,
//! times an optional weight, as one primitive
//! ([`crate::graph::Primitive::RmsNorm`], `lumen.rms_norm`). On MPS one
//! kernel (`mps.metal`, `mps.rs`); elsewhere the reference.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
