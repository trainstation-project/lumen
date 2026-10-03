//! softmax: `exp(x - max) / sum(exp(x - max))` along the last dimension,
//! as one primitive ([`crate::graph::Primitive::Softmax`]), which
//! `TracedTensor.softmax` traces to. On MPS one kernel, online softmax
//! (`mps.metal`, `mps.rs`); elsewhere the reference.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
