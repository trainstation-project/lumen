//! Elementwise primitives (add, sub, mul, div, max, eq, lt, neg, exp, log,
//! rsqrt, tanh, logistic, convert_element_type, select): their kernels per
//! backend, `mps.metal` and `mps.rs` for graph plans on MPS (see
//! [`crate::ops::mps`]).

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
