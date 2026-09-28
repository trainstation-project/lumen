//! The MPS copy kernels.
//!
//! MPS buffers are `MTLBuffer`s in Metal's Shared storage mode: on Apple
//! Silicon's unified memory `buffer.contents` is an ordinary CPU pointer, so
//! an ordinary `memcpy` moves bytes as efficiently as anything Metal offers
//! for a host-side range (PyTorch's MPS `copy_` likewise touches
//! `buffer.contents` directly for host transfers). There is nothing
//! device-specific to call, so both directions are the CPU kernel — this
//! module exists so the dispatch is explicit per device, and so a future
//! MPS-specific path has a home.

pub(super) use super::cpu::{copy_d2h, copy_h2d};
