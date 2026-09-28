//! Device streams (PyTorch: `c10::Stream`): queues that ops submit device
//! work to without waiting for it.

#[cfg(lumen_mps_linked)]
pub mod mps;
