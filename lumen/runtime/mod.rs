//! The runtime: what runs on devices while a program executes: op dispatch, each device's kernel launcher, events,
//! device transfers and waits.

pub(crate) mod cuda;
pub(crate) mod device_transfer;
pub(crate) mod dispatch;
pub(crate) mod event;
pub(crate) mod kernel;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub mod stream;
pub(crate) mod triton;
pub(crate) mod wait;
