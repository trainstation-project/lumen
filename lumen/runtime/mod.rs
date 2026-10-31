//! The runtime: what runs on devices while a program executes: op dispatch, each device's kernel launcher,
//! device transfers and waits.

pub(crate) mod device_transfer;
pub(crate) mod dispatch;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
pub(crate) mod wait;
