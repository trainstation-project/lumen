//! The runtime: what runs on devices while a program executes: op dispatch, each backend's kernel launcher, events,
//! streams, device transfers and waits.

pub(crate) mod device_transfer;
pub(crate) mod dispatch;
pub(crate) mod event;
pub(crate) mod kernel;
pub mod stream;
pub(crate) mod wait;
