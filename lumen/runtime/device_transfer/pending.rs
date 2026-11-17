//! Asynchronous device-to-host copies for staged plans: `to_host` starts the copy and returns the host tensor at
//! once, keyed by its address until `wait` (a host stage's `device_wait` step) blocks for it. On CUDA the copy is
//! a `cudaMemcpyAsync` into pinned memory with an event to wait on; elsewhere it runs at the wait.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use crate::device::Device;
use crate::{Tensor, TensorOptions};

/// A started copy to the host, until its wait.
enum Pending {
    /// A CUDA copy in flight: the event recorded after it on the default stream, with the source and destination
    /// kept alive until it completes.
    #[cfg(lumen_cuda_linked)]
    Event {
        device: usize,
        event: usize,
        _keep: [Tensor; 2],
    },
    /// A copy that runs when waited for.
    Deferred(Box<dyn FnOnce() + Send>),
}

/// Copies started and not yet waited for, by the host address they write.
static PENDING: Mutex<Option<HashMap<usize, Pending>>> = Mutex::new(None);

/// Starts copying device tensor `src` to the host and returns the host tensor, which must not be read before
/// `wait(its address)`: a CUDA copy is in flight into pinned memory; another device's runs at the wait.
pub(crate) fn to_host(src: &Tensor) -> Tensor {
    if src.device() == Device::Cpu(0) || src.numel() == 0 {
        return src.to(Device::Cpu(0));
    }

    let (dst, pending) = start(src);

    PENDING
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(HashMap::new)
        .insert(dst.data_ptr().addr(), pending);

    dst
}

/// The host tensor and the pending copy into it, for device tensor `src`.
fn start(src: &Tensor) -> (Tensor, Pending) {
    #[cfg(lumen_cuda_linked)]
    if let Device::Cuda(index) = src.device()
        && src.is_contiguous()
    {
        let pinned = crate::allocator::pinned::pinned();
        let dst = unsafe { Tensor::empty_with(src.shape(), src.dtype(), Device::Cpu(0), pinned) };
        let event = super::cuda::copy_d2h_async(&dst, src);
        let keep = [src.clone(), dst.clone()];

        return match event {
            Some(event) => (
                dst,
                Pending::Event {
                    device: index,
                    event,
                    _keep: keep,
                },
            ),
            None => (dst, Pending::Deferred(Box::new(|| {}))),
        };
    }

    let options = TensorOptions::new()
        .dtype(src.dtype())
        .device(Device::Cpu(0));
    let dst = unsafe { Tensor::empty(src.shape(), options) };
    let (from, to) = (src.clone(), dst.clone());

    let copy = move || {
        to.copy_(&from.to(Device::Cpu(0)))
            .expect("a host copy of one shape");
    };

    (dst, Pending::Deferred(Box::new(copy)))
}

/// Blocks until the copy writing the host buffer at `addr` (if one is pending) has finished.
pub(crate) fn wait(addr: usize) {
    let pending = PENDING
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_mut()
        .and_then(|p| p.remove(&addr));

    match pending {
        #[cfg(lumen_cuda_linked)]
        Some(Pending::Event { device, event, .. }) => super::cuda::wait_event(device, event),
        Some(Pending::Deferred(copy)) => copy(),
        None => {}
    }
}

/// Blocks until the copies writing `tensors` (those still pending) have finished.
pub(crate) fn wait_all<'a>(tensors: impl IntoIterator<Item = &'a Tensor>) {
    for t in tensors {
        wait(t.data_ptr().addr());
    }
}
