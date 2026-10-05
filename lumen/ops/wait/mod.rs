//! wait: an in-place op on a tensor (PyTorch: `wait_tensor`,
//! `torch.cuda.Event.synchronize`): the host blocks until the device work
//! producing or using the tensor's memory has finished, the tensor's value
//! then final for the host to read or write. Every op that needs a device
//! tensor's value on the host calls it first ([`Tensor::wait_`]): host
//! copies (`device_transfer`), host reads and writes, DLPack, safetensors.
//! Each device's kernel knows its own streams.

use crate::Tensor;
use crate::device::Device;
use crate::ops::{DispatchKey, Op};

pub type WaitKernel = fn(&Tensor);

/// The host waits for a tensor's device work, dispatched on its device.
pub static WAIT: Op<WaitKernel> = Op::new(op_name!("wait"), kernels);

/// `wait`'s static registry.
fn kernels(key: DispatchKey) -> Option<WaitKernel> {
    match key {
        // Host memory, or none: no device work to wait for.
        DispatchKey::Cpu | DispatchKey::Meta => Some(|_| {}),
        // The stream's work so far.
        #[cfg(lumen_mps_linked)]
        DispatchKey::Mps => Some(|_| crate::stream::mps::synchronize()),
        #[cfg(not(lumen_mps_linked))]
        DispatchKey::Mps => None,
        // The device's work.
        #[cfg(lumen_cuda_linked)]
        DispatchKey::Cuda => Some(|t| match t.device() {
            Device::Cuda(index) => crate::stream::cuda::synchronize(index),
            _ => unreachable!("a CUDA storage"),
        }),
        #[cfg(not(lumen_cuda_linked))]
        DispatchKey::Cuda => None,
    }
}

/// A plan's `wait` step: its operand, in `keep` (its value is the operand's
/// memory, the plan's in-place alias), waited for.
#[cfg(lumen_mps_linked)]
pub(crate) fn encode(
    _step: &crate::graph::plan::Step,
    _inputs: &[*const u8],
    _output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    wait(&keep[0]);
    Ok(())
}

/// Wait in place until the device work on `t`'s memory has finished
/// ([`Tensor::wait_`]): profiled as an op writing `t`.
pub fn wait(t: &Tensor) {
    if t.device() == Device::Cpu {
        return;
    }
    let mut op = crate::profiler::record_op(WAIT.name(), || vec![t.ty()]);
    op.outputs(|| vec![t.ty()]);
    WAIT.dispatch(t.device())(t);
}
