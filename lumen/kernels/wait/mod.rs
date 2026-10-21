use crate::Tensor;
use crate::device::Device;
use crate::kernels::{DispatchKey, Op};

pub type WaitKernel = fn(&Tensor);
pub static WAIT: Op<WaitKernel> = Op::new(op_name!("wait"), kernels);

fn kernels(key: DispatchKey) -> Option<WaitKernel> {
    match key {
        DispatchKey::Cpu | DispatchKey::Meta => Some(|_| {}),
        #[cfg(lumen_tpu_linked)]
        DispatchKey::Tpu => Some(|t| {
            let crate::Device::Tpu(index) = t.device() else {
                unreachable!("a TPU tensor")
            };
            crate::allocator::tpu::wait(index).unwrap_or_else(|e| panic!("{e}"))
        }),
        #[cfg(not(lumen_tpu_linked))]
        DispatchKey::Tpu => Some(|_| {}),
        DispatchKey::Mps | DispatchKey::Cuda => Some(|t| {
            let stream = crate::stream::stream_for(t.device());
            stream.expect("a tensor's device is present").synchronize()
        }),
    }
}

#[cfg(lumen_mps_linked)]
pub(crate) fn encode(
    _step: &crate::graph_compiler::plan::Step,
    _inputs: &[*const u8],
    _output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    wait(&keep[0]);
    Ok(())
}

pub fn wait(t: &Tensor) {
    if t.device() == Device::Cpu(0) {
        return;
    }
    let mut op = crate::profiler::record_op(WAIT.name(), || vec![t.ty()]);
    op.outputs(|| vec![t.ty()]);
    WAIT.dispatch(t.device())(t);
}
