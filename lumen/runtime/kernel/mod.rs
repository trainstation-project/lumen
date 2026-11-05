//! Launching compiled kernels: each backend hands one kernel, its buffers and arguments to its queue, without
//! waiting for it. Metal records it into the open command buffer, a TPU batches it into the program it runs on
//! flush, CUDA enqueues it on lumen's stream, and Triton launches its compiled kernels as CUDA's.

#[cfg(all(feature = "python", lumen_cuda_linked))]
pub(crate) mod cuda;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
#[cfg(all(feature = "python", lumen_cuda_linked))]
pub(crate) mod triton;

/// A backend's kernel launcher.
#[cfg_attr(
    not(any(
        lumen_mps_linked,
        lumen_tpu_linked,
        all(feature = "python", lumen_cuda_linked)
    )),
    allow(dead_code)
)]
pub(crate) trait Launch {
    /// One launch: the compiled kernel and what it runs on, in the device's form.
    type Args<'a>;
    /// Queues the kernel in `args` on the device.
    fn launch(args: Self::Args<'_>) -> Result<(), String>;
}
