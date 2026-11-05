//! Launching a Triton kernel: its compiled cubin, launched as any CUDA kernel.

use super::Launch;
use super::cuda::CudaLaunch;
use crate::codegen::utils::cuda::Cuda;
use crate::codegen::utils::triton::Triton;

/// Triton enqueues its compiled kernel on lumen's stream through CUDA's launch.
impl Launch for Triton {
    type Args<'a> = CudaLaunch<'a>;

    /// Launches the compiled kernel in `args` as a CUDA kernel.
    fn launch(args: CudaLaunch<'_>) -> Result<(), String> {
        Cuda::launch(args)
    }
}
