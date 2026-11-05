//! Launching Triton kernels: loop kernels through the launcher Python installs (which may launch them straight from
//! Rust, `cuda::driver`), and a compiled kernel as any CUDA kernel. Each op's own launch (a GEMM's, flash
//! attention's, a scatter-add's) is its codegen's.

use std::any::Any;
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(all(feature = "python", lumen_cuda_linked))]
use super::Launch;
#[cfg(all(feature = "python", lumen_cuda_linked))]
use super::cuda::CudaLaunch;
use crate::Tensor;
use crate::codegen::elementwise::triton::LoopConfig;
use crate::codegen::triton::Kernel;
#[cfg(all(feature = "python", lumen_cuda_linked))]
use crate::codegen::utils::cuda::Cuda;
#[cfg(all(feature = "python", lumen_cuda_linked))]
use crate::codegen::utils::triton::Triton;

#[cfg(all(feature = "python", lumen_cuda_linked))]
/// Triton enqueues its compiled kernel on lumen's stream through CUDA's launch.
impl Launch for Triton {
    type Args<'a> = CudaLaunch<'a>;

    /// Launches the compiled kernel in `args` as a CUDA kernel.
    fn launch(args: CudaLaunch<'_>) -> Result<(), String> {
        Cuda::launch(args)
    }
}

/// A jit helper bound to a kernel's constexpr parameter.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Function {
    /// The kernel parameter it is passed as (e.g. `LOAD0`, `STORE`).
    pub param: String,
    pub name: String,
    pub source: String,
}

impl Function {
    /// Binds the generated `(name, source)` to parameter `param`.
    pub(crate) fn new(param: impl Into<String>, (name, source): (String, String)) -> Self {
        Function {
            param: param.into(),
            name,
            source,
        }
    }
}

/// The kernels a step's launches were compiled for, kept with its generated helpers: the launcher finds
/// a run's kernel here by its config and argument classes instead of building a key each run.
#[derive(Default)]
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub struct Launches(pub(crate) Mutex<Vec<Arc<dyn Any + Send + Sync>>>);

/// Launches a loop kernel over `n` outputs on `programs` programs with the given helper functions,
/// finding or recording its compiled kernel in `launches`.
pub type Loop = fn(
    kernel: &Kernel,
    config: &LoopConfig,
    functions: &[Function],
    launches: &Launches,
    inputs: &[Tensor],
    out: &Tensor,
    n: usize,
    programs: usize,
) -> Result<(), String>;

/// The loop launcher installed from Python.
static LOOP: OnceLock<Loop> = OnceLock::new();

/// Installs the loop launcher; later calls are ignored.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_loop(launch: Loop) {
    let _ = LOOP.set(launch);
}

/// Launches `kernel` over `n` outputs, with enough programs for the block size, lanes and splits; its
/// compiled kernels are kept in `launches`.
pub(crate) fn launch(
    kernel: &Kernel,
    config: &LoopConfig,
    functions: &[Function],
    launches: &Launches,
    inputs: &[Tensor],
    out: &Tensor,
    n: usize,
) -> Result<(), String> {
    let launch = LOOP.get().ok_or(
        "the Triton backend's loops launch from Python (lumen.codegen.utils.triton): none is installed",
    )?;

    let programs = match kernel.segments.as_slice() {
        [] => n.div_ceil(config.block_size / kernel.lanes) * kernel.splits,
        segments => segments.iter().map(|s| s.div_ceil(config.block_size)).sum(),
    };

    launch(
        kernel, config, functions, launches, inputs, out, n, programs,
    )
}
