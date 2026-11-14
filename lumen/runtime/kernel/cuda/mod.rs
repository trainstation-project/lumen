//! Launching CUDA kernels: a compiled kernel (a Triton or CuTe cubin) enqueued with `cuLaunchKernel` straight from
//! Rust, through the driver entry points `driver` loads; and the CuTe launcher and GEMM Python installs.

#[cfg(all(feature = "python", lumen_cuda_linked))]
pub(crate) mod driver;

#[cfg(all(feature = "python", lumen_cuda_linked))]
use std::ffi::{CStr, c_char, c_uint, c_void};
use std::sync::OnceLock;

#[cfg(all(feature = "python", lumen_cuda_linked))]
use super::Launch;
use crate::Tensor;
use crate::codegen::dot_general::cuda::Config;
#[cfg(all(feature = "python", lumen_cuda_linked))]
use crate::codegen::utils::cuda::Cuda;
#[cfg(all(feature = "python", lumen_cuda_linked))]
use driver::{Compiled, PARAM_BYTES, Value, bytes, driver};

/// Launches kernel `name` of module `source` (compiling it on its first launch) on `tensors`, its
/// inputs then its `outputs` outputs, each a flat view of the elements it spans, over `n` elements.
pub type Launcher = fn(
    name: &str,
    source: &str,
    tensors: &[Tensor],
    outputs: usize,
    n: usize,
) -> Result<(), String>;

/// The launcher installed from Python (`lumen.codegen.utils.cuda`).
pub(crate) static LAUNCH: OnceLock<Launcher> = OnceLock::new();

/// Installs the launcher; later calls are ignored.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_launch(launch: Launcher) {
    let _ = LAUNCH.set(launch);
}

/// What a GEMM writes: its product, or the outputs of the epilogue fused after it.
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub enum Output<'a> {
    /// The product, into this (m, n) or batched (l, m, n) tensor.
    Product(&'a Tensor),
    /// The outputs of a GEMM fusion: its generated epilogue (`codegen::dot_general::cuda`), as its module's name
    /// and source, computes them from the product, rounded to `dtype`, reading the fusion's `inputs` and writing
    /// its `outputs`.
    Epilogue {
        code: &'a (String, String),
        dtype: crate::DType,
        inputs: &'a [Tensor],
        outputs: &'a [Tensor],
    },
}

/// Writes `a @ b`, tensors (m, k) and (k, n), or batched (l, m, k) and (l, k, n), each laid out as given, to `output`,
/// with the tile of `config` (the untuned one when `None`), compiling its GEMM on first use.
pub type Gemm =
    fn(a: &Tensor, b: &Tensor, output: Output<'_>, config: Option<&Config>) -> Result<(), String>;

/// The GEMM installed from Python (`lumen.codegen.dot_general.cuda.launch`).
pub(crate) static GEMM: OnceLock<Gemm> = OnceLock::new();

/// Installs the GEMM; later calls are ignored.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_gemm(gemm: Gemm) {
    let _ = GEMM.set(gemm);
}

/// The flash-attention forward installed from Python (`lumen.codegen.attention.cuda.launch`).
pub(crate) static ATTENTION: OnceLock<crate::codegen::attention::cuda::Launch> = OnceLock::new();

/// Installs the flash-attention forward; later calls are ignored.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_attention(attention: crate::codegen::attention::cuda::Launch) {
    let _ = ATTENTION.set(attention);
}

#[cfg(all(feature = "python", lumen_cuda_linked))]
/// One CUDA launch: a compiled kernel, its grid and its runtime parameters in order.
pub(crate) struct CudaLaunch<'a> {
    pub(crate) kernel: &'a Compiled,
    pub(crate) grid: [usize; 3],
    pub(crate) values: &'a [Value],
}

#[cfg(all(feature = "python", lumen_cuda_linked))]
/// CUDA enqueues a kernel on lumen's stream with `cuLaunchKernel`, straight from Rust.
impl Launch for Cuda {
    type Args<'a> = CudaLaunch<'a>;

    /// Launches its `kernel` on `grid` with `values` (its runtime parameters in order: scalars, or the structs
    /// of a CuTe kernel's `tensor` parameters) on lumen's stream (a CUDA graph's capture stream while it
    /// captures), then the two scratch pointers Triton appends, null since the kernel needs no scratch.
    fn launch(args: CudaLaunch<'_>) -> Result<(), String> {
        let CudaLaunch {
            kernel,
            grid,
            values,
        } = args;

        if kernel.types.len() != values.len() {
            return Err(format!(
                "a Triton kernel of {} runtime parameters launched with {} values",
                kernel.types.len(),
                values.len()
            ));
        }

        let mut storage: Vec<[u8; PARAM_BYTES]> = Vec::with_capacity(values.len() + 2);

        for (ty, &value) in kernel.types.iter().zip(values) {
            let mut slot = [0u8; PARAM_BYTES];

            match (ty.as_str(), value) {
                ("tensor", Value::Struct(raw, _)) => slot = raw,
                ("tensor", v) => return Err(format!("a tensor parameter got {v:?}")),
                (ty, v) => match bytes(ty, v)? {
                    Some(b) => slot[..8].copy_from_slice(&b),
                    None => continue,
                },
            }

            storage.push(slot);
        }

        storage.extend([[0u8; PARAM_BYTES]; 2]);
        let mut params: Vec<*mut c_void> =
            storage.iter_mut().map(|b| b.as_mut_ptr().cast()).collect();

        if grid.contains(&0) {
            return Ok(());
        }

        let driver = driver()?;
        let [x, y, z] = grid.map(|g| g as c_uint);

        let result = unsafe {
            (driver.launch)(
                kernel.function as *mut c_void,
                x,
                y,
                z,
                32 * kernel.warps,
                1,
                1,
                kernel.shared,
                crate::runtime::stream::cuda::launch_stream(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        match result {
            0 => Ok(()),
            e => {
                let mut message: *const c_char = std::ptr::null();

                unsafe { (driver.error)(e, &mut message) };

                let message = match message.is_null() {
                    true => "unknown".into(),
                    false => unsafe { CStr::from_ptr(message) }.to_string_lossy(),
                };

                Err(format!("cuLaunchKernel failed: {message} (error {e})"))
            }
        }
    }
}
