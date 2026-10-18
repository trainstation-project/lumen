//! TPU kernels: Pallas kernels (each `lumen/ops/<op>/tpu.py`, lowered by
//! `lumen/ops/tpu/pallas.py`) that libtpu compiles, run on a device's
//! memory: the TPU allocator's one buffer, which the program takes donated
//! and returns written (`allocator::tpu::run`). Rust chooses each kernel's
//! spec (shapes, blocks, dtypes); JAX lowers it, on its CPU platform, from
//! Python, so TPU kernels run only where lumen runs from Python (the
//! `python` feature, JAX installed: the `tpu` extra). Kernels address the
//! memory by row (512 bytes): each buffer they read or write starts at one,
//! at a row the run passes them as a scalar.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use crate::device::tpu::{error, ffi};
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;

/// The StableHLO program of Pallas module `module`'s kernel for `spec`
/// (JSON): its `program(spec)`, through `lumen.ops.tpu.pallas.lower`.
#[cfg(feature = "python")]
fn lower(module: &str, spec: &str) -> Result<String, String> {
    use pyo3::prelude::*;
    Python::attach(|py| {
        py.import("lumen.ops.tpu.pallas")?
            .call_method1("lower", (module, spec))?
            .extract::<String>()
    })
    .map_err(|e| e.to_string())
}

#[cfg(not(feature = "python"))]
fn lower(module: &str, _spec: &str) -> Result<String, String> {
    Err(format!(
        "{module}: TPU kernels are Pallas kernels, lowered by JAX: run lumen from Python, with \
         the `tpu` extra installed"
    ))
}

/// A compiled program (a PJRT executable, kept for the process).
#[derive(Clone, Copy)]
struct Executable(*mut c_void);

// SAFETY: a PJRT executable, which PJRT's API runs from any thread.
unsafe impl Send for Executable {}

/// Module `module`'s kernel for `spec`, lowered and compiled once per
/// process.
fn compiled(module: &str, spec: &str) -> Result<Executable, String> {
    static CACHE: OnceLock<Mutex<HashMap<String, Executable>>> = OnceLock::new();
    let key = format!("{module} {spec}");
    let cache = || {
        CACHE
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    };
    if let Some(&executable) = cache().get(&key) {
        return Ok(executable);
    }
    // Not holding the cache: lowering runs Python, which may run lumen.
    let code = lower(module, spec)?;
    // SAFETY: `code`'s bytes.
    let executable = unsafe { ffi::lumen_tpu_compile(code.as_ptr().cast(), code.len()) };
    if executable.is_null() {
        return Err(error());
    }
    cache().insert(key, Executable(executable));
    Ok(Executable(executable))
}

/// A kernel to launch: the Pallas module of its source (`module`), its
/// spec's fields (`spec`: JSON members, `"name": value, ...`; the device's
/// memory's `rows` are added), its scalars: the rows of `rows` (TPU buffers, each at the start of
/// a row) then `scalars`, and `host` bytes (whole rows) copied to the TPU
/// for it.
pub(crate) struct Launch<'a> {
    pub(crate) module: &'a str,
    pub(crate) spec: String,
    pub(crate) rows: &'a [*const u8],
    pub(crate) scalars: &'a [i32],
    pub(crate) host: &'a [u8],
}

/// Runs `launch` on device `index`'s memory, in order after the kernels
/// before it.
pub(crate) fn launch(index: usize, launch: Launch) -> Result<(), String> {
    let rows = crate::allocator::tpu::rows(index)?;
    let spec = match launch.spec.as_str() {
        "" => format!("{{\"rows\": {rows}}}"),
        fields => format!("{{{fields}, \"rows\": {rows}}}"),
    };
    let executable =
        compiled(launch.module, &spec).map_err(|e| format!("{}: {e}", launch.module))?;
    crate::allocator::tpu::run(
        index,
        executable.0,
        launch.rows,
        launch.scalars,
        launch.host,
    )
    .map_err(|e| format!("{}: {e}", launch.module))
}

/// Runs plan step `step` on device `index`: its operands at `inputs`, its
/// output at `output`.
pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    match step.primitive {
        Primitive::DotGeneral { .. } => {
            super::dot_general::tpu::encode(index, step, inputs, output)
        }
        _ => Err(format!("{} does not run on TPU yet", step.primitive.name())),
    }
}
