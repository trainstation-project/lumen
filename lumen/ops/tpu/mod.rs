//! TPU kernels: lumen's Pallas kernels (each `lumen/ops/<op>/tpu.py`, on
//! `lumen.pallas`), lowered to Mosaic by lumen (`lumen/pallas`), each
//! wrapped in a one-op StableHLO program (a `tpu_custom_call`) that libtpu
//! compiles, run on a device's memory: the TPU allocator's one buffer,
//! which the program takes donated and returns written
//! (`allocator::tpu::run`). Rust chooses each kernel's spec (shapes,
//! blocks, dtypes); the kernel's Python traces it, so TPU kernels run where
//! lumen runs from Python (the `python` feature). Kernels address the
//! memory by row (512 bytes): each buffer they read or write starts at
//! one, at a row the run passes them as a scalar.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use crate::device::tpu::{error, ffi};
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;

/// The Mosaic module of module `module`'s kernel for `spec` (JSON): its
/// `program(spec)`, through `lumen.ops.tpu.pallas.lower`.
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
        "{module}: TPU kernels are lumen.pallas kernels, traced from Python: run lumen from Python"
    ))
}

/// `body` (a Mosaic module) as the StableHLO program libtpu compiles: one
/// `tpu_custom_call` of it on `scalars` i32s, `host` rows of host bytes (if
/// any) and the device's `rows` of memory, which it returns written.
fn program(name: &str, body: &str, scalars: usize, host: Option<usize>, rows: usize) -> String {
    let rows_of = |n| format!("tensor<{n}x1x128xui32>");
    let mut types = vec![format!("tensor<{scalars}xi32>")];
    types.extend(host.map(rows_of));
    types.push(rows_of(rows));
    let memory = types.len() - 1;
    let params: Vec<String> = types
        .iter()
        .enumerate()
        .map(|(i, t)| match i == memory {
            true => format!("%arg{i}: {t} {{tf.aliasing_output = 0 : i32}}"),
            false => format!("%arg{i}: {t}"),
        })
        .collect();
    let args: Vec<String> = (0..types.len()).map(|i| format!("%arg{i}")).collect();
    let layouts: Vec<&str> = (0..types.len())
        .map(|i| match i {
            0 => "dense<0> : tensor<1xindex>",
            _ => "dense<[2, 1, 0]> : tensor<3xindex>",
        })
        .collect();
    let result = rows_of(rows);
    format!(
        "module @lumen attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32}} {{\n  \
         func.func public @main({}) -> {result} {{\n    \
         %0 = stablehlo.custom_call @tpu_custom_call({}) {{backend_config = \
         \"{{\\22custom_call_config\\22: {{\\22body\\22: \\22{}\\22, \\22serialization_format\\22: 1, \
         \\22needs_layout_passes\\22: true}}}}\", kernel_name = \"{name}\", operand_layouts = [{}], \
         output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], \
         operand_index = {memory}, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : \
         tensor<3xindex>]}} : ({}) -> {result}\n    return %0 : {result}\n  }}\n}}\n",
        params.join(", "),
        args.join(", "),
        base64(body.as_bytes()),
        layouts.join(", "),
        types.join(", "),
    )
}

/// `bytes` in standard base64, padded.
fn base64(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..4 {
            match i <= chunk.len() {
                true => out.push(DIGITS[(n >> (18 - 6 * i) & 63) as usize] as char),
                false => out.push('='),
            }
        }
    }
    out
}

/// A compiled program (a PJRT executable, kept for the process).
#[derive(Clone, Copy)]
struct Executable(*mut c_void);

// SAFETY: a PJRT executable, which PJRT's API runs from any thread.
unsafe impl Send for Executable {}

/// Module `module`'s kernel for `spec`, lowered and compiled once per
/// process: a program of `scalars` scalars and `host` rows of host bytes.
fn compiled(
    module: &str,
    spec: &str,
    scalars: usize,
    host: Option<usize>,
    rows: usize,
) -> Result<Executable, String> {
    static CACHE: OnceLock<Mutex<HashMap<String, Executable>>> = OnceLock::new();
    let key = format!("{module} {spec} {scalars} {host:?}");
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
    let code = program(module, &lower(module, spec)?, scalars, host, rows);
    // SAFETY: `code`'s bytes.
    let executable = unsafe { ffi::lumen_tpu_compile(code.as_ptr().cast(), code.len()) };
    if executable.is_null() {
        return Err(error());
    }
    cache().insert(key, Executable(executable));
    Ok(Executable(executable))
}

/// A kernel to launch: the Pallas module of its source (`module`), its
/// spec's fields (`spec`: JSON members, `"name": value, ...`; the scalars'
/// count, the host rows and the memory's `rows` are added), its scalars:
/// the rows of `rows` (TPU buffers, each at the start of a row) then
/// `scalars`, and `host` bytes (whole rows) copied to the TPU for it.
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
    let scalars = launch.rows.len() + launch.scalars.len();
    let host = (!launch.host.is_empty()).then_some(launch.host.len() / crate::allocator::tpu::ROW);
    let device = format!(
        "\"scalars\": {scalars}, \"host_rows\": {}, \"rows\": {rows}",
        host.map_or("null".to_string(), |h| h.to_string())
    );
    let spec = match launch.spec.as_str() {
        "" => format!("{{{device}}}"),
        fields => format!("{{{fields}, {device}}}"),
    };
    let executable = compiled(launch.module, &spec, scalars, host, rows)
        .map_err(|e| format!("{}: {e}", launch.module))?;
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
        Primitive::PallasCall { kernel, .. } => {
            // Its operands, its value, then its other outputs (which `inputs`
            // ends with).
            let n = step.inputs.len();
            let rows: Vec<*const u8> = inputs[..n]
                .iter()
                .copied()
                .chain(std::iter::once(output.cast_const()))
                .chain(inputs[n..].iter().copied())
                .collect();
            let launched = Launch {
                module: "lumen.pallas",
                spec: format!("\"kernel\": {kernel}"),
                rows: &rows,
                scalars: &[],
                host: &[],
            };
            launch(index, launched).map_err(|e| format!("{}: {e}", step.label))
        }
        _ => Err(format!("{} does not run on TPU yet", step.primitive.name())),
    }
}
