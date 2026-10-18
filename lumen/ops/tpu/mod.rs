//! TPU kernels: Mosaic kernels ([`mosaic`]) written by lumen, each wrapped in
//! a one-op StableHLO program (`tpu_custom_call`, as Pallas does) that
//! libtpu compiles, and run on a device's memory: the TPU allocator's one
//! buffer, which the program takes donated and returns written
//! (`allocator::tpu::run`). Its kernels address it by row (512 bytes): each
//! buffer they read or write starts at one, at a row the run passes them as
//! a scalar.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use crate::device::tpu::{error, ffi};
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;

pub(crate) mod mosaic;
#[cfg(test)]
mod tests;

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

/// `code` compiled, once per process.
fn compiled(code: &str) -> Result<Executable, String> {
    static CACHE: OnceLock<Mutex<HashMap<String, Executable>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(&executable) = cache.get(code) {
        return Ok(executable);
    }
    // SAFETY: `code`'s bytes.
    let executable = unsafe { ffi::lumen_tpu_compile(code.as_ptr().cast(), code.len()) };
    if executable.is_null() {
        return Err(error());
    }
    cache.insert(code.to_string(), Executable(executable));
    Ok(Executable(executable))
}

/// A kernel to launch: its module (`mosaic::Kernel::finish`), its scalars:
/// the rows of `rows` (TPU buffers, each at the start of a row) then
/// `scalars`, and `host` bytes (whole rows) copied to the TPU for it.
pub(crate) struct Launch<'a> {
    pub(crate) name: &'a str,
    pub(crate) body: String,
    pub(crate) rows: &'a [*const u8],
    pub(crate) scalars: &'a [i32],
    pub(crate) host: &'a [u8],
}

/// Runs `launch` on device `index`'s memory, in order after the kernels
/// before it.
pub(crate) fn launch(index: usize, launch: Launch) -> Result<(), String> {
    let row = crate::allocator::tpu::ROW;
    let host = (!launch.host.is_empty()).then_some(launch.host.len() / row);
    let count = launch.rows.len() + launch.scalars.len();
    let rows = crate::allocator::tpu::rows(index)?;
    let code = program(launch.name, &launch.body, count, host, rows);
    let executable = compiled(&code).map_err(|e| format!("{}: {e}", launch.name))?;
    crate::allocator::tpu::run(
        index,
        executable.0,
        launch.rows,
        launch.scalars,
        launch.host,
    )
    .map_err(|e| format!("{}: {e}", launch.name))
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
