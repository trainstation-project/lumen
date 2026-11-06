//! Lowering TPU kernels to Mosaic: each Pallas kernel (or generated program) traced by
//! `lumen.codegen.utils.tpu.pallas` for a device's memory, once per spec and device.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::codegen::tpu::GENERATED;

/// Mosaic source of `module`'s kernel for JSON `spec`, lowered by `lumen.codegen.utils.tpu.pallas`.
#[cfg(feature = "python")]
fn lower(module: &str, spec: &str) -> Result<String, String> {
    use pyo3::prelude::*;

    Python::attach(|py| {
        py.import("lumen.codegen.utils.tpu.pallas")?
            .call_method1("lower", (module, spec))?
            .extract::<String>()
    })
    .map_err(|e| e.to_string())
}

/// Mosaic source of the generated program `source` (defining `program(device)`), traced on a device with
/// `scalars` scalars, `host_rows` host rows and `rows` memory rows by `lumen.codegen.utils.tpu.pallas`.
#[cfg(feature = "python")]
fn lower_source(
    source: &str,
    scalars: usize,
    host_rows: Option<usize>,
    rows: usize,
) -> Result<String, String> {
    use pyo3::prelude::*;

    Python::attach(|py| {
        py.import("lumen.codegen.utils.tpu.pallas")?
            .call_method1("lower_source", (source, scalars, host_rows, rows))?
            .extract::<String>()
    })
    .map_err(|e| e.to_string())
}

/// Fails: TPU kernels are traced from Python.
#[cfg(not(feature = "python"))]
fn lower_source(_: &str, _: usize, _: Option<usize>, _: usize) -> Result<String, String> {
    Err("generated TPU kernels are lumen.kernels.pallas programs, traced from Python: run lumen from Python".into())
}

/// Fails: TPU kernels are traced from Python.
#[cfg(not(feature = "python"))]
fn lower(module: &str, _spec: &str) -> Result<String, String> {
    Err(format!(
        "{module}: TPU kernels are lumen.kernels.pallas kernels, traced from Python: run lumen from Python"
    ))
}

/// Mosaic of `module` for `spec`, cached per `(module, spec)`.
fn lowered(module: &str, spec: &str) -> Result<Arc<str>, String> {
    cached(format!("{module} {spec}"), || lower(module, spec))
}

/// Mosaic cached under `key`, lowered by `lower` the first time.
fn cached(key: String, lower: impl FnOnce() -> Result<String, String>) -> Result<Arc<str>, String> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<str>>>> = OnceLock::new();

    let cache = || {
        CACHE
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    };

    if let Some(mosaic) = cache().get(&key) {
        return Ok(Arc::clone(mosaic));
    }

    let mosaic: Arc<str> = lower()?.into();
    cache().insert(key, Arc::clone(&mosaic));
    Ok(mosaic)
}

/// Mosaic of a kernel of `module` with spec `fields` on device `index`, `scalars` scalars and `host` host rows,
/// and the full spec it was lowered from.
pub(crate) fn lowering(
    index: usize,
    module: &str,
    fields: &str,
    scalars: usize,
    host: Option<usize>,
) -> Result<(Arc<str>, Arc<str>), String> {
    let rows = crate::allocator::tpu::rows(index)?;

    let device = format!(
        "\"scalars\": {scalars}, \"host_rows\": {}, \"rows\": {rows}",
        host.map_or("null".to_string(), |h| h.to_string())
    );

    let spec = match (module, fields) {
        (GENERATED, source) => source.to_owned(),
        (_, "") => format!("{{{device}}}"),
        (_, fields) => format!("{{{fields}, {device}}}"),
    };

    // A generated program's Mosaic depends on the device's memory as well as its source.
    let mosaic = match module {
        GENERATED => cached(format!("{scalars} {host:?} {rows}\n{spec}"), || {
            lower_source(&spec, scalars, host, rows)
        }),
        module => lowered(module, &spec),
    }?;

    Ok((mosaic, spec.into()))
}
