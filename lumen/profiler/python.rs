//! Python bindings (`lumen.profiler._C`) for starting and stopping the profiler and reading its results.
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{Activity, Event, EventAvg, EventKind, Profile, ProfilerConfig, RecordGuard, SortBy};
use crate::tensor::python::dtype_name;

/// Nanoseconds as fractional microseconds.
fn us(ns: u64) -> f64 {
    ns as f64 / 1000.0
}

/// Starts a session recording the named activities (`"cpu"`, `"cuda"`, `"mps"`, `"tpu"`).
#[pyfunction]
#[pyo3(name = "_profiler_start")]
fn start(activities: Vec<String>, profile_memory: bool, record_shapes: bool) -> PyResult<()> {
    let activities = activities
        .iter()
        .map(|a| match a.as_str() {
            "cpu" => Ok(Activity::Cpu),
            "cuda" => Ok(Activity::Cuda),
            "mps" => Ok(Activity::Mps),
            "tpu" => Ok(Activity::Tpu),
            other => Err(PyValueError::new_err(format!(
                "unknown profiler activity {other:?}; expected cpu, cuda, mps or tpu"
            ))),
        })
        .collect::<PyResult<Vec<_>>>()?;

    if activities.contains(&Activity::Cuda) && !cuda_timing() {
        return Err(PyRuntimeError::new_err(CUDA_TIMING_MISSING));
    }

    super::start(ProfilerConfig {
        activities,
        profile_memory,
        record_shapes,
    })
    .map_err(PyRuntimeError::new_err)
}

/// Error raised when CUDA timing is asked for in a build without CUPTI.
const CUDA_TIMING_MISSING: &str = "this build of lumen cannot time CUDA: build.rs did not find \
     CUPTI (libcupti). Set CUPTI_LIB_DIR to its directory and rebuild.";

/// True when this build links CUPTI and can time CUDA kernels.
#[pyfunction]
#[pyo3(name = "_profiler_cuda_timing")]
fn cuda_timing() -> bool {
    cfg!(lumen_cupti_linked)
}

/// Ends the session and returns its profile.
#[pyfunction]
#[pyo3(name = "_profiler_stop")]
fn stop() -> PyResult<PyProfile> {
    let inner = super::stop().map_err(PyRuntimeError::new_err)?;
    Ok(PyProfile { inner })
}

/// True when a profiler session is recording.
#[pyfunction]
#[pyo3(name = "_profiler_enabled")]
fn enabled() -> bool {
    super::is_enabled()
}

/// An open `record_function` range held by Python until `exit`.
#[pyclass(name = "_RecordFunction", module = "lumen.profiler._C")]
struct PyRecordFunction {
    guard: Option<RecordGuard>,
}

#[pymethods]
impl PyRecordFunction {
    /// Ends the range; later calls do nothing.
    fn exit(&mut self) {
        self.guard.take();
    }
}

/// Opens a user range named `name`.
#[pyfunction]
#[pyo3(name = "_record_function_enter")]
fn record_function_enter(name: String) -> PyRecordFunction {
    PyRecordFunction {
        guard: Some(super::record_function(name)),
    }
}

/// A finished profile handed to Python.
#[pyclass(name = "_Profile", module = "lumen.profiler._C")]
struct PyProfile {
    inner: Profile,
}

/// The `kind` string Python sees for an event kind.
fn kind_name(kind: EventKind) -> &'static str {
    match kind {
        EventKind::Op => "op",
        EventKind::UserRange => "user_range",
        EventKind::Memory => "memory",
        EventKind::Gpu => "gpu",
        EventKind::HostKernel => "host_kernel",
    }
}

/// An event as a Python dict, times in microseconds and types as `(dtype, shape)` pairs.
fn event_dict<'py>(py: Python<'py>, e: &Event) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", e.id)?;
    d.set_item("name", &e.name)?;
    d.set_item("kind", kind_name(e.kind))?;
    d.set_item("start_us", us(e.start_ns))?;
    d.set_item("end_us", us(e.end_ns))?;
    d.set_item("duration_us", us(e.duration_ns()))?;
    d.set_item("thread", e.thread)?;
    d.set_item("parent", e.parent)?;
    d.set_item("device", e.device.to_string())?;

    let types = |types: &[crate::graph::TensorType]| -> Vec<(&'static str, Vec<usize>)> {
        types
            .iter()
            .map(|t| (dtype_name(t.dtype), t.shape.clone()))
            .collect()
    };

    d.set_item("inputs", types(&e.inputs))?;
    d.set_item("input_strides", &e.input_strides)?;
    d.set_item("outputs", types(&e.outputs))?;
    d.set_item("output_strides", &e.output_strides)?;
    let accum: Vec<&'static str> = e.accum.iter().map(|&d| dtype_name(d)).collect();
    d.set_item("accum", accum)?;
    d.set_item("bytes", e.bytes)?;
    d.set_item("addr", e.addr)?;
    d.set_item("total_allocated", e.total_allocated)?;
    d.set_item("total_reserved", e.total_reserved)?;
    d.set_item("kernel", e.kernel.as_deref())?;
    Ok(d)
}

/// A `key_averages` row as a Python dict, times in microseconds.
fn avg_dict<'py>(py: Python<'py>, a: &EventAvg) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("key", &a.name)?;
    d.set_item("is_device_event", a.is_device_event)?;
    d.set_item("count", a.count)?;
    d.set_item("cpu_time_total", us(a.cpu_time_total))?;
    d.set_item("self_cpu_time_total", us(a.self_cpu_time_total))?;
    d.set_item("cpu_time", us(a.cpu_time_avg()))?;
    d.set_item("device_time_total", us(a.device_time_total))?;
    d.set_item("self_device_time_total", us(a.self_device_time_total))?;
    d.set_item("device_time", us(a.device_time_avg()))?;
    d.set_item("cpu_memory_usage", a.cpu_memory_usage)?;
    d.set_item("self_cpu_memory_usage", a.self_cpu_memory_usage)?;
    d.set_item("device_memory_usage", a.device_memory_usage)?;
    d.set_item("self_device_memory_usage", a.self_device_memory_usage)?;
    d.set_item("input_shapes", &a.input_shapes)?;
    d.set_item("input_strides", &a.input_strides)?;
    Ok(d)
}

#[pymethods]
impl PyProfile {
    /// Every event as a dict, by start time.
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .events()
            .iter()
            .map(|e| event_dict(py, e))
            .collect()
    }

    /// Per-name totals as dicts, optionally split by input types and strides.
    #[pyo3(signature = (group_by_input_shape=false))]
    fn key_averages<'py>(
        &self,
        py: Python<'py>,
        group_by_input_shape: bool,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .key_averages(group_by_input_shape)
            .iter()
            .map(|a| avg_dict(py, a))
            .collect()
    }

    /// The PyTorch-style summary table; a negative `row_limit` means no limit.
    #[pyo3(signature = (sort_by=None, row_limit=None, group_by_input_shape=false))]
    fn table(
        &self,
        sort_by: Option<&str>,
        row_limit: Option<i64>,
        group_by_input_shape: bool,
    ) -> PyResult<String> {
        let sort_by = sort_by
            .map(|s| {
                SortBy::parse(s)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown sort_by key {s:?}")))
            })
            .transpose()?;

        let row_limit = row_limit.and_then(|n| usize::try_from(n).ok());
        Ok(self.inner.table(sort_by, row_limit, group_by_input_shape))
    }

    /// The profile as Chrome trace JSON.
    fn chrome_trace(&self) -> String {
        self.inner.chrome_trace()
    }
}

/// Adds the profiler functions and classes to module `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_timing, m)?)?;
    m.add_function(wrap_pyfunction!(stop, m)?)?;
    m.add_function(wrap_pyfunction!(enabled, m)?)?;
    m.add_function(wrap_pyfunction!(record_function_enter, m)?)?;
    m.add_class::<PyRecordFunction>()?;
    m.add_class::<PyProfile>()?;
    Ok(())
}
