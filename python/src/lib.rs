//! Python bindings for lumen, installed as the native module `lumen._C`
//! (the role `torch._C` plays for `torch`). Semantics live in the Rust
//! core; this layer only converts types and errors.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use ::lumen as core;

// ---------------------------------------------------------------------
// lumen.device
// ---------------------------------------------------------------------

/// A device type, as in `torch.device.type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DeviceType {
    Cpu,
    Mps,
    Cuda,
}

impl DeviceType {
    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "cpu" => Ok(DeviceType::Cpu),
            "mps" => Ok(DeviceType::Mps),
            "cuda" => Ok(DeviceType::Cuda),
            other => Err(PyValueError::new_err(format!(
                "Expected one of cpu, mps, cuda device type at start of device string: {other}"
            ))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            DeviceType::Cpu => "cpu",
            DeviceType::Mps => "mps",
            DeviceType::Cuda => "cuda",
        }
    }
}

/// `lumen.device`, modeled on `torch.device`: a device type plus an
/// optional index (`None` means "the current device of that type").
///
/// ```python
/// lumen.device("cuda:1")
/// lumen.device("cuda", 1)
/// lumen.device("mps")
/// ```
#[pyclass(
    name = "device",
    module = "lumen",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PyDevice {
    kind: DeviceType,
    index: Option<usize>,
}

impl PyDevice {
    fn checked(kind: DeviceType, index: Option<i64>) -> PyResult<Self> {
        let index = match index {
            None => None,
            Some(i) if i < 0 => {
                return Err(PyValueError::new_err(format!(
                    "Device index must not be negative, got {i}"
                )));
            }
            Some(i) => Some(i as usize),
        };
        // The core has a single CPU and a single Metal device.
        if matches!(kind, DeviceType::Cpu | DeviceType::Mps) && index.is_some_and(|i| i != 0) {
            return Err(PyValueError::new_err(format!(
                "{} device index must be 0, got {}",
                kind.name(),
                index.unwrap()
            )));
        }
        Ok(PyDevice { kind, index })
    }

    /// Parse `"type"` or `"type:index"`.
    fn parse(spec: &str) -> PyResult<Self> {
        let (kind, index) = match spec.split_once(':') {
            None => (spec, None),
            Some((kind, index)) => {
                let parsed = index
                    .parse::<i64>()
                    .ok()
                    .filter(|_| index.bytes().all(|b| b.is_ascii_digit()))
                    .ok_or_else(|| {
                        PyValueError::new_err(format!("Invalid device string: '{spec}'"))
                    })?;
                (kind, Some(parsed))
            }
        };
        Self::checked(DeviceType::parse(kind)?, index)
    }
}

#[pymethods]
impl PyDevice {
    #[new]
    #[pyo3(signature = (device, index = None))]
    fn new(device: &Bound<'_, PyAny>, index: Option<i64>) -> PyResult<Self> {
        if let Ok(other) = device.cast::<PyDevice>() {
            if index.is_some() {
                return Err(PyTypeError::new_err(
                    "device(device) does not take an index",
                ));
            }
            return Ok(other.get().clone());
        }
        let spec: &str = device.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "device() expects a string or a device, got {}",
                device
                    .get_type()
                    .name()
                    .map_or("?".into(), |n| n.to_string())
            ))
        })?;
        match index {
            None => Self::parse(spec),
            Some(_) if spec.contains(':') => Err(PyValueError::new_err(format!(
                "type (string) must not include an index because index was passed \
                 explicitly: {spec}"
            ))),
            Some(_) => Self::checked(DeviceType::parse(spec)?, index),
        }
    }

    /// The device type: `"cpu"`, `"mps"` or `"cuda"`.
    #[getter(r#type)]
    fn kind(&self) -> &'static str {
        self.kind.name()
    }

    /// The device index, or `None` for "the current device".
    #[getter]
    fn index(&self) -> Option<usize> {
        self.index
    }

    fn __str__(&self) -> String {
        match self.index {
            None => self.kind.name().to_owned(),
            Some(i) => format!("{}:{i}", self.kind.name()),
        }
    }

    fn __repr__(&self) -> String {
        match self.index {
            None => format!("device(type='{}')", self.kind.name()),
            Some(i) => format!("device(type='{}', index={i})", self.kind.name()),
        }
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let cls = py.get_type::<PyDevice>().into_any().unbind();
        Ok((cls, (self.__str__(),)))
    }
}

// ---------------------------------------------------------------------
// lumen.config
// ---------------------------------------------------------------------

/// Process-wide runtime settings, exposed as the single instance
/// `lumen.config` (PyTorch reads the equivalents from environment
/// variables).
#[pyclass(name = "Config", module = "lumen")]
struct Config;

#[pymethods]
impl Config {
    /// Whether device caching allocators cache freed memory (default
    /// `True`). Set `False` to send every allocation straight to the device
    /// (PyTorch: `PYTORCH_NO_CUDA_MEMORY_CACHING=1`), e.g. to debug memory
    /// errors. Applies to subsequent allocations.
    #[getter]
    fn memory_caching(&self) -> bool {
        core::config::memory_caching()
    }

    #[setter]
    fn set_memory_caching(&self, enabled: bool) {
        core::config::set_memory_caching(enabled);
    }

    fn __repr__(&self) -> String {
        let caching = if core::config::memory_caching() {
            "True"
        } else {
            "False"
        };
        format!("lumen.config(memory_caching={caching})")
    }
}

// ---------------------------------------------------------------------
// module
// ---------------------------------------------------------------------

#[pymodule]
#[pyo3(name = "_C")]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_class::<Config>()?;
    m.add("config", Config)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
