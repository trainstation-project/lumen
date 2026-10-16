use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate as core;

pub(crate) fn resolve_device(device: Option<&Bound<'_, PyAny>>) -> PyResult<core::Device> {
    let device = match device {
        None => return Ok(core::Device::Cpu(0)),
        Some(d) if d.is_none() => return Ok(core::Device::Cpu(0)),
        Some(d) => match d.cast::<PyDevice>() {
            Ok(d) => d.get().to_core(),
            Err(_) => {
                let spec: &str = d.extract().map_err(|_| {
                    PyTypeError::new_err("device must be a string or a lumen.device")
                })?;
                PyDevice::parse(spec)?.to_core()
            }
        },
    };
    core::allocator::allocator_for(device).map_err(PyRuntimeError::new_err)?;
    Ok(device)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DeviceType {
    Cpu,
    Mps,
    Cuda,
    Meta,
}

impl DeviceType {
    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "cpu" => Ok(DeviceType::Cpu),
            "mps" => Ok(DeviceType::Mps),
            "cuda" => Ok(DeviceType::Cuda),
            "meta" => Ok(DeviceType::Meta),
            other => Err(PyValueError::new_err(format!(
                "Expected one of cpu, mps, cuda, meta device type at start of device string: {other}"
            ))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            DeviceType::Cpu => "cpu",
            DeviceType::Mps => "mps",
            DeviceType::Cuda => "cuda",
            DeviceType::Meta => "meta",
        }
    }
}

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
        Ok(PyDevice { kind, index })
    }

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

    fn to_core(&self) -> core::Device {
        let index = self.index.unwrap_or(0);
        match self.kind {
            DeviceType::Cpu => core::Device::Cpu(index),
            DeviceType::Mps => core::Device::Mps(index),
            DeviceType::Cuda => core::Device::Cuda(index),
            DeviceType::Meta => core::Device::Meta(index),
        }
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

    #[getter(r#type)]
    fn kind(&self) -> &'static str {
        self.kind.name()
    }

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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()
}
