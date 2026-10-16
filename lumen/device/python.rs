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

#[pyclass(
    name = "device",
    module = "lumen",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyDevice {
    device: core::Device,
    indexed: bool,
}

impl PyDevice {
    fn checked(kind: &str, index: Option<i64>) -> PyResult<Self> {
        let index = match index {
            None => None,
            Some(i) if i < 0 => {
                return Err(PyValueError::new_err(format!(
                    "Device index must not be negative, got {i}"
                )));
            }
            Some(i) => Some(i as usize),
        };
        let device = core::Device::of_type(kind, index.unwrap_or(0)).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Expected one of cpu, mps, cuda, meta device type at start of device string: {kind}"
            ))
        })?;
        Ok(PyDevice {
            device,
            indexed: index.is_some(),
        })
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
        Self::checked(kind, index)
    }

    pub(crate) fn from_core(device: core::Device) -> Self {
        PyDevice {
            device,
            indexed: true,
        }
    }

    fn to_core(&self) -> core::Device {
        self.device
    }

    fn name(&self) -> &'static str {
        self.device.backend().name()
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
            Some(_) => Self::checked(spec, index),
        }
    }

    #[getter(r#type)]
    fn kind(&self) -> &'static str {
        self.name()
    }

    #[getter]
    fn index(&self) -> Option<usize> {
        self.indexed.then(|| self.device.index())
    }

    fn is_available(&self) -> bool {
        self.device.check().is_ok()
    }

    fn __str__(&self) -> String {
        match self.indexed {
            true => self.device.to_string(),
            false => self.name().to_owned(),
        }
    }

    fn __repr__(&self) -> String {
        match self.index() {
            None => format!("device(type='{}')", self.name()),
            Some(i) => format!("device(type='{}', index={i})", self.name()),
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
