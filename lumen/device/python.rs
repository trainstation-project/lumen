//! Python bindings for devices: the `lumen.device` class and the `lumen.<type>` submodules.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate as core;

/// A device from a Python `lumen.device`, a `"type[:index]"` string, or `None` (CPU 0).
pub(crate) fn parse_device(device: Option<&Bound<'_, PyAny>>) -> PyResult<core::Device> {
    Ok(match device {
        None => core::Device::Cpu(0),
        Some(d) if d.is_none() => core::Device::Cpu(0),
        Some(d) => match d.cast::<PyDevice>() {
            Ok(d) => d.get().to_core(),
            Err(_) => {
                let spec: &str = d.extract().map_err(|_| {
                    PyTypeError::new_err("device must be a string or a lumen.device")
                })?;
                PyDevice::parse(spec)?.to_core()
            }
        },
    })
}

/// Like [`parse_device`], but also fails if the device has no allocator on this machine.
pub(crate) fn resolve_device(device: Option<&Bound<'_, PyAny>>) -> PyResult<core::Device> {
    let device = parse_device(device)?;
    core::allocator::allocator_for(device).map_err(PyRuntimeError::new_err)?;
    Ok(device)
}

/// Python `lumen.device`: a device type with an optional index, like `torch.device`.
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
    /// Whether an index was given; unindexed devices print without one and report `index` as `None`.
    indexed: bool,
}

impl PyDevice {
    /// A device of type `kind`, rejecting negative indices and unknown types.
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

    /// Parses `"type"` or `"type:index"`, the index being decimal digits only.
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

    /// An indexed Python device for `device`.
    pub(crate) fn from_core(device: core::Device) -> Self {
        PyDevice {
            device,
            indexed: true,
        }
    }

    /// The underlying core device (index 0 when unindexed).
    fn to_core(&self) -> core::Device {
        self.device
    }

    /// Device type name, e.g. `"mps"`.
    fn name(&self) -> &'static str {
        self.device.backend().name()
    }
}

#[pymethods]
impl PyDevice {
    /// Copies a `device`, or parses a string with the index either inside it or passed separately.
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

    /// Device type name, exposed to Python as `type`.
    #[getter(r#type)]
    fn kind(&self) -> &'static str {
        self.name()
    }

    /// Device index, or `None` when none was given.
    #[getter]
    fn index(&self) -> Option<usize> {
        self.indexed.then(|| self.device.index())
    }

    /// `"type:index"`, or just `"type"` when unindexed.
    fn __str__(&self) -> String {
        match self.indexed {
            true => self.device.to_string(),
            false => self.name().to_owned(),
        }
    }

    /// `device(type='...', index=...)`, like torch.
    fn __repr__(&self) -> String {
        match self.index() {
            None => format!("device(type='{}')", self.name()),
            Some(i) => format!("device(type='{}', index={i})", self.name()),
        }
    }

    /// Pickles as `device(str(self))`.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let cls = py.get_type::<PyDevice>().into_any().unbind();
        Ok((cls, (self.__str__(),)))
    }
}

/// Builds the `lumen.<kind>` submodule with `device_count`, `is_available` and `synchronize`.
fn device_module<'py>(py: Python<'py>, kind: &'static str) -> PyResult<Bound<'py, PyModule>> {
    use pyo3::types::{PyCFunction, PyDict, PyTuple};
    let m = PyModule::new(py, &format!("{}.{kind}", crate::LIBRARY_NAME))?;
    let backend = core::Device::of_type(kind, 0)
        .expect("a device type")
        .backend();
    m.add_function(PyCFunction::new_closure(
        py,
        Some(c"device_count"),
        None,
        move |_: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| backend.device_count(),
    )?)?;
    m.add_function(PyCFunction::new_closure(
        py,
        Some(c"is_available"),
        None,
        move |_: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| backend.is_available(),
    )?)?;
    m.add_function(PyCFunction::new_closure(
        py,
        Some(c"synchronize"),
        None,
        move |args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>| -> PyResult<()> {
            // The device may be passed by keyword or as the first positional argument.
            let given = match kwargs {
                Some(k) => k.get_item("device")?,
                None => None,
            };
            let given = given.or(args.get_item(0).ok());
            let device = match given {
                Some(d) if !d.is_none() => resolve_device(Some(&d))?,
                _ => core::Device::of_type(kind, 0).expect("a device type"),
            };
            if device.backend().name() != kind {
                return Err(PyValueError::new_err(format!(
                    "{kind}.synchronize: {device} is not a {kind} device"
                )));
            }
            let stream = core::stream::stream_for(device).map_err(PyRuntimeError::new_err)?;
            let _op = core::profiler::record_op(op_name!("synchronize"), Vec::new);
            args.py().detach(|| stream.synchronize());
            Ok(())
        },
    )?)?;
    Ok(m)
}

/// Registers the device class and each device-type submodule (also in `sys.modules`) on `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let modules = m.py().import("sys")?.getattr("modules")?;
    for kind in ["cpu", "cuda", "meta", "mps", "tpu", "trainium"] {
        let sub = device_module(m.py(), kind)?;
        modules.set_item(sub.name()?, &sub)?;
        m.add(kind, sub)?;
    }
    m.add_class::<PyDevice>()
}
