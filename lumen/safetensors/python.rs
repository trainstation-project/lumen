//! Bindings for [`crate::safetensors`]; the Python API (`save_file`,
//! `load_file`, `safe_open`, ...) is in `lumen/safetensors/__init__.py`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::exceptions::{PyKeyError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::{Error, SafeTensorsFile};
use crate::python::resolve_device;
use crate::tensor::python::PyTensor;
use crate::{Device, Tensor};

fn to_py(e: Error) -> PyErr {
    match e {
        Error::Io(e) => PyOSError::new_err(e.to_string()),
        Error::TensorNotFound(name) => PyKeyError::new_err(name),
        e => PyValueError::new_err(e.to_string()),
    }
}

type Named<'py> = Vec<(String, PyRef<'py, PyTensor>)>;

fn named<'a>(tensors: &'a Named<'_>) -> Vec<(&'a str, &'a Tensor)> {
    tensors
        .iter()
        .map(|(n, t)| (n.as_str(), &t.inner))
        .collect()
}

fn wrap(tensors: Vec<(String, Tensor)>) -> Vec<(String, PyTensor)> {
    tensors
        .into_iter()
        .map(|(n, t)| (n, PyTensor::wrap(t)))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (tensors, metadata = None))]
fn _safetensors_serialize<'py>(
    py: Python<'py>,
    tensors: Named<'_>,
    metadata: Option<BTreeMap<String, String>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = super::serialize(&named(&tensors), metadata).map_err(to_py)?;
    Ok(PyBytes::new(py, &bytes))
}

#[pyfunction]
#[pyo3(signature = (tensors, path, metadata = None))]
fn _safetensors_save_file(
    tensors: Named<'_>,
    path: PathBuf,
    metadata: Option<BTreeMap<String, String>>,
) -> PyResult<()> {
    super::save_file(&named(&tensors), metadata, path).map_err(to_py)
}

#[pyfunction]
#[pyo3(signature = (data, device = None))]
fn _safetensors_deserialize(
    data: &[u8],
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<(String, PyTensor)>> {
    let device = resolve_device(device)?;
    super::deserialize(data, device).map(wrap).map_err(to_py)
}

/// `lumen.safetensors.safe_open`: an open file, read a tensor at a time.
#[pyclass(name = "safe_open", module = "lumen.safetensors")]
struct PySafeOpen {
    file: Option<SafeTensorsFile>,
    device: Device,
}

impl PySafeOpen {
    fn file(&self) -> PyResult<&SafeTensorsFile> {
        self.file
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("the file is closed"))
    }
}

#[pymethods]
impl PySafeOpen {
    #[new]
    #[pyo3(signature = (filename, device = None))]
    fn new(filename: PathBuf, device: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let device = resolve_device(device)?;
        let file = SafeTensorsFile::open(filename).map_err(to_py)?;
        Ok(Self {
            file: Some(file),
            device,
        })
    }

    /// The tensors' names, sorted.
    fn keys(&self) -> PyResult<Vec<String>> {
        let mut names = self.offset_keys()?;
        names.sort();
        Ok(names)
    }

    /// The tensors' names, in file order.
    fn offset_keys(&self) -> PyResult<Vec<String>> {
        let tensors = self.file()?.metadata().tensors();
        Ok(tensors.iter().map(|(n, _)| n.clone()).collect())
    }

    /// The file's `__metadata__`, or None.
    fn metadata(&self) -> PyResult<Option<BTreeMap<String, String>>> {
        Ok(self.file()?.metadata().metadata().cloned())
    }

    fn get_tensor(&self, name: &str) -> PyResult<PyTensor> {
        let t = self.file()?.tensor(name, self.device).map_err(to_py)?;
        Ok(PyTensor::wrap(t))
    }

    /// Every tensor, as `{name: tensor}` in file order.
    fn get_tensors(&self) -> PyResult<Vec<(String, PyTensor)>> {
        self.file()?.tensors(self.device).map(wrap).map_err(to_py)
    }

    fn close(&mut self) {
        self.file = None;
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, _args: &Bound<'_, pyo3::types::PyTuple>) {
        self.close();
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_safetensors_serialize, m)?)?;
    m.add_function(wrap_pyfunction!(_safetensors_save_file, m)?)?;
    m.add_function(wrap_pyfunction!(_safetensors_deserialize, m)?)?;
    m.add_class::<PySafeOpen>()
}
