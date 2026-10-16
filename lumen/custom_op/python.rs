use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate as core;
use crate::tensor::python::PyTensor;

/// A Metal kernel of your own (`lumen.custom_op.MetalKernel(source, name)`), for a
/// custom op to launch into lumen's MPS stream.
#[pyclass(name = "MetalKernel", module = "lumen", frozen)]
pub(crate) struct PyMetalKernel {
    inner: core::custom_op::MetalKernel,
}

#[pymethods]
impl PyMetalKernel {
    #[new]
    #[pyo3(signature = (source, name = None))]
    fn new(source: &str, name: Option<&str>) -> PyResult<Self> {
        let inner =
            core::custom_op::MetalKernel::new(source, name).map_err(PyValueError::new_err)?;
        Ok(PyMetalKernel { inner })
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    #[pyo3(signature = (tensors, args = Vec::new(), *, grid = None, threadgroups = None))]
    fn launch(
        &self,
        tensors: Vec<PyRef<'_, PyTensor>>,
        args: Vec<Bound<'_, PyAny>>,
        grid: Option<Vec<usize>>,
        threadgroups: Option<Vec<usize>>,
    ) -> PyResult<()> {
        use core::custom_op::Grid;
        let sizes = |mut s: Vec<usize>| {
            s.resize(3, 1);
            [s[0], s[1], s[2]]
        };
        let grid = match (grid, threadgroups) {
            (Some(s), None) => Grid::Threads(sizes(s)),
            (None, Some(s)) => Grid::Groups(sizes(s)),
            _ => {
                return Err(PyTypeError::new_err(
                    "launch takes one of grid and threadgroups",
                ));
            }
        };
        let args = args
            .iter()
            .map(|a| match a.cast::<PyBytes>() {
                Ok(bytes) => Ok(bytes.as_bytes().to_vec()),
                Err(_) => a.call_method0("tobytes")?.extract::<Vec<u8>>(),
            })
            .collect::<PyResult<Vec<Vec<u8>>>>()?;
        let tensors: Vec<core::Tensor> = tensors.iter().map(|t| t.inner.clone()).collect();
        self.inner
            .launch(&tensors, &args, grid)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "<{}.MetalKernel {}>",
            crate::LIBRARY_NAME,
            self.inner.name()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMetalKernel>()
}
