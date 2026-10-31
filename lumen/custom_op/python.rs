//! Python binding of `MetalKernel`.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate as core;
use crate::tensor::python::PyTensor;

/// Python `lumen.custom_op.MetalKernel`: a compiled Metal kernel callable from a custom op.
#[pyclass(name = "MetalKernel", module = "lumen", frozen)]
pub(crate) struct PyMetalKernel {
    inner: core::custom_op::mps::MetalKernel,
}

#[pymethods]
impl PyMetalKernel {
    /// Compiles `source`, choosing kernel `name` when the source defines several.
    #[new]
    #[pyo3(signature = (source, name = None))]
    fn new(source: &str, name: Option<&str>) -> PyResult<Self> {
        let inner =
            core::custom_op::mps::MetalKernel::new(source, name).map_err(PyValueError::new_err)?;

        Ok(PyMetalKernel { inner })
    }

    /// Name of the compiled kernel function.
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Launches over `grid` threads or `threadgroups` groups (exactly one, up to 3-D); each of `args`
    /// is `bytes` or an object with `tobytes()`, such as a NumPy scalar.
    #[pyo3(signature = (tensors, args = Vec::new(), *, grid = None, threadgroups = None))]
    fn launch(
        &self,
        tensors: Vec<PyRef<'_, PyTensor>>,
        args: Vec<Bound<'_, PyAny>>,
        grid: Option<Vec<usize>>,
        threadgroups: Option<Vec<usize>>,
    ) -> PyResult<()> {
        use core::custom_op::mps::Grid;

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

    /// Shows the kernel name.
    fn __repr__(&self) -> String {
        format!(
            "<{}.MetalKernel {}>",
            crate::LIBRARY_NAME,
            self.inner.name()
        )
    }
}

/// Adds `MetalKernel` and the kernel registration functions to the Python module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMetalKernel>()?;
    super::register::register(m)
}
