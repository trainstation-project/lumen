//! Python bindings for `lumen.distributed`: `TCPStore`, `DeviceMesh` and the process group functions.
use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};
use std::sync::Arc;
use std::time::Duration;

use super::{DeviceMesh, TcpStore};

/// Converts a Python timeout (seconds or a `timedelta`) to a `Duration`, defaulting to `DEFAULT_TIMEOUT` for `None`.
fn duration(timeout: Option<&Bound<'_, PyAny>>) -> PyResult<Duration> {
    let Some(timeout) = timeout.filter(|t| !t.is_none()) else {
        return Ok(super::DEFAULT_TIMEOUT);
    };
    let seconds: f64 = match timeout.hasattr("total_seconds")? {
        true => timeout.call_method0("total_seconds")?.extract()?,
        false => timeout.extract()?,
    };
    Duration::try_from_secs_f64(seconds).map_err(|_| {
        PyValueError::new_err(format!(
            "timeout must be a non-negative duration, got {seconds}"
        ))
    })
}

/// Maps the `-1` "unset" sentinel of a Python int argument to `None`, rejecting other negatives.
fn index(name: &str, value: i64) -> PyResult<Option<usize>> {
    match value {
        -1 => Ok(None),
        v if v >= 0 => Ok(Some(v as usize)),
        v => Err(PyValueError::new_err(format!(
            "{name} must not be negative, got {v}"
        ))),
    }
}

/// Key-value store over TCP shared by every rank of a job; one process hosts it with `is_master=True`.
#[pyclass(name = "TCPStore", module = "lumen.distributed", frozen)]
pub(crate) struct PyTcpStore {
    store: Arc<TcpStore>,
}

#[pymethods]
impl PyTcpStore {
    /// Hosts or connects to the store at `host_name:port`.
    #[new]
    #[pyo3(signature = (host_name, port, *, is_master = false, timeout = None))]
    fn new(
        py: Python<'_>,
        host_name: &str,
        port: u16,
        is_master: bool,
        timeout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let timeout = duration(timeout)?;
        let store = py
            .detach(|| TcpStore::new(host_name, port, is_master, timeout))
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyTcpStore {
            store: Arc::new(store),
        })
    }

    /// Host name the store was created with.
    #[getter]
    fn host(&self) -> &str {
        self.store.host()
    }

    /// Port the store listens on.
    #[getter]
    fn port(&self) -> u16 {
        self.store.port()
    }

    /// Stores `value` (str or bytes) under `key`.
    fn set(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value: Vec<u8> = match value.extract::<String>() {
            Ok(text) => text.into_bytes(),
            Err(_) => value.extract()?,
        };
        py.detach(|| self.store.set(key, &value))
            .map_err(PyRuntimeError::new_err)
    }

    /// Returns the bytes under `key`, waiting for it to be set.
    fn get<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyBytes>> {
        let value = py
            .detach(|| self.store.get(key))
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyBytes::new(py, &value))
    }

    /// Atomically adds `amount` to the integer counter at `key` and returns the new value.
    fn add(&self, py: Python<'_>, key: &str, amount: i64) -> PyResult<i64> {
        py.detach(|| self.store.add(key, amount))
            .map_err(PyRuntimeError::new_err)
    }

    /// Blocks until every key in `keys` is set, up to `timeout` or the store's own timeout.
    #[pyo3(signature = (keys, timeout = None))]
    fn wait(
        &self,
        py: Python<'_>,
        keys: Vec<String>,
        timeout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let timeout = match timeout.filter(|t| !t.is_none()) {
            None => self.store.timeout(),
            Some(t) => duration(Some(t))?,
        };
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        py.detach(|| self.store.wait(&keys, timeout))
            .map_err(PyRuntimeError::new_err)
    }
}

/// Initialises the default process group from `store` or by rendezvous at `init_method` (env:// by default).
#[pyfunction]
#[pyo3(signature = (*, init_method = None, timeout = None, world_size = -1, rank = -1, store = None))]
fn init_process_group(
    py: Python<'_>,
    init_method: Option<&str>,
    timeout: Option<&Bound<'_, PyAny>>,
    world_size: i64,
    rank: i64,
    store: Option<&Bound<'_, PyTcpStore>>,
) -> PyResult<()> {
    let timeout = duration(timeout)?;
    let (rank, world_size) = (index("rank", rank)?, index("world_size", world_size)?);
    let (store, rank, world_size) = match (store, init_method) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "init_method and store are mutually exclusive",
            ));
        }
        (Some(store), None) => {
            let required = |name| PyValueError::new_err(format!("a store needs {name}"));
            (
                store.get().store.clone(),
                rank.ok_or_else(|| required("rank"))?,
                world_size.ok_or_else(|| required("world_size"))?,
            )
        }
        (None, init_method) => py
            .detach(|| {
                super::rendezvous(init_method.unwrap_or("env://"), rank, world_size, timeout)
            })
            .map_err(PyValueError::new_err)?,
    };
    py.detach(|| super::init_process_group(store, rank, world_size))
        .map_err(PyRuntimeError::new_err)
}

/// Tears down the default process group.
#[pyfunction]
fn destroy_process_group() -> PyResult<()> {
    super::destroy_process_group().map_err(PyRuntimeError::new_err)
}

/// Blocks until every process in `ranks` reaches the barrier (`lumen.distributed.barrier`).
#[pyfunction]
fn _barrier(py: Python<'_>, ranks: Vec<usize>) -> PyResult<()> {
    py.detach(|| super::barrier(&ranks))
        .map_err(PyRuntimeError::new_err)
}

/// Reports whether the default process group is initialised.
#[pyfunction]
fn is_initialized() -> bool {
    super::is_initialized()
}

/// Flattens a nested sequence of ranks row-major into `ranks`, recording each level's length in `shape`.
fn flatten_mesh(
    mesh: &Bound<'_, PyAny>,
    depth: usize,
    shape: &mut Vec<usize>,
    ranks: &mut Vec<usize>,
) -> PyResult<()> {
    if let Ok(rank) = mesh.extract::<i64>() {
        if depth != shape.len() {
            return Err(PyValueError::new_err(
                "the mesh must be a rectangular array",
            ));
        }
        let rank = usize::try_from(rank).map_err(|_| {
            PyValueError::new_err(format!("mesh ranks must not be negative, got {rank}"))
        })?;
        ranks.push(rank);
        return Ok(());
    }
    let rows: Vec<Bound<'_, PyAny>> = mesh
        .try_iter()
        .map_err(|_| PyTypeError::new_err("the mesh must be a nested sequence of ranks"))?
        .collect::<PyResult<_>>()?;
    // the first row at a depth sets that dimension's size; later rows must match it
    match shape.get(depth) {
        None if ranks.is_empty() => shape.push(rows.len()),
        Some(&size) if size == rows.len() => {}
        _ => {
            return Err(PyValueError::new_err(
                "the mesh must be a rectangular array",
            ));
        }
    }
    for row in &rows {
        flatten_mesh(row, depth + 1, shape, ranks)?;
    }
    Ok(())
}

/// Rebuilds the nested Python list of `ranks` laid out row-major over `shape`.
fn nested<'py>(py: Python<'py>, shape: &[usize], ranks: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    match shape {
        [] => Ok(ranks[0].into_pyobject(py)?.into_any()),
        [size, rest @ ..] => {
            let step = ranks.len() / size;
            let rows = (0..*size)
                .map(|i| nested(py, rest, &ranks[i * step..(i + 1) * step]))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, rows)?.into_any())
        }
    }
}

/// Python view of a `DeviceMesh` for the calling process.
#[pyclass(
    name = "DeviceMesh",
    module = "lumen.distributed",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyDeviceMesh {
    mesh: DeviceMesh,
}

impl PyDeviceMesh {
    /// Resolves this process's rank on `device_type` and builds the mesh, initialising the process group if needed.
    fn build(
        py: Python<'_>,
        device_type: &str,
        shape: Vec<usize>,
        ranks: Vec<usize>,
        mesh_dim_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let backend = super::mesh_backend(device_type).map_err(PyValueError::new_err)?;
        let (rank, world_size) = py
            .detach(|| backend.world(ranks.len()))
            .map_err(PyRuntimeError::new_err)?;
        let mesh = DeviceMesh::new(device_type, shape, ranks, mesh_dim_names, rank, world_size)
            .map_err(PyValueError::new_err)?;
        Ok(PyDeviceMesh { mesh })
    }

    /// Resolves a mesh dimension given as a name, a possibly negative index, or `None` for a 1-D mesh.
    fn mesh_dim(&self, mesh_dim: Option<&Bound<'_, PyAny>>) -> PyResult<usize> {
        let Some(mesh_dim) = mesh_dim.filter(|d| !d.is_none()) else {
            return match self.mesh.ndim() {
                1 => Ok(0),
                n => Err(PyRuntimeError::new_err(format!(
                    "the DeviceMesh has {n} dimensions: pass mesh_dim"
                ))),
            };
        };
        if let Ok(name) = mesh_dim.extract::<String>() {
            return self.mesh.dim(&name).map_err(PyKeyError::new_err);
        }
        let dim: i64 = mesh_dim.extract().map_err(|_| {
            PyTypeError::new_err("mesh_dim must be an int or a mesh dimension name")
        })?;
        let ndim = self.mesh.ndim() as i64;
        match dim {
            d if (0..ndim).contains(&d) => Ok(d as usize),
            d if (-ndim..0).contains(&d) => Ok((d + ndim) as usize),
            d => Err(PyIndexError::new_err(format!(
                "mesh_dim {d} is out of range for a {ndim}-dimensional DeviceMesh"
            ))),
        }
    }
}

#[pymethods]
impl PyDeviceMesh {
    /// Builds a mesh on `device_type` from a nested sequence of ranks.
    #[new]
    #[pyo3(signature = (device_type, mesh, *, mesh_dim_names = None))]
    fn new(
        py: Python<'_>,
        device_type: &str,
        mesh: &Bound<'_, PyAny>,
        mesh_dim_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let (mut shape, mut ranks) = (Vec::new(), Vec::new());
        flatten_mesh(mesh, 0, &mut shape, &mut ranks)?;
        Self::build(py, device_type, shape, ranks, mesh_dim_names)
    }

    /// Canonical device type name.
    #[getter]
    fn device_type(&self) -> &'static str {
        self.mesh.device_type()
    }

    /// The mesh's ranks as a nested list.
    #[getter]
    fn mesh<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        nested(py, self.mesh.shape(), self.mesh.ranks())
    }

    /// Dimension names as a tuple, or `None`.
    #[getter]
    fn mesh_dim_names<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.mesh
            .dim_names()
            .map(|names| PyTuple::new(py, names))
            .transpose()
    }

    /// Size of each mesh dimension.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.mesh.shape())
    }

    /// Number of mesh dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.mesh.ndim()
    }

    /// Returns the total rank count, or the size of `mesh_dim` when given.
    #[pyo3(signature = (mesh_dim = None))]
    fn size(&self, mesh_dim: Option<&Bound<'_, PyAny>>) -> PyResult<usize> {
        match mesh_dim.filter(|d| !d.is_none()) {
            None => Ok(self.mesh.size()),
            Some(d) => Ok(self.mesh.shape()[self.mesh_dim(Some(d))?]),
        }
    }

    /// Returns this process's global rank.
    fn get_rank(&self) -> usize {
        self.mesh.rank()
    }

    /// Returns this process's coordinate along `mesh_dim`.
    #[pyo3(signature = (mesh_dim = None))]
    fn get_local_rank(&self, mesh_dim: Option<&Bound<'_, PyAny>>) -> PyResult<usize> {
        self.mesh
            .local_rank(self.mesh_dim(mesh_dim)?)
            .map_err(PyRuntimeError::new_err)
    }

    /// Returns this process's coordinate in the mesh, or `None` if it is not a member.
    fn get_coordinate(&self) -> Option<Vec<usize>> {
        self.mesh.coordinate()
    }

    /// Returns the submesh over one or more named dimensions that contains this process.
    fn __getitem__(&self, mesh_dim_names: &Bound<'_, PyAny>) -> PyResult<Self> {
        let names: Vec<String> = match mesh_dim_names.extract::<String>() {
            Ok(name) => vec![name],
            Err(_) => mesh_dim_names.extract().map_err(|_| {
                PyTypeError::new_err(
                    "index a DeviceMesh by a mesh dimension name or a tuple of them",
                )
            })?,
        };
        let mesh = self.mesh.slice(&names).map_err(PyKeyError::new_err)?;
        Ok(PyDeviceMesh { mesh })
    }

    /// Formats the mesh like its constructor call.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mesh = self.mesh(py)?.repr()?;
        Ok(match self.mesh_dim_names(py)? {
            None => format!("DeviceMesh('{}', {mesh})", self.mesh.device_type()),
            Some(names) => format!(
                "DeviceMesh('{}', {mesh}, mesh_dim_names={})",
                self.mesh.device_type(),
                names.repr()?
            ),
        })
    }
}

/// Builds a mesh of `mesh_shape` over ranks `0..product(mesh_shape)`.
#[pyfunction]
#[pyo3(signature = (device_type, mesh_shape, *, mesh_dim_names = None))]
fn init_device_mesh(
    py: Python<'_>,
    device_type: &str,
    mesh_shape: Vec<usize>,
    mesh_dim_names: Option<Vec<String>>,
) -> PyResult<PyDeviceMesh> {
    let count = mesh_shape.iter().product();
    PyDeviceMesh::build(
        py,
        device_type,
        mesh_shape,
        (0..count).collect(),
        mesh_dim_names,
    )
}

/// Returns this process's rank in the default group, or `RANK` before it is initialised.
#[pyfunction]
fn get_rank() -> PyResult<usize> {
    super::get_rank().map_err(PyRuntimeError::new_err)
}

/// Returns the default group's world size, or `WORLD_SIZE` before it is initialised.
#[pyfunction]
fn get_world_size() -> PyResult<usize> {
    super::get_world_size().map_err(PyRuntimeError::new_err)
}

/// Adds the distributed classes and functions to the `lumen.distributed._C` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDeviceMesh>()?;
    m.add_class::<PyTcpStore>()?;
    m.add_function(wrap_pyfunction!(init_process_group, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_process_group, m)?)?;
    m.add_function(wrap_pyfunction!(is_initialized, m)?)?;
    m.add_function(wrap_pyfunction!(_barrier, m)?)?;
    m.add_function(wrap_pyfunction!(init_device_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(get_rank, m)?)?;
    m.add_function(wrap_pyfunction!(get_world_size, m)?)
}
