//! Python bindings of the graph compiler: the `Plan` class, kernel launch capture for code
//! dumps, and the hooks that launch Triton kernels through Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::Plan;
use super::plan::Buffer;
use crate::DType;
use crate::device::Device;
use crate::device::python::resolve_device;
use crate::graph::python::{PyGraph, primitive_dict, scope_names};
use crate::graph::{Primitive, TensorType};
use crate::tensor::python::{PyTensor, dtype_name};

/// A compiled plan of a traced graph, runnable on tensors from Python.
#[pyclass(name = "Plan", module = "lumen", frozen)]
struct PyPlan {
    inner: Plan,
}

#[pymethods]
impl PyPlan {
    /// Compiles `graph` for `device`; `donate`, `parameters`, `packable` and `scalars` are input
    /// positions, `donate_into` `(input, output)` pairs, and `fuse` overrides MPS and CUDA fusion.
    #[new]
    #[pyo3(signature = (graph, device = None, fuse = None, donate = Vec::new(), parameters = None, packable = Vec::new(), scalars = Vec::new(), donate_into = Vec::new()))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        graph: PyRef<'_, PyGraph>,
        device: Option<&Bound<'_, PyAny>>,
        fuse: Option<bool>,
        donate: Vec<usize>,
        parameters: Option<Vec<usize>>,
        packable: Vec<usize>,
        scalars: Vec<usize>,
        donate_into: Vec<(usize, usize)>,
    ) -> PyResult<Self> {
        let n = graph.inner.inputs().len();

        let all = donate
            .iter()
            .chain(donate_into.iter().map(|(i, _)| i))
            .chain(parameters.iter().flatten())
            .chain(&packable)
            .chain(&scalars);

        if let Some(&i) = all.into_iter().find(|&&i| i >= n) {
            return Err(PyValueError::new_err(format!(
                "the graph has {n} inputs, got input {i}"
            )));
        }

        // Position lists become per-input masks for the compiler options.
        let mask = |positions: &[usize]| (0..n).map(|i| positions.contains(&i)).collect();
        let mut config = crate::graph_compiler::config::configs();
        config.mps.fuse = fuse.unwrap_or(config.mps.fuse);
        config.cuda.fuse = fuse.unwrap_or(config.cuda.fuse);

        let options = crate::graph_compiler::Options {
            config,
            donate,
            donate_into,
            parameters: parameters.as_deref().map(mask),
            packable: mask(&packable),
            scalars: mask(&scalars),
        };

        let inner =
            crate::graph_compiler::compile_with(&graph.inner, resolve_device(device)?, &options)
                .map_err(PyRuntimeError::new_err)?;

        Ok(PyPlan { inner })
    }

    /// Packed parameter blocks as `(member input positions, dimension they are packed along)`.
    #[getter]
    fn packed(&self) -> Vec<(Vec<usize>, usize)> {
        self.inner.packed().to_vec()
    }

    /// Per packed block, the region sizes its members are interleaved with.
    #[getter]
    fn interleaved(&self) -> Vec<Vec<usize>> {
        self.inner.interleaved().to_vec()
    }

    /// Runs an owned plan with its state in `workspace`, returning the outputs.
    fn run_in(
        &self,
        workspace: PyRef<'_, PyTensor>,
        inputs: Vec<PyRef<'_, PyTensor>>,
    ) -> PyResult<Vec<PyTensor>> {
        let inputs: Vec<_> = inputs.iter().map(|t| t.inner.clone()).collect();

        let outputs = self
            .inner
            .run_in(&workspace.inner, &inputs)
            .map_err(PyValueError::new_err)?;

        Ok(outputs.into_iter().map(PyTensor::wrap).collect())
    }

    /// Whether any step runs on the Apple Neural Engine.
    #[getter]
    fn neural_engine(&self) -> bool {
        self.inner.steps().iter().any(|s| {
            matches!(&s.primitive, Primitive::Fusion { name, .. } if name.starts_with(crate::graph::NEURAL_ENGINE))
        })
    }

    /// Whether a CUDA graph can capture this plan's steps: it runs as steps (not host and device
    /// stages), and none moves values to or from the host, waits on the host or another stream,
    /// calls out to a custom op or a Pallas kernel, or runs a collective.
    #[getter]
    fn capturable(&self) -> bool {
        self.inner.staged.is_none()
            && !self.inner.steps().iter().any(|s| {
                matches!(
                    s.primitive,
                    Primitive::HostWait
                        | Primitive::DeviceWait
                        | Primitive::ToHost
                        | Primitive::ToDevice
                        | Primitive::CustomCall { .. }
                        | Primitive::PallasCall { .. }
                        | Primitive::AllReduce { .. }
                        | Primitive::AllGather { .. }
                        | Primitive::ReduceScatter { .. }
                )
            })
    }

    /// Drops the cached Neural Engine models of this plan's steps (MPS builds only).
    fn _invalidate_neural_engine(&self) {
        #[cfg(lumen_mps_linked)]
        crate::graph_compiler::mps::ane::invalidate(self.inner.steps());
    }

    /// Bytes of device workspace the plan needs.
    #[getter]
    fn workspace_bytes(&self) -> usize {
        self.inner.workspace_bytes()
    }

    /// Byte alignment the workspace must have.
    #[getter]
    fn workspace_alignment(&self) -> usize {
        self.inner.workspace_alignment()
    }

    /// Each step as a dict of its primitive's fields plus buffers, types, views, scratch, scope and sources.
    fn steps<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let typed = |(buffer, ty): &(Buffer, TensorType)| {
            (buffer.to_string(), dtype_name(ty.dtype), ty.shape.clone())
        };

        self.inner
            .steps()
            .iter()
            .map(|step| {
                let d = primitive_dict(py, &step.primitive)?;
                d.set_item("inputs", step.inputs.iter().map(typed).collect::<Vec<_>>())?;
                d.set_item("output", typed(&step.output))?;
                let extra: Vec<_> = step.extra_outputs.iter().map(typed).collect();
                d.set_item("extra_outputs", extra)?;
                d.set_item("label", step.label)?;

                let views = step
                    .views
                    .iter()
                    .map(|v| v.as_ref().map(|v| (v.offset, v.strides.clone())));

                d.set_item("views", views.collect::<Vec<_>>())?;
                d.set_item("scratch", step.scratch)?;
                d.set_item("scope", scope_names(step.scope))?;
                d.set_item("sources", step.sources.clone())?;
                Ok(d)
            })
            .collect()
    }

    /// Runs the plan on `inputs`, on `device` if given, else on the inputs' device.
    #[pyo3(signature = (inputs, device = None))]
    fn run(
        &self,
        inputs: Vec<PyRef<'_, PyTensor>>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<PyTensor>> {
        let inputs: Vec<_> = inputs.iter().map(|t| t.inner.clone()).collect();

        let outputs = match device {
            Some(d) if !d.is_none() => self.inner.run_on(&inputs, resolve_device(Some(d))?),
            _ => self.inner.run(&inputs),
        }
        .map_err(PyValueError::new_err)?;

        Ok(outputs.into_iter().map(PyTensor::wrap).collect())
    }

    /// The plan's text listing of steps and buffers.
    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    /// The plan's text listing of steps and buffers.
    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

/// Starts recording kernel launches.
#[pyfunction]
fn _capture_launches() {
    crate::profiler::launched::capture();
}

/// A captured launch: step index, step label, kernel name, generated code, and Pallas
/// `(module, spec)` template if any.
type Captured = (
    usize,
    &'static str,
    String,
    Option<String>,
    Option<(String, String)>,
);

/// Launches recorded since `_capture_launches`.
#[pyfunction]
fn _captured_launches() -> Vec<Captured> {
    crate::profiler::launched::captured()
        .into_iter()
        .map(|l| (l.step, l.label, l.kernel, l.code, l.template))
        .collect()
}

/// Metal source of a generated fusion kernel by name, or `None` (always on non-MPS builds).
#[pyfunction]
fn _kernel_source(name: &str) -> Option<String> {
    #[cfg(lumen_mps_linked)]
    return super::mps::fusion_source(name);

    #[cfg(not(lumen_mps_linked))]
    return {
        let _ = name;
        None
    };
}

/// Triton helper functions as `(parameter, name, source)` triples for the Python launcher.
fn triton_functions(functions: &[super::triton::Function]) -> Vec<(&str, &str, &str)> {
    functions
        .iter()
        .map(|f| (f.param.as_str(), f.name.as_str(), f.source.as_str()))
        .collect()
}

/// Kernel source preceded by the helper functions passed to it, for launch capture.
fn triton_code(kernel: &str, functions: &[super::triton::Function]) -> String {
    let passed: Vec<String> = functions
        .iter()
        .map(|f| format!("# {} = {}\n{}", f.param, f.name, f.source))
        .collect();

    [passed.join("\n"), kernel.to_owned()].join("\n")
}

/// A kernel compiled for one of a step's launches: the kernel's name, launch config and template
/// constexprs, and the device and argument classes it was compiled for.
#[cfg(lumen_cuda_linked)]
struct Entry {
    name: String,
    config: [usize; 6],
    constexprs: Vec<(&'static str, super::triton::Constexpr)>,
    device: usize,
    classes: Vec<super::triton::direct::Class>,
    kernel: std::sync::Arc<super::triton::direct::Compiled>,
}

/// Launches a Triton kernel from Rust when it compiles to a plain launch, returning whether it did (or
/// had nothing to launch); otherwise the caller's Python launch runs it. The kernel is found in the
/// step's `launches` by its `name`, launch `config`, template `constexprs` and its `values`' classes (its
/// runtime parameters, in order, each run's own); the first launch of a class builds its `identity` (the
/// kernel, its constexprs and launch options) and compiles it with `compile`, the Python launcher's call
/// with `compile=True`.
#[cfg(lumen_cuda_linked)]
#[allow(clippy::too_many_arguments)]
fn direct(
    launches: &super::triton::Launches,
    name: &str,
    config: [usize; 6],
    constexprs: &[(&'static str, super::triton::Constexpr)],
    identity: impl FnOnce() -> String,
    on: &crate::Tensor,
    values: &[super::triton::direct::Value],
    grid: [usize; 3],
    compile: impl for<'py> FnOnce(Python<'py>, Bound<'py, PyDict>) -> PyResult<Py<PyAny>>,
) -> Result<bool, String> {
    use super::triton::direct;
    use std::sync::PoisonError;

    let Device::Cuda(device) = on.device() else {
        return Ok(false);
    };

    let found = launches
        .0
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .filter_map(|e| e.downcast_ref::<Entry>())
        .find(|e| {
            e.name == name
                && e.config == config
                && e.constexprs == constexprs
                && e.device == device
                && e.classes.len() == values.len()
                && e.classes.iter().zip(values).all(|(c, v)| *c == v.class())
        })
        .map(|e| e.kernel.clone());

    let kernel = match found {
        Some(kernel) => kernel,
        None => {
            let key = direct::key(&identity(), device, values);
            let mut nothing = false;

            let kernel = direct::compiled(&key, || {
                Python::attach(|py| {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("compile", true)?;
                    let info = compile(py, kwargs)?.into_bound(py);

                    if info.is_none() {
                        nothing = true;

                        return Ok(direct::Compiled {
                            function: 0,
                            warps: 0,
                            shared: 0,
                            plain: false,
                            types: Vec::new(),
                        });
                    }

                    let (function, warps, shared, plain, types): (
                        usize,
                        u32,
                        u32,
                        bool,
                        Vec<String>,
                    ) = info.extract()?;

                    Ok(direct::Compiled {
                        function,
                        warps,
                        shared,
                        plain,
                        types,
                    })
                })
                .map_err(|e: PyErr| e.to_string())
            })?;

            if nothing {
                return Ok(true);
            }

            launches
                .0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(std::sync::Arc::new(Entry {
                    name: name.to_owned(),
                    config,
                    constexprs: constexprs.to_vec(),
                    device,
                    classes: values.iter().map(|v| v.class()).collect(),
                    kernel: kernel.clone(),
                }));

            kernel
        }
    };

    if !kernel.plain {
        return Ok(false);
    }

    direct::launch(&kernel, grid, values)?;
    Ok(true)
}

/// The names of a launch's generated helper functions, which content-hash their sources.
#[cfg(lumen_cuda_linked)]
fn names(functions: &[super::triton::Function]) -> String {
    let names: Vec<&str> = functions.iter().map(|f| f.name.as_str()).collect();
    names.join(",")
}

/// Makes `t`'s CUDA device current before a launch.
fn current(t: &crate::Tensor) {
    if let Device::Cuda(index) = t.device() {
        crate::device::cuda::CUDA.set_device(index);
    }
}

/// Launches lumen's Triton GEMM on `[a, b, out]` with `sizes`, the tile `config`, and the fused epilogue.
fn triton_gemm(
    tensors: [&crate::Tensor; 3],
    sizes: [usize; 4],
    accum_dtype: DType,
    config: &super::triton::Config,
    functions: &[super::triton::Function],
    launches: &super::triton::Launches,
    epilogue: &super::triton::Epilogue,
) -> Result<(), String> {
    crate::profiler::launched::launched("dot_general", || {
        let template = format!(
            "# @triton.jit: lumen/codegen/dot_general/triton/gemm_kernel.py's kernel, BLOCK_SIZE_M={} BLOCK_SIZE_N={} BLOCK_SIZE_K={} GROUP_SIZE_M={} stages={} warps={}\n",
            config.block_m,
            config.block_n,
            config.block_k,
            config.group_m,
            config.stages,
            config.warps
        );

        Some(triton_code(&template, functions))
    });

    #[cfg(not(lumen_cuda_linked))]
    let _ = launches;

    current(tensors[2]);

    let call = |py: Python<'_>, kwargs: Option<Bound<'_, PyDict>>| -> PyResult<Py<PyAny>> {
        let tensors = tensors.map(|t| PyTensor::wrap(t.clone()));
        let extra: Vec<PyTensor> = epilogue
            .tensors
            .iter()
            .map(|t| PyTensor::wrap(t.clone()))
            .collect();
        py.import("lumen.codegen.dot_general.triton.gemm_kernel")?
            .getattr("dot_general")?
            .call(
                (
                    tensors,
                    sizes,
                    dtype_name(accum_dtype),
                    (
                        config.block_m,
                        config.block_n,
                        config.block_k,
                        config.group_m,
                        config.stages,
                        config.warps,
                    ),
                    triton_functions(functions),
                    extra,
                    epilogue.scalars,
                    (
                        epilogue.flags.group,
                        epilogue.flags.interleaved,
                        epilogue.flags.rows,
                    ),
                ),
                kwargs.as_ref(),
            )
            .map(Bound::unbind)
    };

    #[cfg(lumen_cuda_linked)]
    {
        use super::triton::direct::Value;
        let [l, m, _, n] = sizes;
        let flags = epilogue.flags;
        let halves = flags.group > 1 && !flags.interleaved;
        let grid_m = if halves && flags.rows {
            m / flags.group
        } else {
            m
        };
        let grid_n = if halves && !flags.rows {
            n / flags.group
        } else {
            n
        };
        let programs = grid_m.div_ceil(config.block_m) * grid_n.div_ceil(config.block_n);
        let extra =
            epilogue
                .tensors
                .iter()
                .enumerate()
                .map(|(k, t)| match epilogue.scalars.contains(&k) {
                    true => Value::scalar(t),
                    false => Value::of(t),
                });
        let values: Vec<Value> = tensors
            .iter()
            .map(|t| Value::of(t))
            .chain(extra)
            .chain(sizes[1..].iter().map(|&s| Value::Int(s as i128)))
            .collect();
        let identity = || {
            format!(
                "gemm|{accum_dtype}|{config:?}|{:?}|{:?}|{}",
                (flags.group, flags.interleaved, flags.rows),
                epilogue.scalars,
                names(functions)
            )
        };
        let tiles = [
            config.block_m,
            config.block_n,
            config.block_k,
            config.group_m,
            config.stages,
            config.warps,
        ];
        let grid = [l, programs, 1];
        if direct(
            launches,
            "gemm",
            tiles,
            &[],
            identity,
            tensors[2],
            &values,
            grid,
            |py, kw| call(py, Some(kw)),
        )? {
            return Ok(());
        }
    }

    Python::attach(|py| call(py, None).map(|_| ())).map_err(|e| e.to_string())
}

/// Launches a generated Triton elementwise loop kernel over `n` elements in `programs` programs.
#[allow(clippy::too_many_arguments)]
fn triton_loop(
    kernel: &super::triton::Kernel,
    config: &crate::codegen::elementwise::triton::LoopConfig,
    functions: &[super::triton::Function],
    launches: &super::triton::Launches,
    inputs: &[crate::Tensor],
    out: &crate::Tensor,
    n: usize,
    programs: usize,
) -> Result<(), String> {
    crate::profiler::launched::launched(&kernel.name, || {
        let source = match kernel.template {
            Some(template) => {
                let (module, name) = template.kernel();

                format!(
                    "# @triton.jit: {module}'s {name}, BLOCK_SIZE={} warps={}\n",
                    config.block_size, config.warps
                )
            }
            None => format!(
                "# BLOCK_SIZE={} warps={}\n{}",
                config.block_size, config.warps, kernel.source
            ),
        };

        Some(triton_code(&source, functions))
    });

    let constexprs = kernel
        .template
        .map(|t| t.constexprs(n, &kernel.segments))
        .unwrap_or_default();

    #[cfg(not(lumen_cuda_linked))]
    let _ = launches;

    current(out);

    let call = |py: Python<'_>, kwargs: Option<Bound<'_, PyDict>>| -> PyResult<Py<PyAny>> {
        let inputs: Vec<PyTensor> = inputs.iter().map(|t| PyTensor::wrap(t.clone())).collect();
        // Inputs passed by value (scalars) are named by position for the launcher.
        let scalars: Vec<usize> = (0..kernel.by_value.len())
            .filter(|&k| kernel.by_value[k])
            .collect();

        // A template kernel takes the tensors as its `EXTRA` tuple and its constexprs by name.
        if let Some(template) = kernel.template {
            let (module, name) = template.kernel();

            let tensors: Vec<PyTensor> = inputs
                .into_iter()
                .chain([PyTensor::wrap(out.clone())])
                .collect();

            let values = PyDict::new(py);

            for (key, value) in &constexprs {
                match value {
                    super::triton::Constexpr::Int(v) => values.set_item(key, v)?,
                    super::triton::Constexpr::Ints(v) => {
                        values.set_item(key, pyo3::types::PyTuple::new(py, v)?)?
                    }
                    super::triton::Constexpr::DType(dtype) => {
                        let name = crate::codegen::utils::triton::triton_type(*dtype)
                            .expect("a Triton dtype")
                            .trim_start_matches("tl.");

                        values.set_item(key, py.import("triton.language")?.getattr(name)?)?
                    }
                }
            }

            return py
                .import("lumen.codegen.utils.triton")?
                .getattr("template")?
                .call(
                    (
                        module,
                        name,
                        tensors,
                        scalars,
                        triton_functions(functions),
                        values,
                        programs,
                        config.block_size,
                        config.warps,
                    ),
                    kwargs.as_ref(),
                )
                .map(Bound::unbind);
        }

        py.import("lumen.codegen.utils.triton")?
            .getattr("launch")?
            .call(
                (
                    kernel.name.as_str(),
                    kernel.source.as_str(),
                    inputs,
                    scalars,
                    PyTensor::wrap(out.clone()),
                    triton_functions(functions),
                    n,
                    programs,
                    config.block_size,
                    config.warps,
                ),
                kwargs.as_ref(),
            )
            .map(Bound::unbind)
    };

    #[cfg(lumen_cuda_linked)]
    {
        use super::triton::direct::Value;
        let scalar = |k: usize| kernel.by_value.get(k).copied().unwrap_or(false);
        let values: Vec<Value> = inputs
            .iter()
            .enumerate()
            .map(|(k, t)| match scalar(k) {
                true => Value::scalar(t),
                false => Value::of(t),
            })
            .chain([Value::of(out)])
            // A template kernel's element counts are constexprs; a generated one takes `n`.
            .chain(kernel.template.is_none().then_some(Value::Int(n as i128)))
            .collect();
        let identity = || {
            format!(
                "loop|{}|{}|{}|{}|{:?}",
                kernel.name,
                config.block_size,
                config.warps,
                names(functions),
                constexprs
            )
        };
        let tiles = [config.block_size, config.warps, 0, 0, 0, 0];
        if direct(
            launches,
            &kernel.name,
            tiles,
            &constexprs,
            identity,
            out,
            &values,
            [programs, 1, 1],
            |py, kw| call(py, Some(kw)),
        )? {
            return Ok(());
        }
    }

    Python::attach(|py| call(py, None).map(|_| ())).map_err(|e| e.to_string())
}

/// Launches one of lumen's Triton flash-attention kernels (`launch.kernel`) with its tile config.
fn triton_attention(launch: &super::triton::Launch) -> Result<(), String> {
    let config = launch.config;

    crate::profiler::launched::launched("flash_attention", || {
        let template = format!(
            "# @triton.jit: lumen/codegen/attention/triton's {} kernel, BLOCK_SIZE_M={} BLOCK_SIZE_N={} stages={} warps={}\n",
            launch.kernel, config.block_m, config.block_n, config.stages, config.warps
        );

        Some(triton_code(&template, launch.functions))
    });

    let device = launch
        .tensors
        .iter()
        .flatten()
        .next()
        .ok_or("an attention reads tensors")?;

    current(device);

    let call = |py: Python<'_>, kwargs: Option<Bound<'_, PyDict>>| -> PyResult<Py<PyAny>> {
        let tensors: Vec<Option<PyTensor>> = launch
            .tensors
            .iter()
            .map(|t| t.as_ref().map(|t| PyTensor::wrap(t.clone())))
            .collect();
        py.import("lumen.codegen.attention.triton.kernel")?
            .getattr("attention")?
            .call(
                (
                    launch.kernel,
                    tensors,
                    triton_functions(launch.functions),
                    launch.sizes,
                    launch.causal,
                    launch.scale,
                    (config.block_m, config.block_n, config.stages, config.warps),
                ),
                kwargs.as_ref(),
            )
            .map(Bound::unbind)
    };

    #[cfg(lumen_cuda_linked)]
    {
        use super::triton::direct::Value;
        let [batch, heads, sq, sk, h, hv] = launch.sizes;
        // The dK/dV kernel always takes a dQ; one it doesn't read, like any absent tensor, is the first.
        let mut tensors: Vec<Option<&crate::Tensor>> =
            launch.tensors.iter().map(Option::as_ref).collect();
        if launch.kernel == "dkdv" && tensors.len() == 8 {
            tensors.push(None);
        }
        let absent: Vec<bool> = tensors.iter().map(Option::is_none).collect();
        let mut values: Vec<Value> = tensors
            .iter()
            .map(|t| Value::of(t.unwrap_or(device)))
            .collect();
        // The forward kernel's `EXTRA` tuple, flattened, holds at least the first tensor.
        if launch.kernel == "forward" && values.len() == 6 {
            values.push(Value::of(device));
        }
        values.extend([sq, sk].map(|s| Value::Int(s as i128)));
        values.push(Value::Int(launch.causal.unwrap_or(0) as i128));
        if launch.kernel != "forward" {
            values.push(Value::Float(launch.scale));
        }
        let blocks = match launch.kernel {
            "dkdv" => sk.div_ceil(config.block_n),
            _ => sq.div_ceil(config.block_m),
        };
        let identity = || {
            format!(
                "attention|{}|{h},{hv}|{}|{absent:?}|{config:?}|{}",
                launch.kernel,
                launch.causal.is_some(),
                names(launch.functions)
            )
        };
        let tiles = [
            config.block_m,
            config.block_n,
            config.stages,
            config.warps,
            0,
            0,
        ];
        if direct(
            launch.launches,
            launch.kernel,
            tiles,
            &[],
            identity,
            device,
            &values,
            [blocks, heads, batch],
            |py, kw| call(py, Some(kw)),
        )? {
            return Ok(());
        }
    }

    Python::attach(|py| call(py, None).map(|_| ())).map_err(|e| e.to_string())
}

/// Registers the Triton launch hooks and the module's Python functions and class.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    super::triton::set_gemm(triton_gemm);
    super::triton::set_loop(triton_loop);
    super::triton::set_attention(triton_attention);
    m.add_function(wrap_pyfunction!(_capture_launches, m)?)?;
    m.add_function(wrap_pyfunction!(_captured_launches, m)?)?;
    m.add_function(wrap_pyfunction!(_kernel_source, m)?)?;
    m.add_class::<PyPlan>()
}
