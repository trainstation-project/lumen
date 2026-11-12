//! Python bindings of `lumen.autotune`: the kernel config classes a `lumen.autotune.use` block forces in place of
//! autotuning (each defined next to its kernel's config, most with `config_class!`), and the block's entry and exit.

use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use super::forced::{self, Forced};
use crate::codegen::attention::cuda::PyCudaAttentionConfig;
#[cfg(lumen_mps_linked)]
use crate::codegen::attention::mps::PyMpsAttentionConfig;
use crate::codegen::attention::triton::PyTritonAttentionConfig;
use crate::codegen::dot_general::cuda::PyCudaGemmConfig;
#[cfg(lumen_mps_linked)]
use crate::codegen::dot_general::mps::PyMpsGemmConfig;
use crate::codegen::dot_general::triton::PyTritonGemmConfig;
#[cfg(lumen_mps_linked)]
use crate::codegen::elementwise::mps::PyMpsElementwiseConfig;
use crate::codegen::elementwise::triton::PyTritonLoopConfig;

/// `value`'s Debug text as Python source (`true` as `True`, `false` as `False`).
pub(crate) fn py_repr(value: &dyn std::fmt::Debug) -> String {
    match format!("{value:?}").as_str() {
        "true" => "True".into(),
        "false" => "False".into(),
        v => v.into(),
    }
}

/// Defines `$py`, the Python class `lumen.autotune.$name` wrapping kernel config `$inner`, a struct of the listed
/// fields (each with an optional constructor default): its constructor, a getter per field, its repr, and the
/// extra methods given (its `candidates`).
macro_rules! config_class {
    (
        $(#[doc = $doc:literal])*
        $py:ident($inner:ident) = $name:literal {
            $($field:ident: $ty:ty $(= $default:expr)?),* $(,)?
        }
        $($extra:tt)*
    ) => {
        $(#[doc = $doc])*
        #[pyo3::pyclass(name = $name, module = "lumen.autotune", frozen, eq, from_py_object)]
        #[derive(Clone, PartialEq)]
        pub(crate) struct $py(pub(crate) $inner);

        #[pyo3::pymethods]
        impl $py {
            /// The config of these fields.
            #[new]
            #[pyo3(signature = ($($field $(= $default)?),*))]
            fn new($($field: $ty),*) -> Self {
                $py($inner { $($field),* })
            }

            $(
                /// The config's field of this name.
                #[getter]
                fn $field(&self) -> $ty {
                    self.0.$field
                }
            )*

            /// The config as its constructor call.
            fn __repr__(&self) -> String {
                let fields: Vec<String> = vec![
                    $(format!("{}={}", stringify!($field), crate::autotune::python::py_repr(&self.0.$field))),*
                ];

                format!("{}({})", $name, fields.join(", "))
            }

            $($extra)*
        }
    };
}

pub(crate) use config_class;

/// The forced config Python object `config` stands for: any kernel config class of `lumen.autotune`.
fn forced_of(config: &Bound<'_, PyAny>) -> PyResult<Forced> {
    macro_rules! try_class {
        ($($class:ty),*) => {
            $(
                if let Ok(c) = config.extract::<$class>() {
                    return Ok(Arc::new(c.0) as Forced);
                }
            )*
        };
    }

    try_class!(
        PyCudaAttentionConfig,
        PyCudaGemmConfig,
        PyTritonGemmConfig,
        PyTritonAttentionConfig,
        PyTritonLoopConfig
    );

    // The Metal kernels' configs exist in a build with Metal.
    #[cfg(lumen_mps_linked)]
    try_class!(
        PyMpsGemmConfig,
        PyMpsAttentionConfig,
        PyMpsElementwiseConfig
    );

    Err(PyTypeError::new_err(format!(
        "lumen.autotune.use takes kernel configs (lumen.autotune's config classes), got {}",
        config.get_type().name()?
    )))
}

/// Enters a `lumen.autotune.use` block forcing `configs` (at most one of each kernel family's type).
#[pyfunction]
fn _push(configs: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
    let forced = configs
        .iter()
        .map(forced_of)
        .collect::<PyResult<Vec<_>>>()?;
    forced::push(forced);
    Ok(())
}

/// Leaves the innermost `lumen.autotune.use` block.
#[pyfunction]
fn _pop() {
    forced::pop();
}

/// Adds the config classes (the Metal kernels' in a build with Metal) and the block's entry and exit to
/// `lumen.autotune._C`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCudaAttentionConfig>()?;
    m.add_class::<PyCudaGemmConfig>()?;
    m.add_class::<PyTritonGemmConfig>()?;
    m.add_class::<PyTritonAttentionConfig>()?;
    m.add_class::<PyTritonLoopConfig>()?;

    #[cfg(lumen_mps_linked)]
    {
        m.add_class::<PyMpsGemmConfig>()?;
        m.add_class::<PyMpsAttentionConfig>()?;
        m.add_class::<PyMpsElementwiseConfig>()?;
    }

    m.add_function(wrap_pyfunction!(_push, m)?)?;
    m.add_function(wrap_pyfunction!(_pop, m)?)
}
