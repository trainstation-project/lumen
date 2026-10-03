//! `lumen.config.compiler`: bindings for [`crate::compiler::config`], the
//! flags graphs are compiled with.

use pyo3::prelude::*;

use super::config::{self, CompilerConfig};

/// The graph compilers' flags (``lumen.config.compiler``), read when a
/// graph is compiled: ``lumen.compile`` compiles again once they change.
/// The defaults run every program exactly as traced.
#[pyclass(name = "CompilerConfig", module = "lumen")]
pub(crate) struct PyCompilerConfig;

/// A getter and a setter on [`PyCompilerConfig`] for each flag.
macro_rules! flags {
    ($($(#[$doc:meta])* $name:ident, $set:ident: $ty:ty;)*) => {
        #[pymethods]
        impl PyCompilerConfig {
            $(
                $(#[$doc])*
                #[getter]
                fn $name(&self) -> $ty {
                    config::config().$name
                }

                #[setter]
                fn $set(&self, value: $ty) {
                    config::update(|c| c.$name = value);
                }
            )*

            /// Every flag back to its default.
            fn reset(&self) {
                config::update(|c| *c = CompilerConfig::default());
            }

            fn __repr__(&self) -> String {
                let c = config::config();
                let flags: Vec<String> = vec![$(format!("{}={}", stringify!($name), py_value(&c.$name))),*];
                format!("{}.config.compiler({})", crate::LIBRARY_NAME, flags.join(", "))
            }
        }
    };
}

flags! {
    /// Fuse primitives into generated kernels (where the device does).
    fuse, set_fuse: bool;
    /// Merge dots sharing an operand (XLA's DotMerger); needs ``fuse``.
    merge_dots, set_merge_dots: bool;
    /// Run normalization diamonds (softmax, RMS and layer norms, however
    /// written) as one row kernel each.
    normalization_diamonds, set_normalization_diamonds: bool;
    /// Fuse the elementwise primitives after a reduction into its kernel.
    reduction_epilogues, set_reduction_epilogues: bool;
    /// Compute an expensive value several fusions read in the first, as
    /// another output of its kernel.
    multi_output_fusion, set_multi_output_fusion: bool;
    /// A softmax's max and sum in one pass of its row kernel (online
    /// softmax): not what the program computes, its rounding differs.
    online_softmax, set_online_softmax: bool;
    /// The most elements of a row each thread of a row kernel keeps in
    /// registers between passes (0: none).
    row_cache, set_row_cache: usize;
}

/// A flag's value as Python writes it.
fn py_value(value: &dyn std::fmt::Display) -> String {
    match value.to_string().as_str() {
        "true" => "True".into(),
        "false" => "False".into(),
        v => v.into(),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCompilerConfig>()
}
