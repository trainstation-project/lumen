//! `lumen.config.compiler`: bindings for [`crate::compiler::config`], the
//! flags graphs are compiled with.

use pyo3::prelude::*;

use super::config::{self, CompilerConfig};

/// The graph compilers' flags (``lumen.config.compiler``), read when a
/// graph is compiled: ``lumen.compile`` compiles again once they change.
/// The defaults run every program exactly as traced, but for
/// ``online_softmax`` and ``flash_attention`` (on).
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
    /// Fuse the elementwise primitives after a contraction (a dot, an
    /// attention: a bias, a residual, an activation, a cast) into its
    /// kernel.
    contraction_epilogues, set_contraction_epilogues: bool;
    /// Compute an expensive value several fusions read in the first, as
    /// another output of its kernel.
    multi_output_fusion, set_multi_output_fusion: bool;
    /// Independent small loop fusions of as many elements as one kernel
    /// (XLA's horizontal loop fusion).
    horizontal_fusion, set_horizontal_fusion: bool;
    /// A softmax's max and sum in one pass of its row kernel (online
    /// softmax; on by default): not what the program computes, its rounding
    /// differs; off runs softmax exactly as traced.
    online_softmax, set_online_softmax: bool;
    /// Attention, however written, as one flash-attention kernel (on by
    /// default): not what the program computes, its rounding differs; off
    /// runs attention exactly as traced.
    flash_attention, set_flash_attention: bool;
    /// A cross entropy of a matmul's logits nothing else reads, however
    /// written, as one linear cross entropy, the logits never stored (on by
    /// default): not what the program computes, its rounding differs; off
    /// runs it exactly as traced.
    fused_linear_cross_entropy, set_fused_linear_cross_entropy: bool;
    /// A dot of few output tiles and a long contraction split along it
    /// across threadgroups (on by default): not what the program computes,
    /// the partials are added in another order; off runs dots exactly as
    /// traced.
    split_k, set_split_k: bool;
    /// Run every kernel deterministically (off by default: a kernel may add
    /// atomically, in no fixed order: attention's backward's dQ, split-K).
    deterministic, set_deterministic: bool;
    /// On MPS, run large float16 dots of a compiled function's fixed weights
    /// (and the float16 work after them) on the Apple Neural Engine, through
    /// Core ML (off by default): not what the program computes, the Neural
    /// Engine accumulates its dots narrower than float32; off runs every
    /// dot on lumen's kernels.
    neural_engine, set_neural_engine: bool;
    /// The most elements of a row each thread of a row kernel keeps in
    /// registers between passes (0: none).
    row_cache, set_row_cache: usize;
    /// The most workspace a compiled function should need, in bytes: above
    /// it, values are recomputed where they are read later rather than kept
    /// alive (rematerialization), until it fits or nothing more can be
    /// saved (0, the default: no limit).
    memory_limit, set_memory_limit: usize;
    /// The rows ``F.linear_cross_entropy`` takes at a time, its gradients
    /// computed with its losses a chunk at a time (XMA's chunked fused
    /// linear cross entropy); None, the default: not chunked.
    fused_linear_cross_entropy_chunk_size, set_fused_linear_cross_entropy_chunk_size: Option<usize>;
    /// Clamp the indices a gather, a scatter-add or a linear cross entropy
    /// reads at into range before its kernel does (off by default: an index
    /// out of range is the program's error, read as given).
    safe_kernels, set_safe_kernels: bool;
}

/// A flag's value as Python writes it.
fn py_value(value: &dyn std::fmt::Debug) -> String {
    match format!("{value:?}").as_str() {
        "true" => "True".into(),
        "false" => "False".into(),
        v => v
            .strip_prefix("Some(")
            .and_then(|v| v.strip_suffix(')'))
            .unwrap_or(v)
            .into(),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCompilerConfig>()
}
