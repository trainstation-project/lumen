//! The compiler's configuration: one set of settings every device's compiler reads (a backend reads the
//! flags it has a use for and ignores the rest), the global settings store, and its Python view
//! (`lumen.config.compiler`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{PoisonError, RwLock};

/// Defines [`CompilerConfig`] with the listed flags and their defaults, and (with `python`) its Python
/// class, a getter and setter per flag reading and writing the global store.
macro_rules! compiler_config {
    ($($(#[doc = $doc:literal])* $flag:ident, $set:ident: $ty:ty = $default:expr;)*) => {
        /// The compiler's settings, read by every device's compiler.
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct CompilerConfig {
            $($(#[doc = $doc])* pub $flag: $ty,)*
        }

        impl Default for CompilerConfig {
            /// Every flag at its default.
            fn default() -> Self {
                CompilerConfig {
                    $($flag: $default,)*
                }
            }
        }

        /// Python view of the compiler's settings (`lumen.config.compiler`), reading and writing the
        /// global store.
        #[cfg(feature = "python")]
        #[pyo3::pyclass(name = "CompilerConfig", module = "lumen")]
        pub(crate) struct PyCompilerConfig;

        #[cfg(feature = "python")]
        #[pyo3::pymethods]
        impl PyCompilerConfig {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $flag(&self) -> $ty {
                    config().$flag
                }

                #[setter]
                fn $set(&self, value: $ty) {
                    update(|c| c.$flag = value);
                }
            )*

            /// Restores every setting to its default.
            fn reset(&self) {
                update(|c| *c = CompilerConfig::default());
            }

            /// Counter bumped on every config change, used to invalidate compiled caches.
            #[getter]
            fn _version(&self) -> u64 {
                version()
            }

            /// All the settings, as Python keyword arguments.
            fn __repr__(&self) -> String {
                let c = config();
                let flags: Vec<String> =
                    vec![$(format!("{}={}", stringify!($flag), py_value(&c.$flag))),*];

                format!("{}.config.compiler({})", crate::LIBRARY_NAME, flags.join(", "))
            }
        }
    };
}

compiler_config! {
    /// Merge dots that share an operand.
    merge_dots, set_merge_dots: bool = true;
    /// Fuse attention into flash-attention kernels.
    flash_attention, set_flash_attention: bool = true;
    /// Rows per chunk of the fused linear cross entropy; `None` disables chunking.
    fused_linear_cross_entropy_chunk_size, set_fused_linear_cross_entropy_chunk_size: Option<usize> = None;
    /// Byte budget for the plan's memory; 0 for unlimited.
    memory_limit, set_memory_limit: usize = 0;
    /// Directory to dump the graph to after each pass.
    dump_to, set_dump_to: Option<String> = None;
    /// Drop a value's round trip through another float dtype where the forward reads it, the narrower
    /// dtype kept for the gradients' reads (off: the program runs as traced).
    preserve_precision, set_preserve_precision: bool = false;
    /// Run CUDA programs through the Triton backend.
    use_triton_backend, set_use_triton_backend: bool = false;
    /// Fuse elementwise chains, reductions and epilogues into kernels.
    fuse, set_fuse: bool = true;
    /// Time candidate kernel configurations and keep the fastest.
    autotune, set_autotune: bool = true;
    /// Clamp indices kernels read (off: indices are read as given).
    safe_kernels, set_safe_kernels: bool = false;
    /// Merge independent small kernels into one launch.
    horizontal_fusion, set_horizontal_fusion: bool = true;
    /// Write a fusion's values read elsewhere as extra outputs.
    multi_output_fusion, set_multi_output_fusion: bool = true;
    /// Split long contractions of skinny matmuls across threadgroups.
    split_k, set_split_k: bool = true;
    /// Move a factor of a matmul's operand from a reduction over rows wider than a GEMM's tile to its
    /// output (CODA's reordering; rounding differs from the traced program).
    linear_reordering, set_linear_reordering: bool = false;
    /// Fuse normalization diamonds (row reductions broadcast back over their rows) into row kernels.
    normalization_diamonds, set_normalization_diamonds: bool = true;
    /// Fuse a cross entropy of a matmul into one kernel that never writes the logits.
    fused_linear_cross_entropy, set_fused_linear_cross_entropy: bool = true;
    /// Pick kernels whose results don't depend on scheduling (sorted scatter-adds).
    deterministic, set_deterministic: bool = false;
    /// Capture a CUDA plan's launches into a CUDA graph and replay it.
    cuda_graphs, set_cuda_graphs: bool = true;
    /// Fuse elementwise epilogues into reductions.
    reduction_epilogues, set_reduction_epilogues: bool = true;
    /// Fuse elementwise epilogues into matmuls and attention.
    contraction_epilogues, set_contraction_epilogues: bool = true;
    /// Compute a softmax's max and sum in one rescaled pass (rounding differs from the traced program).
    online_softmax, set_online_softmax: bool = true;
    /// Run fusions the Apple Neural Engine can on it.
    neural_engine, set_neural_engine: bool = false;
    /// Row elements a thread keeps in registers between a row kernel's passes.
    row_cache, set_row_cache: usize = 8;
}

impl CompilerConfig {
    /// Byte budget for the compiled plan's memory; `None` when unlimited.
    pub fn memory_limit(&self) -> Option<usize> {
        (self.memory_limit > 0).then_some(self.memory_limit)
    }

    /// Directory to dump the graph to after each pass.
    pub fn dump_to(&self) -> Option<&str> {
        self.dump_to.as_deref()
    }
}

/// Adds the `CompilerConfig` class to `lumen.graph_compiler.config._C`.
#[cfg(feature = "python")]
pub(crate) fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;

    m.add_class::<PyCompilerConfig>()
}

/// `value`'s Debug text as Python source (`True`, `False`, `Some(x)` as `x`).
#[cfg(feature = "python")]
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

/// The current config; `None` until first changed.
static CONFIG: RwLock<Option<CompilerConfig>> = RwLock::new(None);

/// Bumped on every change, so compiled-graph caches can tell when the config changed.
static VERSION: AtomicU64 = AtomicU64::new(0);

/// Number of config changes so far.
pub fn version() -> u64 {
    VERSION.load(Ordering::Acquire)
}

/// A snapshot of the current config.
pub fn config() -> CompilerConfig {
    let config = CONFIG.read().unwrap_or_else(PoisonError::into_inner);
    config.clone().unwrap_or_default()
}

/// Applies `change` to the config and bumps the version.
pub fn update(change: impl FnOnce(&mut CompilerConfig)) {
    let mut config = CONFIG.write().unwrap_or_else(PoisonError::into_inner);
    let mut next = config.clone().unwrap_or_default();
    change(&mut next);
    *config = Some(next);
    VERSION.fetch_add(1, Ordering::Release);
}
