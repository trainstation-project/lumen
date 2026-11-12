//! Matmul kernel configuration shared by every Metal GEMM algorithm.

/// Which Metal matmul kernel a config runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Algorithm {
    /// Simdgroup-matrix tiles staged through threadgroup memory (`matmul_sg`).
    Tiled,
    /// Metal Performance Primitives tensor ops (`matmul_mpp`).
    Mpp,
    /// Few output rows: lanes split K and reduce partial sums (`matmul_skinny`).
    Skinny,
    /// Matrix-vector product with a simd reduction per output (`matmul_gemv`).
    Gemv,
}

/// Tile shape, thread count and algorithm of one matmul kernel variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmConfig {
    /// Output tile rows per threadgroup.
    pub bm: usize,
    /// Output tile columns per threadgroup.
    pub bn: usize,
    /// K elements consumed per step (for GEMV, elements each lane loads per step).
    pub bk: usize,
    /// Threads per threadgroup.
    pub threads: usize,
    /// Simdgroup columns the tile is split into (tiled algorithm).
    pub cols: usize,
    /// Whether the tiled kernel prefetches the next K tile while computing.
    pub prefetch: bool,
    pub algorithm: Algorithm,
}

/// Threadgroup memory available to one matmul threadgroup, in bytes.
pub(crate) const MAX_THREADGROUP_BYTES: usize = 32 << 10;

/// `lumen.autotune.MpsGemmConfig`: one Metal matmul kernel variant, its `algorithm` "tiled" (simdgroup matrices),
/// "mpp" (Metal Performance Primitives), "skinny" (few output rows) or "gemv" (matrix-vector).
#[cfg(feature = "python")]
#[pyo3::pyclass(
    name = "MpsGemmConfig",
    module = "lumen.autotune",
    frozen,
    eq,
    from_py_object
)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyMpsGemmConfig(pub(crate) GemmConfig);

/// The Python name of each algorithm.
#[cfg(feature = "python")]
const ALGORITHMS: [(&str, Algorithm); 4] = [
    ("tiled", Algorithm::Tiled),
    ("mpp", Algorithm::Mpp),
    ("skinny", Algorithm::Skinny),
    ("gemv", Algorithm::Gemv),
];

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl PyMpsGemmConfig {
    /// The config of a `bm` x `bn` tile stepping `bk` along K on `threads` threads, `cols` simdgroup columns, the
    /// next K tile prefetched or not, run by `algorithm`.
    #[new]
    #[pyo3(signature = (bm, bn, bk, threads, cols=1, prefetch=false, algorithm="tiled"))]
    fn new(
        bm: usize,
        bn: usize,
        bk: usize,
        threads: usize,
        cols: usize,
        prefetch: bool,
        algorithm: &str,
    ) -> pyo3::PyResult<Self> {
        let Some(&(_, algorithm)) = ALGORITHMS.iter().find(|(name, _)| *name == algorithm) else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "MpsGemmConfig: algorithm is one of tiled, mpp, skinny, gemv, got {algorithm}"
            )));
        };

        Ok(PyMpsGemmConfig(GemmConfig {
            bm,
            bn,
            bk,
            threads,
            cols,
            prefetch,
            algorithm,
        }))
    }

    /// Every config the autotuner can time (each fits a given matmul only where its kernel takes it), or those of
    /// `algorithm`.
    #[staticmethod]
    #[pyo3(signature = (algorithm=None))]
    fn candidates(algorithm: Option<&str>) -> Vec<Self> {
        let all = super::config()
            .iter()
            .chain(super::mpp_configs())
            .chain(super::skinny_configs())
            .chain(super::gemv_configs(true))
            .chain(super::gemv_configs(false));

        all.filter(|c| algorithm.is_none_or(|a| PyMpsGemmConfig(**c).algorithm() == a))
            .map(|c| PyMpsGemmConfig(*c))
            .collect()
    }

    /// Output tile rows per threadgroup.
    #[getter]
    fn bm(&self) -> usize {
        self.0.bm
    }

    /// Output tile columns per threadgroup.
    #[getter]
    fn bn(&self) -> usize {
        self.0.bn
    }

    /// K elements consumed per step.
    #[getter]
    fn bk(&self) -> usize {
        self.0.bk
    }

    /// Threads per threadgroup.
    #[getter]
    fn threads(&self) -> usize {
        self.0.threads
    }

    /// Simdgroup columns the tile is split into.
    #[getter]
    fn cols(&self) -> usize {
        self.0.cols
    }

    /// Whether the next K tile is prefetched.
    #[getter]
    fn prefetch(&self) -> bool {
        self.0.prefetch
    }

    /// The kernel that runs it: "tiled", "mpp", "skinny" or "gemv".
    #[getter]
    fn algorithm(&self) -> &'static str {
        ALGORITHMS
            .iter()
            .find(|(_, a)| *a == self.0.algorithm)
            .map_or("tiled", |(name, _)| name)
    }

    /// The config as its constructor call.
    fn __repr__(&self) -> String {
        let c = &self.0;
        format!(
            "MpsGemmConfig(bm={}, bn={}, bk={}, threads={}, cols={}, prefetch={}, algorithm={:?})",
            c.bm,
            c.bn,
            c.bk,
            c.threads,
            c.cols,
            crate::autotune::python::py_repr(&c.prefetch),
            self.algorithm()
        )
    }
}
