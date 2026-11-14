//! Matmul kernel configuration shared by every Metal GEMM algorithm.
//!
//! Autotuning the Metal GEMMs: tiled, Metal Performance Primitives, skinny and GEMV configs for GEMM fusions.

use crate::autotune::Kernels;
use crate::codegen::dot_general::GemmTuning;
use crate::codegen::dot_general::mps as gemm;
use crate::codegen::kinds::gemm_dot;
use crate::codegen::utils::mps::Mps;
use crate::graph::{Primitive, TensorType, Var};
use crate::graph_compiler::CompilerConfig;
use crate::plan::step::mps::Encode;
use crate::plan::{Buffer, MpsStep, Tuned};
use crate::runtime::kernel::mps::{compile_kernels, compile_mpp_kernels, uncompiled};
use crate::tensor::contiguous_strides;
use std::collections::BTreeMap;

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

/// Tuning backend for GEMM fusions.
#[derive(Default)]
pub(crate) struct Gemms {
    sources: BTreeMap<String, String>,
    config: CompilerConfig,
}

/// Creates an empty GEMM tuner for `config`.
impl Gemms {
    pub(crate) fn new(config: &CompilerConfig) -> Self {
        Gemms {
            sources: BTreeMap::new(),
            config: config.clone(),
        }
    }
}

/// Config used for GEMMs whose epilogue writes row tiles to scratch.
const ROW_TILES: GemmConfig = GemmConfig {
    bm: gemm::ROW_TILE,
    bn: gemm::ROW_TILE,
    bk: 32,
    threads: 256,
    cols: 1,
    prefetch: false,
    algorithm: gemm::Algorithm::Tiled,
};

/// Output size, in elements, from which a GEMM counts as large enough to tune.
const TUNED_OUTPUTS: usize = 64 << 10;

/// The dot of a GEMM fusion with its lhs, rhs and output types.
pub(crate) fn matmul_of(
    step: &MpsStep,
) -> Option<(&Primitive, &TensorType, &TensorType, &TensorType)> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        return None;
    };

    let dot = gemm_dot(body)?;
    let ty = |v| body.type_of(v);

    Some((
        &dot.primitive,
        ty(dot.inputs[0]),
        ty(dot.inputs[1]),
        ty(dot.output),
    ))
}

/// Whether `step` is a float GEMM fusion that the tiled GEMM kernels can run, with views only on its dot's
/// operands (read through their strides).
fn tiled_gemm(step: &MpsStep) -> bool {
    let float = step.output.1.dtype.is_float();
    let views = !crate::codegen::mps::loads_views(step);

    let tiled = match &step.primitive {
        Primitive::Fusion { name, body, .. } => {
            !name.starts_with(crate::graph::NEURAL_ENGINE)
                && crate::runtime::kernel::mps::launch_of(body)
                    == crate::runtime::kernel::mps::Launch::Gemm
                && (gemm::row_dot(body).is_none() || gemm::tile_scratch(body).is_some())
        }
        _ => false,
    };

    float && views && tiled
}

impl Kernels for Gemms {
    type Step = MpsStep;
    type Config = GemmConfig;

    /// Candidate GEMM configs for `step`, filtered to those its epilogue fits in threadgroup memory.
    fn configs(&self, step: &MpsStep) -> Vec<GemmConfig> {
        let Some((dot, lhs, rhs, out)) = matmul_of(step) else {
            return Vec::new();
        };

        let Primitive::Fusion { body, .. } = &step.primitive else {
            return Vec::new();
        };

        let Some(d) = gemm_dot(body) else {
            return Vec::new();
        };

        let epilogue = gemm::gemm_flags(body, d);
        let group = gemm::gemm_group(body, d);
        let dual = gemm::gemm_mpp_pair(body, d);
        let staged = !epilogue.plain() && !dual;

        // Operand views tune the tiled and plain MPP kernels, which take the views' strides; the GEMV and
        // skinny kernels are picked for whole operands.
        let viewed = step.views.iter().any(Option::is_some);

        if !viewed && tiled_gemm(step) && gemm::gemv_config(body).is_some() {
            let (vector_lhs, _) = gemm::gemv_of(body, d).expect("a matrix-vector dot");
            let tiles = gemm::tile_scratch(body).is_some();

            return gemm::gemv_configs(vector_lhs)
                .iter()
                .filter(|c| !tiles || gemm::gemv_rows(c, vector_lhs).is_multiple_of(gemm::ROW_TILE))
                .filter(|c| group.is_none_or(|(g, rows)| (if rows { c.bm } else { c.bn }) % g == 0))
                .copied()
                .collect();
        }

        let skinny = gemm::skinny(dot, lhs, rhs, out);
        let large = out.numel() >= TUNED_OUTPUTS;

        if !tiled_gemm(step) || !(large || skinny) {
            return Vec::new();
        }

        if viewed {
            let tiled =
                !skinny && gemm::gemv_config(body).is_none() && gemm::tile_scratch(body).is_none();

            // A view's strides, else the operand's contiguous ones.
            let strides = |v: Var| {
                let k = body.inputs().iter().position(|&i| i == v);

                k.and_then(|k| step.views[k].as_ref()).map_or_else(
                    || contiguous_strides(&body.type_of(v).shape),
                    |view| view.strides.clone(),
                )
            };

            // MPP's layout (TL, TR) is generated from contiguous strides, so a view keeps it to read through MPP.
            let (ls, rs) = (strides(d.inputs[0]), strides(d.inputs[1]));
            let layout = gemm::mpp_transposes_at(dot, lhs, rhs, out, &ls, &rs);
            let mpp = large
                && epilogue.plain()
                && !dual
                && group.is_none()
                && layout.is_some()
                && layout == gemm::mpp_transposes(dot, lhs, rhs, out)
                && crate::runtime::kernel::mps::mpp_available();

            let mpp: &[GemmConfig] = match mpp {
                true => gemm::mpp_configs(),
                false => &[],
            };

            return match tiled {
                true => gemm::config()
                    .iter()
                    .chain(mpp)
                    .filter(|c| {
                        group.is_none_or(|(g, rows)| (if rows { c.bm } else { c.bn }) % g == 0)
                    })
                    .copied()
                    .collect(),
                false => Vec::new(),
            };
        }

        // Threadgroup bytes for the per-simdgroup epilogue staging area.
        let stage = |c: &GemmConfig| c.threads / 32 * 64 * 4;
        let halves = epilogue.group > 1 && !epilogue.interleaved;

        let mpp_fits = |c: &&GemmConfig| match () {
            _ if dual => c.bn >= 2 * 32,
            _ if staged => {
                let half = match epilogue.rows {
                    true => c.bm,
                    false => c.bn,
                };

                c.bm * c.bn * 4 + stage(c) <= gemm::MAX_THREADGROUP_BYTES
                    && (!halves || half >= 2 * 32)
            }
            _ => true,
        };

        let skinny_fits = |c: &&GemmConfig| {
            let partials = c.threads / (c.bn / c.bk) * (c.bn + 1) * 4;
            epilogue.plain() || partials + 8 * c.bn * 4 + stage(c) <= gemm::MAX_THREADGROUP_BYTES
        };

        let mpp = large
            && gemm::mpp_transposes(dot, lhs, rhs, out).is_some()
            && crate::runtime::kernel::mps::mpp_available();

        let mpp: Vec<GemmConfig> = match mpp {
            true => gemm::mpp_configs()
                .iter()
                .filter(mpp_fits)
                .copied()
                .collect(),
            false => Vec::new(),
        };

        // Row-tile epilogues only tune between the row-tile config and wide MPP tiles.
        if gemm::tile_scratch(body).is_some() {
            let wide = mpp.iter().filter(|c| c.bn % gemm::ROW_TILE == 0);

            return match mpp.is_empty() {
                true => Vec::new(),
                false => std::iter::once(&ROW_TILES).chain(wide).copied().collect(),
            };
        }

        let skinny: Vec<GemmConfig> = match skinny {
            true => gemm::skinny_configs()
                .iter()
                .filter(skinny_fits)
                .copied()
                .collect(),
            false => Vec::new(),
        };

        gemm::config()
            .iter()
            .chain(&mpp)
            .chain(&skinny)
            .filter(|c| group.is_none_or(|(g, rows)| (if rows { c.bm } else { c.bn }) % g == 0))
            .copied()
            .collect()
    }

    /// Whether `step` is a matmul fusion run by the generated GEMM kernels, of any size (tuned only when large or
    /// skinny).
    fn handles(&self, step: &MpsStep) -> bool {
        matmul_of(step).is_some() && tiled_gemm(step)
    }

    /// Whether `config` can run `step`: a GEMV config only a matrix-vector product.
    fn accepts(&self, step: &MpsStep, config: &GemmConfig) -> bool {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            return false;
        };

        self.handles(step)
            && (config.algorithm != gemm::Algorithm::Gemv || gemm::gemv_config(body).is_some())
    }

    /// Generates the GEMM variant for `config` and an encoder that launches it.
    fn build(&mut self, step: &MpsStep, config: GemmConfig) -> Tuned {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            unreachable!("a GEMM fusion")
        };

        let by_value: Vec<bool> = step
            .inputs
            .iter()
            .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
            .collect();

        let (kernel, source) = gemm::gemm_variant(body, &by_value, config, &self.config);
        self.sources.insert(kernel.clone(), source);
        let name = kernel.clone();
        let views = step.views.clone();

        // Tuned for the step's views: other views (an input bound strided at run time) take the default kernel.
        let encode: Encode = Box::new(move |step, inputs, output, scratch, keep| {
            let Primitive::Fusion { name: fusion, .. } = &step.primitive else {
                unreachable!("a GEMM fusion")
            };

            let tuned = (step.views == views).then_some((name.as_str(), config));

            crate::runtime::kernel::mps::encode_with(
                step, fusion, tuned, None, inputs, output, scratch, keep,
            )
        });

        Tuned::new(kernel, encode)
    }

    /// Compiles new GEMM variants, routing Metal Performance Primitives ones to the MPP compiler.
    fn compile(&mut self) -> Result<(), String> {
        let (mpp, sg): (BTreeMap<_, _>, BTreeMap<_, _>) = uncompiled(&mut self.sources)
            .into_iter()
            .partition(|(_, source)| source.contains("matmul_mpp"));

        compile_kernels(sg)?;
        compile_mpp_kernels(mpp)
    }
}

/// The Metal GEMMs' configs.
impl GemmTuning for Mps {
    type Tuner = Gemms;

    /// A GEMM tuner generating variants under `config`.
    fn gemm_tuner(config: &CompilerConfig) -> Gemms {
        Gemms::new(config)
    }
}
