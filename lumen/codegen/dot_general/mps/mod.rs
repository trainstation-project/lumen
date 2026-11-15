//! Metal `dot_general` kernels: picking the tile, collapsing operands into batched matmul form and launching.
//!
//! Matmul kernel configuration shared by every Metal GEMM algorithm.
//!
//! Autotuning the Metal GEMMs: tiled, Metal Performance Primitives, skinny and GEMV configs for GEMM fusions.

mod codegen;
mod gemv;
mod linear_cross_entropy;
mod mpp;
mod row;
mod skinny;
mod tiled;

pub(crate) use codegen::*;
pub(crate) use gemv::*;
pub(crate) use linear_cross_entropy::*;
pub(crate) use mpp::*;
pub(crate) use row::*;
pub(crate) use skinny::*;
pub(crate) use tiled::*;

use crate::codegen::dot_general::mps as gemm;
use crate::codegen::kernel::Kernel;
use crate::codegen::kinds::gemm_dot;
use crate::graph::Primitive;
use crate::graph::TensorType;
use crate::graph::Var;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::passes::dot::canonicalize::matmul_order;
use crate::plan::step::mps::Encoding;
use crate::plan::{Buffer, MpsStep, StepBase};
use crate::runtime::kernel::mps::Grid;
use crate::runtime::kernel::mps::{compile_kernels, compile_mpp_kernels, variants};
use crate::tensor::contiguous_strides;
use std::collections::BTreeMap;

/// Default float tile (rows, columns).
const FLOAT_TILE: (usize, usize) = (128, 64);
/// Float tile when M leaves a 33..=64-row remainder on the default tile.
const MID_TILE: (usize, usize) = (64, 64);
/// Float tile for small problems or a short last row tile.
const SMALL_TILE: (usize, usize) = (32, 32);
/// Tile for 32-bit and narrower integer outputs.
const INT_TILE: (usize, usize) = (64, 64);
/// Tile for 64-bit integer outputs.
const WIDE_TILE: (usize, usize) = (32, 32);
/// Below this many default tiles the small tile is used, for more parallelism.
const SMALL_TILES: usize = 32;

/// Innermost stride of each of the three dim groups of `order` once collapsed, or `None` if a
/// group's dims are not contiguous with each other.
pub(crate) fn collapsed(
    ty: &TensorType,
    strides: &[usize],
    order: &[usize],
    (i, j): (usize, usize),
) -> Option<[usize; 3]> {
    let mut out = [0; 3];

    for (k, group) in [&order[..i], &order[i..j], &order[j..]]
        .into_iter()
        .enumerate()
    {
        out[k] = collapse(group.iter().map(|&d| (ty.shape[d], strides[d])))?.1;
    }

    Some(out)
}

/// Whether operand `k` with `strides` can be read by the matmul kernels without a copy.
pub(crate) fn reads_strided(
    p: &Primitive,
    operands: [&TensorType; 2],
    k: usize,
    strides: &[usize],
) -> bool {
    let order = matmul_order(p, operands[0].shape.len(), operands[1].shape.len());

    let (dims, split) = match k {
        0 => (&order.lhs, order.lhs_split),
        _ => (&order.rhs, order.rhs_split),
    };

    collapsed(operands[k], strides, dims, split).is_some()
}

/// Whether to use the small or mid float tile for a `b x m x n` output, and the tile chosen.
fn float_tiles(b: usize, m: usize, n: usize) -> (bool, bool, (usize, usize)) {
    // Small tiles when few default tiles exist or the last row tile would be mostly empty.
    let large = b * m.div_ceil(FLOAT_TILE.0) * n.div_ceil(FLOAT_TILE.1);
    let small = large < SMALL_TILES || matches!(m % FLOAT_TILE.0, 1..=32);
    let mid = !small && matches!(m % FLOAT_TILE.0, 33..=64);

    let tile = match (small, mid) {
        (true, _) => SMALL_TILE,
        (_, true) => MID_TILE,
        _ => FLOAT_TILE,
    };

    (small, mid, tile)
}

/// Number of threadgroups the float matmul launches for a `b x m x n` output.
pub(crate) fn float_groups(b: usize, m: usize, n: usize) -> usize {
    let (_, _, (tm, tn)) = float_tiles(b, m, n);
    b * m.div_ceil(tm) * n.div_ceil(tn)
}

/// Launch shape and parameters of one matmul.
pub(crate) struct MatmulLaunch {
    /// Whether the small float tile is used.
    pub small: bool,
    /// Whether the mid float tile is used.
    pub mid: bool,
    pub grid: Grid,
    /// `[M, N, K, lhs batch/M/K strides, rhs batch/K/N strides]`, in elements.
    pub p: [usize; 9],
}

/// Collapses `dot_general` operands into one batched matmul and sizes its grid; `tile` overrides
/// the float tile choice. Errors when an operand's dims cannot be collapsed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_matmul(
    prim: &Primitive,
    lhs: &TensorType,
    rhs: &TensorType,
    out: &TensorType,
    lhs_strides: &[usize],
    rhs_strides: &[usize],
    name: &str,
    tile: Option<(usize, usize)>,
) -> Result<MatmulLaunch, String> {
    let order = matmul_order(prim, lhs.shape.len(), rhs.shape.len());

    let not_matmul = || {
        format!(
            "{name}: an operand is not in matmul form: compile the graph for MPS (lumen.compile, Plan(graph, \"mps\"))"
        )
    };

    let [lsb, lsm, lsk] =
        collapsed(lhs, lhs_strides, &order.lhs, order.lhs_split).ok_or_else(not_matmul)?;

    let [rsb, rsk, rsn] =
        collapsed(rhs, rhs_strides, &order.rhs, order.rhs_split).ok_or_else(not_matmul)?;

    let size =
        |ty: &TensorType, dims: &[usize]| dims.iter().map(|&d| ty.shape[d]).product::<usize>();

    let (b, m) = (
        size(lhs, &order.lhs[..order.lhs_split.0]),
        size(lhs, &order.lhs[order.lhs_split.0..order.lhs_split.1]),
    );

    let (k, n) = (
        size(lhs, &order.lhs[order.lhs_split.1..]),
        size(rhs, &order.rhs[order.rhs_split.1..]),
    );

    let float = out.dtype.is_float();

    let (small, mid, tile) = match tile {
        None => float_tiles(b, m, n),
        Some(tile) => (false, false, tile),
    };

    let (small, mid) = (float && small, float && mid);

    let (tm, tn) = match float {
        true => tile,
        false if out.dtype.size_of() == 8 => WIDE_TILE,
        false => INT_TILE,
    };

    Ok(MatmulLaunch {
        small,
        mid,
        grid: Grid::Groups([n.div_ceil(tn), m.div_ceil(tm), b]),
        p: [m, n, k, lsb, lsm, lsk, rsb, rsk, rsn],
    })
}

/// Merges dims `(size, stride)` (size-1 dims dropped) into one, if each stride equals the next's
/// size times stride; returns the total size and innermost stride.
fn collapse(dims: impl IntoIterator<Item = (usize, usize)>) -> Option<(usize, usize)> {
    let dims: Vec<(usize, usize)> = dims.into_iter().filter(|d| d.0 != 1).collect();

    if dims.windows(2).any(|w| w[0].1 != w[1].0 * w[1].1) {
        return None;
    }

    Some((
        dims.iter().map(|d| d.0).product(),
        dims.last().map_or(0, |d| d.1),
    ))
}

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
        let all = self::config()
            .iter()
            .chain(self::mpp_configs())
            .chain(self::skinny_configs())
            .chain(self::gemv_configs(true))
            .chain(self::gemv_configs(false));

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

/// The Metal GEMMs (tiled, Metal Performance Primitives, skinny, GEMV) running a GEMM fusion, a variant generated
/// for each config.
#[derive(Default)]
pub(crate) struct Gemm {
    /// The compiler config its variants are generated under.
    pub(crate) config: CompilerConfig,
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
    step: &StepBase,
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
fn tiled_gemm(step: &StepBase) -> bool {
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

impl Kernel for Gemm {
    type Step = MpsStep;
    type Buffers = Encoding;
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

    /// Whether `config` can run `step`: a GEMV config only a matrix-vector product.
    fn accepts(&self, step: &MpsStep, config: &GemmConfig) -> bool {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            return false;
        };

        config.algorithm != gemm::Algorithm::Gemv || gemm::gemv_config(body).is_some()
    }

    /// Generates the variant for each of `configs`, compiling them together.
    fn prepare(&self, step: &MpsStep, configs: &[GemmConfig]) -> Result<(), String> {
        self.variants(step, configs).map(|_| ())
    }

    /// Encodes the variant for `config` (its untuned kernel when `None`).
    fn run(&self, step: &MpsStep, config: Option<&GemmConfig>, e: &Encoding) -> Result<(), String> {
        let Some(&config) = config else {
            return step.encode_untuned(&e.inputs, e.output, e.scratch, e.keep.clone());
        };

        let Primitive::Fusion { name: fusion, .. } = &step.primitive else {
            unreachable!("a GEMM fusion")
        };

        let name = self.variants(step, &[config])?.remove(0);

        crate::runtime::kernel::mps::encode_with(
            step,
            fusion,
            Some((name.as_str(), config)),
            None,
            &e.inputs,
            e.output,
            e.scratch,
            e.keep.clone(),
        )
    }
}

impl Gemm {
    /// Names of the fusion's GEMM variants for `configs`, compiled on first use: Metal Performance Primitives ones
    /// by the MPP compiler.
    fn variants(&self, step: &MpsStep, configs: &[GemmConfig]) -> Result<Vec<String>, String> {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            unreachable!("a GEMM fusion")
        };

        let by_value: Vec<bool> = step
            .inputs
            .iter()
            .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
            .collect();

        let generate =
            |&config: &GemmConfig| gemm::gemm_variant(body, &by_value, config, &self.config);

        let compile = |sources: BTreeMap<String, String>| {
            let (mpp, sg): (BTreeMap<_, _>, BTreeMap<_, _>) = sources
                .into_iter()
                .partition(|(_, source)| source.contains("matmul_mpp"));

            compile_kernels(sg)?;
            compile_mpp_kernels(mpp)
        };

        variants(step, configs, generate, compile)
    }
}

/// Whether the Metal GEMMs run `step`: a matmul fusion the tiled kernels take, of any size (tuned only when large
/// or skinny).
pub(crate) fn runs(step: &StepBase) -> bool {
    matmul_of(step).is_some() && tiled_gemm(step)
}
