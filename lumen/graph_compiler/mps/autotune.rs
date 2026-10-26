//! MPS autotuning backends: candidate tunings for loop fusions and GEMM configs, and the timer that runs them.

use std::collections::{BTreeMap, HashMap};
use std::sync::PoisonError;

use super::{SOURCES, Tuning, codegen, compile_kernels, compile_mpp_kernels, record_launch};
use crate::allocator::Scratch;
use crate::device::Device;
use crate::graph::{Primitive, TensorType, View};
use crate::graph_compiler::MpsConfig;
use crate::graph_compiler::common::autotune::{Kernels, Timer};
use crate::graph_compiler::plan::{Buffer, Step, Tuned};
use crate::kernels::dot_general::mps::{self as gemm, GemmConfig};
use crate::kernels::mps::Encode;

/// What a tunable loop fusion needs to regenerate its kernel under another tuning.
pub(super) struct Tunable {
    /// Per fusion input, whether it is passed by value.
    pub by_value: Vec<bool>,
    /// Strided views of fusion inputs, keyed by input index, as `(offset, strides)`.
    pub views: HashMap<usize, (usize, Vec<usize>)>,
}

/// Byte alignment of each buffer carved out of the timing scratch.
const ALIGNMENT: usize = 256;

/// Tuning backend for loop fusions.
pub(super) struct Loops<'a> {
    /// Fusions that may be retuned, by name.
    pub tunable: &'a BTreeMap<String, Tunable>,
    pub config: &'a MpsConfig,
    /// Kernel sources built so far, compiled together by `compile`.
    pub sources: BTreeMap<String, String>,
}

/// Takes `sources`, keeping only kernels not already compiled.
fn uncompiled(sources: &mut BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut sources = std::mem::take(sources);
    let compiled = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    sources.retain(|name, _| !compiled.contains_key(name));
    sources
}

impl Kernels for Loops<'_> {
    type Config = Tuning;

    /// Tunings to try for a tunable fusion of a mid-sized output, or none.
    fn configs(&self, step: &Step) -> Vec<Tuning> {
        let Primitive::Fusion { name, body, .. } = &step.primitive else {
            return Vec::new();
        };
        match self.tunable.contains_key(name)
            && codegen::TUNED_ELEMENTS.contains(&step.output.1.numel())
        {
            true => codegen::tunings_of(body),
            false => Vec::new(),
        }
    }

    /// Generates the fusion's kernel under `tuning` and an encoder that launches it.
    fn build(&mut self, step: &Step, tuning: Tuning) -> Tuned {
        let Primitive::Fusion { name, body, .. } = &step.primitive else {
            unreachable!("a fusion")
        };
        let t = &self.tunable[name];
        let (kernel, source) =
            codegen::kernel(body, &t.by_value, self.config, &t.views, Some(tuning));
        record_launch(&kernel, body, Some(tuning));
        self.sources.insert(kernel.clone(), source);
        let name = kernel.clone();
        let encode: Encode = Box::new(move |step, inputs, output, scratch, keep| {
            super::encode_with(step, &name, None, inputs, output, scratch, keep)
        });
        Tuned::new(kernel, encode)
    }

    /// Compiles the newly generated loop kernels.
    fn compile(&mut self) -> Result<(), String> {
        compile_kernels(uncompiled(&mut self.sources))
    }
}

/// Tuning backend for GEMM fusions.
#[derive(Default)]
pub(super) struct Gemms {
    sources: BTreeMap<String, String>,
    config: MpsConfig,
}

/// Creates an empty GEMM tuner for `config`.
impl Gemms {
    pub(super) fn new(config: &MpsConfig) -> Self {
        Gemms {
            sources: BTreeMap::new(),
            config: config.clone(),
        }
    }
}

/// Config used for GEMMs whose epilogue writes row tiles to scratch.
const ROW_TILES: GemmConfig = GemmConfig {
    bm: codegen::ROW_TILE,
    bn: codegen::ROW_TILE,
    bk: 32,
    threads: 256,
    cols: 1,
    prefetch: false,
    algorithm: gemm::Algorithm::Tiled,
};

/// Output size, in elements, from which a GEMM counts as large enough to tune.
const TUNED_OUTPUTS: usize = 64 << 10;

/// The dot of a GEMM fusion with its lhs, rhs and output types.
pub(super) fn matmul_of(
    step: &Step,
) -> Option<(&Primitive, &TensorType, &TensorType, &TensorType)> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        return None;
    };
    let dot = codegen::gemm_dot(body)?;
    let ty = |v| body.type_of(v);
    Some((
        &dot.primitive,
        ty(dot.inputs[0]),
        ty(dot.inputs[1]),
        ty(dot.output),
    ))
}

/// Whether `step` is a float GEMM fusion over whole buffers that the tiled GEMM kernels can run.
fn tiled_gemm(step: &Step) -> bool {
    let float = step.output.1.dtype.is_float();
    let whole = step.views.iter().all(Option::is_none);
    let tiled = match &step.primitive {
        Primitive::Fusion { name, body, .. } => {
            !name.starts_with(crate::graph::NEURAL_ENGINE)
                && super::launch_of(body) == super::Launch::Gemm
                && (codegen::row_dot(body).is_none() || codegen::tile_scratch(body).is_some())
        }
        _ => false,
    };
    float && whole && tiled
}

impl Kernels for Gemms {
    type Config = GemmConfig;

    /// Candidate GEMM configs for `step`, filtered to those its epilogue fits in threadgroup memory.
    fn configs(&self, step: &Step) -> Vec<GemmConfig> {
        let Some((dot, lhs, rhs, out)) = matmul_of(step) else {
            return Vec::new();
        };
        let Primitive::Fusion { body, .. } = &step.primitive else {
            return Vec::new();
        };
        let Some(d) = codegen::gemm_dot(body) else {
            return Vec::new();
        };
        let epilogue = codegen::gemm_flags(body, d);
        let group = codegen::gemm_group(body, d);
        let dual = codegen::gemm_mpp_pair(body, d);
        let staged = !epilogue.plain() && !dual;
        if tiled_gemm(step) && codegen::gemv_config(body).is_some() {
            let (vector_lhs, _) = codegen::gemv_of(body, d).expect("a matrix-vector dot");
            let tiles = codegen::tile_scratch(body).is_some();
            return gemm::gemv_configs(vector_lhs)
                .iter()
                .filter(|c| {
                    !tiles || gemm::gemv_rows(c, vector_lhs).is_multiple_of(codegen::ROW_TILE)
                })
                .filter(|c| group.is_none_or(|(g, rows)| (if rows { c.bm } else { c.bn }) % g == 0))
                .copied()
                .collect();
        }
        let skinny = gemm::skinny(dot, lhs, rhs, out);
        let large = out.numel() >= TUNED_OUTPUTS;
        if !tiled_gemm(step) || !(large || skinny) {
            return Vec::new();
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
        let mpp =
            large && gemm::mpp_transposes(dot, lhs, rhs, out).is_some() && super::mpp_available();
        let mpp: Vec<GemmConfig> = match mpp {
            true => gemm::mpp_configs()
                .iter()
                .filter(mpp_fits)
                .copied()
                .collect(),
            false => Vec::new(),
        };
        // Row-tile epilogues only tune between the row-tile config and wide MPP tiles.
        if codegen::tile_scratch(body).is_some() {
            let wide = mpp.iter().filter(|c| c.bn % codegen::ROW_TILE == 0);
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
        gemm::configs()
            .iter()
            .chain(&mpp)
            .chain(&skinny)
            .filter(|c| group.is_none_or(|(g, rows)| (if rows { c.bm } else { c.bn }) % g == 0))
            .copied()
            .collect()
    }

    /// Generates the GEMM variant for `config` and an encoder that launches it.
    fn build(&mut self, step: &Step, config: GemmConfig) -> Tuned {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            unreachable!("a GEMM fusion")
        };
        let by_value: Vec<bool> = step
            .inputs
            .iter()
            .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
            .collect();
        let (kernel, source) = codegen::gemm_variant(body, &by_value, config, &self.config);
        self.sources.insert(kernel.clone(), source);
        let name = kernel.clone();
        let encode: Encode = Box::new(move |step, inputs, output, scratch, keep| {
            let Primitive::Fusion { name: fusion, .. } = &step.primitive else {
                unreachable!("a GEMM fusion")
            };
            super::encode_with(
                step,
                fusion,
                Some((&name, config)),
                inputs,
                output,
                scratch,
                keep,
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

/// Timer that runs tuning candidates on the first MPS device.
pub(super) struct Mps;

impl Timer for Mps {
    type Buffers = Buffers;

    fn device(&self) -> Device {
        Device::Mps(0)
    }

    /// Allocates the timing buffers for `step`.
    fn buffers(&self, step: &Step) -> Result<Option<Buffers>, String> {
        Ok(Buffers::new(step))
    }

    /// Encodes one launch, rotating through input copies so repeats read cold memory.
    fn run(&self, buffers: &Buffers, step: &Step, rep: usize) -> Result<(), String> {
        let inputs = &buffers.copies[rep % buffers.copies.len()];
        crate::kernels::mps::encode(step, inputs, buffers.output, buffers.scratch, Vec::new())
    }
}

/// Input bytes to spread across copies so repeated runs miss the cache.
const COLD_BYTES: usize = 64 << 20;
/// Most input copies allocated for timing.
const MOST_COPIES: usize = 256;

/// Zeroed device memory holding rotated input copies, the outputs and scratch for a timed step.
pub(super) struct Buffers {
    _memory: Scratch,
    /// Per copy, the input pointers followed by the extra output pointers.
    copies: Vec<Vec<*const u8>>,
    output: *mut u8,
    scratch: *mut u8,
}

/// Bytes spanned by a tensor of type `ty` under `view`, including the view's offset when `offset` is set.
fn extent(ty: &TensorType, view: Option<&View>, offset: bool) -> usize {
    let size = ty.dtype.size_of();
    match view {
        _ if ty.numel() == 0 => 0,
        None => ty.numel() * size,
        Some(view) => {
            let last: usize = ty
                .shape
                .iter()
                .zip(&view.strides)
                .map(|(&d, &s)| (d - 1) * s)
                .sum();
            (last + 1 + if offset { view.offset } else { 0 }) * size
        }
    }
}

/// Rounds `bytes` up to a nonzero multiple of `ALIGNMENT`.
fn aligned(bytes: usize) -> usize {
    bytes.div_ceil(ALIGNMENT).max(1) * ALIGNMENT
}

impl Buffers {
    /// Lays out the buffers for `step`, halving the copy count until the allocation succeeds.
    fn new(step: &Step) -> Option<Self> {
        let inputs: Vec<usize> = step
            .inputs
            .iter()
            .zip(&step.views)
            .map(|((b, ty), view)| match b {
                Buffer::Scalar(_) => ty.dtype.size_of(),
                _ => extent(ty, view.as_ref(), false),
            })
            .map(aligned)
            .collect();
        let outputs: Vec<usize> = std::iter::once(&step.output)
            .chain(&step.extra_outputs)
            .enumerate()
            .map(|(k, (_, ty))| extent(ty, step.output_views.get(k).and_then(Option::as_ref), true))
            .map(aligned)
            .collect();
        let scratch = aligned(step.scratch.map_or(0, |(_, bytes)| bytes));
        let read: usize = inputs.iter().sum();
        let written: usize = outputs.iter().sum();
        let mut copies = COLD_BYTES.div_ceil(read).clamp(2, MOST_COPIES);
        let memory = loop {
            match Scratch::zeroed(read * copies + written + scratch) {
                Some(memory) => break memory,
                None if copies > 1 => copies /= 2,
                None => return None,
            }
        };
        // Layout: `copies` input sets back to back, then the outputs, then scratch.
        let base = memory.as_ptr();
        let mut at = read * copies;
        let mut outs = Vec::with_capacity(outputs.len());
        for size in &outputs {
            outs.push(base.wrapping_add(at));
            at += size;
        }
        let copies = (0..copies)
            .map(|c| {
                let mut at = c * read;
                let ins = inputs.iter().map(|size| {
                    let p = base.wrapping_add(at).cast_const();
                    at += size;
                    p
                });
                ins.chain(outs[1..].iter().map(|p| p.cast_const()))
                    .collect()
            })
            .collect();
        Some(Buffers {
            output: outs[0],
            scratch: base.wrapping_add(at),
            _memory: memory,
            copies,
        })
    }
}
