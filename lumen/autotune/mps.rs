//! MPS autotuning backends: candidate tunings for elementwise fusions and GEMM configs, and the timer that runs them.

use std::collections::{BTreeMap, HashMap};
use std::sync::PoisonError;

use super::{Kernels, Timer};
use crate::allocator::Scratch;
use crate::codegen::dot_general::mps::{self as gemm, GemmConfig};
use crate::codegen::elementwise::mps::TUNED_ELEMENTS;
use crate::codegen::elementwise::mps::Tuning;
use crate::codegen::kinds::gemm_dot;
use crate::device::Device;
use crate::graph::{Primitive, TensorType, Var, View};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::mps::{
    SOURCES, compile_kernels, compile_mpp_kernels, compile_routed, record_launch,
};
use crate::plan::{Buffer, Step, Tuned};
use crate::runtime::mps::Encode;
use crate::tensor::contiguous_strides;

/// What a tunable elementwise fusion needs to regenerate its kernel under another tuning.
pub(crate) struct Tunable {
    /// Per fusion input, whether it is passed by value.
    pub by_value: Vec<bool>,
    /// Strided views of fusion inputs, keyed by input index, as `(offset, strides)`.
    pub views: HashMap<usize, (usize, Vec<usize>)>,
}

/// Byte alignment of each buffer carved out of the timing scratch.
const ALIGNMENT: usize = 256;

/// Tuning backend for elementwise fusions.
pub(crate) struct Elementwise<'a> {
    /// Fusions that may be retuned, by name.
    pub tunable: &'a BTreeMap<String, Tunable>,
    pub config: &'a CompilerConfig,
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

impl Kernels for Elementwise<'_> {
    type Config = Tuning;

    /// Tunings to try for a tunable fusion of a mid-sized output, or none.
    fn configs(&self, step: &Step) -> Vec<Tuning> {
        let Primitive::Fusion { name, body, .. } = &step.primitive else {
            return Vec::new();
        };

        match self.tunable.contains_key(name) && TUNED_ELEMENTS.contains(&step.output.1.numel()) {
            true => crate::codegen::mps::tunings_of(body),
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
            crate::codegen::mps::kernel(body, &t.by_value, self.config, &t.views, Some(tuning));

        record_launch(&kernel, body, Some(tuning));
        self.sources.insert(kernel.clone(), source);
        let name = kernel.clone();

        let encode: Encode = Box::new(move |step, inputs, output, scratch, keep| {
            crate::graph_compiler::mps::encode_with(
                step, &name, None, None, inputs, output, scratch, keep,
            )
        });

        Tuned::new(kernel, encode)
    }

    /// Compiles the newly generated elementwise kernels.
    fn compile(&mut self) -> Result<(), String> {
        compile_kernels(uncompiled(&mut self.sources))
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
    step: &Step,
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
fn tiled_gemm(step: &Step) -> bool {
    let float = step.output.1.dtype.is_float();
    let views = !crate::codegen::mps::loads_views(step);

    let tiled = match &step.primitive {
        Primitive::Fusion { name, body, .. } => {
            !name.starts_with(crate::graph::NEURAL_ENGINE)
                && crate::graph_compiler::mps::launch_of(body)
                    == crate::graph_compiler::mps::Launch::Gemm
                && (gemm::row_dot(body).is_none() || gemm::tile_scratch(body).is_some())
        }
        _ => false,
    };

    float && views && tiled
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
                && crate::graph_compiler::mps::mpp_available();

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
            && crate::graph_compiler::mps::mpp_available();

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

            crate::graph_compiler::mps::encode_with(
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

/// Tuning backend for flash attention forwards: the kernel (8x8 matrices or Metal Performance Primitives), its
/// simdgroups and its key block.
#[derive(Default)]
pub(crate) struct Attentions {
    /// Kernel sources built so far, compiled together by `compile`.
    sources: BTreeMap<String, String>,
}

impl Kernels for Attentions {
    type Config = crate::codegen::attention::mps::Tuning;

    /// The tunings of an attention forward step's matched attention, laid out as the step reads it.
    fn configs(&self, step: &Step) -> Vec<Self::Config> {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            return Vec::new();
        };

        let strides = crate::graph_compiler::mps::attention_strides(step, body);

        match crate::graph_compiler::common::attention::of_body_strided(body, &strides) {
            Some(a) => crate::codegen::attention::mps::tunings(&a),
            None => Vec::new(),
        }
    }

    /// Generates the forward for `tuning` and an encoder launching it.
    fn build(&mut self, step: &Step, tuning: Self::Config) -> Tuned {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            unreachable!("an attention fusion")
        };

        let strides = crate::graph_compiler::mps::attention_strides(step, body);
        let a = crate::graph_compiler::common::attention::of_body_strided(body, &strides)
            .expect("an attention it was tuned for");

        let (kernel, source) = crate::codegen::attention::mps::forward_with(body, &a, tuning)
            .expect("a tuning generates as the untuned forward does");

        self.sources.insert(kernel.clone(), source);
        let name = kernel.clone();
        let views = step.views.clone();

        // Tuned for the step's views: other views (an input bound strided at run time) take the default kernel.
        let encode: Encode = Box::new(move |step, inputs, output, scratch, keep| {
            let Primitive::Fusion { name: fusion, .. } = &step.primitive else {
                unreachable!("an attention fusion")
            };

            let tuned = (step.views == views).then_some((name.as_str(), tuning));

            crate::graph_compiler::mps::encode_with(
                step, fusion, None, tuned, inputs, output, scratch, keep,
            )
        });

        Tuned::new(kernel, encode)
    }

    /// Compiles the newly generated forwards.
    fn compile(&mut self) -> Result<(), String> {
        compile_routed(uncompiled(&mut self.sources))
    }
}

/// Timer that runs tuning candidates on the first MPS device.
pub(crate) struct Mps;

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
        crate::runtime::mps::encode(step, inputs, buffers.output, buffers.scratch, Vec::new())
    }
}

/// Input bytes to spread across copies so repeated runs miss the cache.
const COLD_BYTES: usize = 64 << 20;
/// Most input copies allocated for timing.
const MOST_COPIES: usize = 256;

/// Zeroed device memory holding rotated input copies, the outputs and scratch for a timed step.
pub(crate) struct Buffers {
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
