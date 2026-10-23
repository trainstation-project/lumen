use std::collections::{BTreeMap, HashMap};
use std::ops::RangeInclusive;
use std::sync::PoisonError;

use super::{SOURCES, Tuning, codegen, compile_kernels, record_launch};
use crate::allocator::Scratch;
use crate::graph::{Primitive, TensorType, View};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::common::autotune::{Kernels, Timer};
use crate::graph_compiler::plan::{Buffer, Step, Tuned};
use crate::kernels::dot_general::mps::{self as gemm, GemmConfig};
use crate::kernels::mps::{Captured, Encode, capture};

pub(super) struct Tunable {
    pub by_value: Vec<bool>,
    pub views: HashMap<usize, (usize, Vec<usize>)>,
}

const TUNED_ELEMENTS: RangeInclusive<usize> = (64 << 10)..=(16 << 20);
const ALIGNMENT: usize = 256;

pub(super) struct Loops<'a> {
    pub tunable: &'a BTreeMap<String, Tunable>,
    pub config: &'a CompilerConfig,
    pub sources: BTreeMap<String, String>,
}

fn compile(sources: &mut BTreeMap<String, String>) -> Result<(), String> {
    let mut sources = std::mem::take(sources);
    {
        let compiled = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
        sources.retain(|name, _| !compiled.contains_key(name));
    }
    compile_kernels(sources)
}

impl Kernels for Loops<'_> {
    type Config = Tuning;

    fn configs(&self, step: &Step) -> Vec<Tuning> {
        let Primitive::Fusion { name, body, .. } = &step.primitive else {
            return Vec::new();
        };
        match self.tunable.contains_key(name) && TUNED_ELEMENTS.contains(&step.output.1.numel()) {
            true => codegen::tunings_of(body),
            false => Vec::new(),
        }
    }

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

    fn compile(&mut self) -> Result<(), String> {
        compile(&mut self.sources)
    }
}

#[derive(Default)]
pub(super) struct Gemms {
    sources: BTreeMap<String, String>,
}

const TUNED_OUTPUTS: usize = 64 << 10;

fn tiled_gemm(step: &Step) -> bool {
    let float = step.output.1.dtype.is_float() && step.output.1.numel() >= TUNED_OUTPUTS;
    let whole = step.views.iter().all(Option::is_none);
    let tiled = match &step.primitive {
        Primitive::DotGeneral { .. } => true,
        Primitive::Fusion { name, body, .. } => {
            !name.starts_with(crate::graph::NEURAL_ENGINE)
                && super::launch_of(body) == super::Launch::Gemm
                && codegen::row_dot(body).is_none()
                && codegen::tile_scratch(body).is_none()
        }
        _ => false,
    };
    float && whole && tiled
}

impl Kernels for Gemms {
    type Config = GemmConfig;

    fn configs(&self, step: &Step) -> Vec<GemmConfig> {
        match tiled_gemm(step) {
            true => gemm::configs().to_vec(),
            false => Vec::new(),
        }
    }

    fn build(&mut self, step: &Step, config: GemmConfig) -> Tuned {
        let (kernel, source) = match &step.primitive {
            Primitive::Fusion { body, .. } => {
                let by_value: Vec<bool> = step
                    .inputs
                    .iter()
                    .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
                    .collect();
                codegen::gemm_variant(body, &by_value, config)
            }
            p => {
                let Primitive::DotGeneral { accum_dtype, .. } = p else {
                    unreachable!("a matmul")
                };
                let (t, a, o) = (
                    codegen::metal_type(step.inputs[0].1.dtype),
                    codegen::metal_type(*accum_dtype),
                    codegen::metal_type(step.output.1.dtype),
                );
                gemm::matmul_source(t, a, o, config)
            }
        };
        self.sources.insert(kernel.clone(), source);
        let name = kernel.clone();
        let encode: Encode =
            Box::new(
                move |step, inputs, output, scratch, keep| match &step.primitive {
                    Primitive::Fusion { name: fusion, .. } => super::encode_with(
                        step,
                        fusion,
                        Some((&name, config)),
                        inputs,
                        output,
                        scratch,
                        keep,
                    ),
                    _ => gemm::encode_tuned(step, &name, config, inputs, output, keep),
                },
            );
        Tuned::new(kernel, encode)
    }

    fn compile(&mut self) -> Result<(), String> {
        compile(&mut self.sources)
    }
}

pub(super) struct Mps;

impl Timer for Mps {
    type Buffers = Buffers;

    fn buffers(&self, step: &Step) -> Result<Option<Buffers>, String> {
        Ok(Buffers::new(step))
    }

    fn time(&self, buffers: &Buffers, step: &Step, reps: usize) -> Result<f64, String> {
        buffers.run(step, reps)
    }
}

const COLD_BYTES: usize = 64 << 20;
const MOST_COPIES: usize = 256;

pub(super) struct Buffers {
    _memory: Scratch,
    copies: Vec<Vec<*const u8>>,
    output: *mut u8,
    scratch: *mut u8,
}

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

fn aligned(bytes: usize) -> usize {
    bytes.div_ceil(ALIGNMENT).max(1) * ALIGNMENT
}

impl Buffers {
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
        let mut copies = COLD_BYTES.div_ceil(read).clamp(1, MOST_COPIES);
        let memory = loop {
            match Scratch::zeroed(read * copies + written + scratch) {
                Some(memory) => break memory,
                None if copies > 1 => copies /= 2,
                None => return None,
            }
        };
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

    fn run(&self, step: &Step, reps: usize) -> Result<f64, String> {
        let launches = self
            .copies
            .iter()
            .map(|inputs| {
                capture(|| {
                    crate::kernels::mps::encode(step, inputs, self.output, self.scratch, Vec::new())
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        (0..launches[0].len())
            .map(|l| {
                let copies: Vec<&Captured> = launches.iter().map(|c| &c[l]).collect();
                Captured::time(&copies, reps.max(copies.len()))
            })
            .sum()
    }
}
