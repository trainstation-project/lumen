use std::collections::{BTreeMap, HashMap};
use std::ops::RangeInclusive;
use std::sync::PoisonError;

use super::{SOURCES, Tuning, codegen, compile_kernels, encode, record_launch};
use crate::allocator::Scratch;
use crate::graph::{Primitive, TensorType, View};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::common::autotune::Autotuner;
use crate::graph_compiler::plan::{Buffer, Step};
use crate::kernels::mps::{Captured, capture};

pub(super) struct Tunable {
    pub by_value: Vec<bool>,
    pub views: HashMap<usize, (usize, Vec<usize>)>,
}

const TUNED_ELEMENTS: RangeInclusive<usize> = (64 << 10)..=(16 << 20);
const TIMED_ELEMENTS: usize = 16 << 20;
const ALIGNMENT: usize = 256;

pub(super) struct MpsAutotuner<'a> {
    pub tunable: &'a BTreeMap<String, Tunable>,
    pub config: &'a CompilerConfig,
    pub sources: BTreeMap<String, String>,
}

fn fusion(step: &Step) -> (&String, &crate::graph::Graph) {
    let Primitive::Fusion { name, body, .. } = &step.primitive else {
        unreachable!("a fusion")
    };
    (name, body)
}

impl Autotuner for MpsAutotuner<'_> {
    type Config = Tuning;

    fn device(&self) -> &'static str {
        "mps"
    }

    fn key(&self, step: &Step) -> Option<String> {
        let Primitive::Fusion { name, .. } = &step.primitive else {
            return None;
        };
        let n = step.output.1.numel();
        (self.tunable.contains_key(name) && TUNED_ELEMENTS.contains(&n))
            .then(|| format!("{name}/{n}"))
    }

    fn candidates(&self, step: &Step) -> Vec<Tuning> {
        codegen::tunings_of(fusion(step).1)
    }

    fn generate(&mut self, step: &Step, tuning: Tuning) -> String {
        let (name, body) = fusion(step);
        let t = &self.tunable[name];
        let (kernel, source) =
            codegen::kernel(body, &t.by_value, self.config, &t.views, Some(tuning));
        record_launch(&kernel, body, Some(tuning));
        self.sources.insert(kernel.clone(), source);
        kernel
    }

    fn compile(&mut self) -> Result<(), String> {
        let mut sources = std::mem::take(&mut self.sources);
        {
            let compiled = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
            sources.retain(|name, _| !compiled.contains_key(name));
        }
        compile_kernels(sources)
    }

    fn time(&self, step: &Step, kernels: &[String]) -> Result<Vec<f64>, String> {
        let reps = (TIMED_ELEMENTS / step.output.1.numel()).clamp(4, 256);
        let Some(buffers) = Buffers::new(step) else {
            return Ok(Vec::new());
        };
        kernels
            .iter()
            .map(|kernel| {
                let mut timed = step.clone();
                let Primitive::Fusion { name, .. } = &mut timed.primitive else {
                    unreachable!("a fusion")
                };
                name.clone_from(kernel);
                buffers.run(&timed, reps)
            })
            .collect()
    }

    fn apply(&self, step: &mut Step, kernel: &str, _: Tuning) {
        let Primitive::Fusion { name, .. } = &mut step.primitive else {
            unreachable!("a fusion")
        };
        *name = kernel.to_owned();
    }
}

const COLD_BYTES: usize = 64 << 20;
const MOST_COPIES: usize = 256;

struct Buffers {
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
            .map(|inputs| capture(|| encode(step, inputs, self.output, self.scratch, Vec::new())))
            .collect::<Result<Vec<_>, String>>()?;
        (0..launches[0].len())
            .map(|l| {
                let copies: Vec<&Captured> = launches.iter().map(|c| &c[l]).collect();
                Captured::time(&copies, reps.max(copies.len()))
            })
            .sum()
    }
}
