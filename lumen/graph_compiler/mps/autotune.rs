use std::collections::{BTreeMap, HashMap};
use std::ops::RangeInclusive;
use std::sync::PoisonError;

use super::{SOURCES, Tuning, codegen, compile_kernels, encode, record_launch};
use crate::graph::{Primitive, TensorType, View};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::common::autotune::Autotuner;
use crate::graph_compiler::plan::{Buffer, Step};
use crate::kernels::mps::capture;
use crate::{DType, Device, Tensor, TensorOptions};

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
        let buffers = Buffers::new(step);
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

struct Buffers {
    memory: Tensor,
    inputs: Vec<*const u8>,
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

impl Buffers {
    fn new(step: &Step) -> Self {
        let inputs = step
            .inputs
            .iter()
            .zip(&step.views)
            .map(|((b, ty), view)| match b {
                Buffer::Scalar(_) => ty.dtype.size_of(),
                _ => extent(ty, view.as_ref(), false),
            });
        let outputs = std::iter::once(&step.output)
            .chain(&step.extra_outputs)
            .enumerate()
            .map(|(k, (_, ty))| {
                extent(ty, step.output_views.get(k).and_then(Option::as_ref), true)
            });
        let scratch = step.scratch.map_or(0, |(_, bytes)| bytes);
        let sizes: Vec<usize> = inputs.chain(outputs).chain([scratch]).collect();
        let mut at = Vec::with_capacity(sizes.len());
        let mut total = 0;
        for size in &sizes {
            at.push(total);
            total += size.div_ceil(ALIGNMENT).max(1) * ALIGNMENT;
        }
        let options = TensorOptions::new().dtype(DType::U8).device(Device::Mps(0));
        let memory = Tensor::zeros(&[total], options);
        crate::stream::mps::synchronize();
        let base = memory.data_ptr();
        let pointer = |k: usize| base.wrapping_add(at[k]);
        let ninputs = step.inputs.len();
        let inputs = (0..ninputs)
            .chain(ninputs + 1..ninputs + 1 + step.extra_outputs.len())
            .map(|k| pointer(k).cast_const())
            .collect();
        Buffers {
            output: pointer(ninputs),
            scratch: pointer(sizes.len() - 1),
            memory,
            inputs,
        }
    }

    fn run(&self, step: &Step, reps: usize) -> Result<f64, String> {
        let launches = capture(|| {
            let keep = vec![self.memory.clone()];
            encode(step, &self.inputs, self.output, self.scratch, keep)
        })?;
        launches.iter().map(|l| l.time(reps)).sum()
    }
}
