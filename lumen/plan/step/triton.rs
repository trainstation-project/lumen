//! Running plan steps on CUDA through Triton: a step dispatched to its kernel, and a loop step's helpers.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use super::unsupported;
use crate::Device;
use crate::plan::Step;
use crate::plan::run::Buffers;

use crate::codegen::attention::triton::launch as attention;
use crate::codegen::dot_general::triton::launch as gemm;
use crate::codegen::dynamic_slice::DynamicSliceCodegen;
use crate::codegen::elementwise::ElementwiseCodegen;
use crate::codegen::elementwise::triton::copy;
use crate::codegen::elementwise::triton::{LoopConfig, untuned_loop};
use crate::codegen::kinds::reduction_root;
use crate::codegen::layout::{LayoutCodegen, transpose_tiling};
use crate::codegen::row::triton::{row_count, row_groups, row_looped};
use crate::codegen::store::triton::{load, store};
use crate::codegen::triton::{Kernel, ROW, Template};
use crate::codegen::utils::triton as codegen;
use crate::graph::{Primitive, UPDATES};
use crate::plan::StepBase;
use crate::runtime::kernel::triton::{Function, Launches, launch};
use crate::{DType, Tensor, TensorOptions};

/// A step of Triton kernels on CUDA.
#[derive(Debug, Clone)]
pub struct TritonStep {
    base: StepBase,
    /// The loop kernel codegen generated for the step; none for a step another kernel (a GEMM's, flash
    /// attention's, a collective) runs.
    pub(crate) kernel: Option<Kernel>,
}

impl TritonStep {
    /// `base` with the loop kernel codegen generated for it.
    pub(crate) fn new(base: StepBase, kernel: Option<Kernel>) -> Self {
        TritonStep { base, kernel }
    }
}

impl Deref for TritonStep {
    type Target = StepBase;

    /// The step's primitive and buffers.
    fn deref(&self) -> &StepBase {
        &self.base
    }
}

impl DerefMut for TritonStep {
    /// The step's primitive and buffers.
    fn deref_mut(&mut self) -> &mut StepBase {
        &mut self.base
    }
}

/// Runs `run`, a step's launch labelled `label` writing `out`: captured into a CUDA graph, the nodes it adds are
/// the step's, by name; profiled, its kernels are timed under its name (each keeps its own as `kernel`).
pub(crate) fn traced(
    label: &'static str,
    out: &Tensor,
    run: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    #[cfg(all(feature = "python", lumen_cuda_linked))]
    let run = || crate::runtime::kernel::cuda::driver::counted(label, run);

    #[cfg(lumen_cupti_linked)]
    let run = || {
        crate::profiler::next_launch(label, Default::default);
        crate::profiler::cupti::correlated(out.device(), run)
    };

    let _ = (label, out);
    run()
}

impl Step for TritonStep {
    /// Runs the step on its tensors through its kernel.
    fn run(&self, buffers: &Buffers) -> Result<(), String> {
        let (inputs, out, outs) = buffers.tensors(self);
        traced(self.label, &out, || self.run_raw(&inputs, &out, &outs))
    }

    /// Whether the step's kernel reads operand `k`, a caller input of `shape`, as laid out (attention any layout
    /// with a contiguous last dim, copying others itself).
    fn reads_strided(&self, k: usize, shape: &[usize], _: &[usize]) -> bool {
        self.inputs[k].1.shape == shape
            && matches!(
                self.primitive,
                Primitive::DotGeneral { .. }
                    | Primitive::Fusion { .. }
                    | Primitive::Cumsum { .. }
                    | Primitive::LinearCrossEntropy
            )
    }

    /// Errors unless `device` is a CUDA device.
    fn check(steps: &[Self], device: Device) -> Result<(), String> {
        match unsupported(steps, device, |d| matches!(d, Device::Cuda(_)), "CUDA") {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The step without its kernel.
    fn into_base(self) -> StepBase {
        self.base
    }
}

impl TritonStep {
    /// Runs the step on the given tensors: GEMMs through the GEMM kernel, everything else through its loop kernel.
    pub(crate) fn run_raw(
        &self,
        inputs: &[Tensor],
        out: &Tensor,
        outs: &[Tensor],
    ) -> Result<(), String> {
        let (step, kernel) = (&self.base, self.kernel.as_ref());

        match (&step.primitive, kernel) {
            (p, _) if attention::is_attention(p) => attention::run(step, inputs, out, outs),
            _ if gemm::is_gemm(step) => gemm::run(step, inputs, out, outs),
            #[cfg(lumen_cuda_linked)]
            (p, _) if p.is_collective() => crate::distributed::nccl::run(p, inputs, out),
            (Primitive::DeviceWait, _) => {
                #[cfg(lumen_cuda_linked)]
                crate::distributed::nccl::device_wait(&inputs[0])?;

                match out.data_ptr() == inputs[0].data_ptr() {
                    true => Ok(()),
                    false => copy(out, &inputs[0]),
                }
            }
            (Primitive::HostWait, _) => {
                crate::runtime::stream::stream_for(out.device())?.synchronize();

                match out.data_ptr() == inputs[0].data_ptr() {
                    true => Ok(()),
                    false => copy(out, &inputs[0]),
                }
            }
            (
                Primitive::Fusion { .. }
                | Primitive::Cumsum { .. }
                | Primitive::DynamicUpdateSlice
                | Primitive::ScatterAdd { .. },
                Some(kernel),
            ) => {
                // A fusion's other outputs follow its inputs, each with its own store.
                let outs: &[Tensor] = match &step.primitive {
                    Primitive::Fusion { .. } => outs,
                    _ => &[],
                };

                if let Primitive::ScatterAdd { .. } = step.primitive
                    && out.dtype() == DType::Bool
                {
                    return Err(format!(
                        "{}: booleans have no sum to scatter-add",
                        step.label
                    ));
                }
                let key = layouts(inputs.iter().chain([out]).chain(outs));

                let helpers = step
                    .memo
                    .get(key, || Ok(loop_helpers(step, kernel, inputs, out, outs)))?;

                let (functions, launches) = (&helpers.functions, &helpers.launches);
                let inputs: &[Tensor] = &[inputs, outs].concat();
                let untuned = untuned_loop(kernel.lanes);
                let config = step.launch::<LoopConfig>().unwrap_or(&untuned);
                // A looped row kernel walks one row a program, a block of it at a time; one with column
                // partials one group of rows a program; a tiled transpose one tile a program.
                let looped;

                let kernel = match &step.primitive {
                    Primitive::Fusion { name, body, .. }
                        if name == ROW && (row_looped(body) || row_groups(body).is_some())
                            || name != ROW && transpose_tiling(body).is_some() =>
                    {
                        looped = Kernel {
                            lanes: config.block_size,
                            ..kernel.clone()
                        };

                        &looped
                    }
                    _ => kernel,
                };

                // Grid size: outputs, or a row fusion's rows (its row groups with partials), or a tiled
                // transpose's tiles, or lines for a scan (a scatter-add sizes its own launches).
                let n = match &step.primitive {
                    Primitive::Fusion { name, body, .. } if name == ROW => {
                        Some(row_groups(body).unwrap_or(row_count(body)))
                    }
                    Primitive::Fusion { body, .. } if transpose_tiling(body).is_some() => {
                        transpose_tiling(body).map(|tiling| tiling.tiles())
                    }
                    Primitive::Cumsum { axis, .. } => out.numel().checked_div(out.shape()[*axis]),
                    Primitive::DynamicUpdateSlice => Some(inputs[1].numel()),
                    Primitive::Fusion { name, .. } if name == UPDATES => Some(inputs[1].numel()),
                    _ => Some(out.numel()),
                };

                // In-place ops start from a copy of their operand.
                let updates =
                    matches!(&step.primitive, Primitive::Fusion { name, .. } if name == UPDATES);

                if (updates
                    || matches!(
                        step.primitive,
                        Primitive::DynamicUpdateSlice | Primitive::ScatterAdd { .. }
                    ))
                    && out.data_ptr() != inputs[0].data_ptr()
                {
                    copy(out, &inputs[0])?;
                }

                if let Some(Template::ScatterAdd { .. }) = kernel.template {
                    return crate::codegen::scatter_add::triton::launch::run(
                        kernel, launches, inputs, out,
                    );
                }

                let n = n.unwrap_or(0);

                if kernel.splits == 1 {
                    return launch(kernel, config, functions, launches, inputs, out, n);
                }

                // A split reduction writes per-split partials, then a second kernel combines them into `out`.
                let Primitive::Fusion { body, .. } = &step.primitive else {
                    unreachable!("a split reduction")
                };

                let reduced = reduction_root(body).expect("a reduction").output;

                let options = TensorOptions::new()
                    .dtype(body.type_of(reduced).dtype)
                    .device(out.device());

                let partials = unsafe { Tensor::empty(&[n * kernel.splits], options) };
                // The first pass also writes the fusion's other outputs; the second only the result.
                launch(kernel, config, functions, launches, inputs, &partials, n)?;

                let last = Kernel {
                    name: format!("{}_final", kernel.name),
                    source: kernel.source.clone(),
                    by_value: [kernel.by_value.as_slice(), &[false]].concat(),
                    lanes: kernel.splits.next_power_of_two(),
                    splits: 1,
                    segments: Vec::new(),
                    template: None,
                };

                let inputs = [inputs, std::slice::from_ref(&partials)].concat();

                let finals: Vec<Function> = functions
                    .iter()
                    .filter(|f| f.param == "STORE" || !f.param.starts_with("STORE"))
                    .cloned()
                    .collect();

                launch(&last, config, &finals, launches, &inputs, out, n)
            }
            _ => Err(format!("{}: has no Triton kernel on CUDA yet", step.label)),
        }
    }
}

/// A loop step's generated helpers for its tensors' layouts, and the kernels compiled for them.
struct Helpers {
    /// The loads of its tensor inputs, then the stores of its outputs; a templated step's `BODY` instead.
    functions: Vec<Function>,
    launches: Launches,
}

/// A loop step's helpers for its tensors' layouts: the loads of its tensor inputs, the stores of `out`
/// and `outs`, or a templated step's `BODY`.
fn loop_helpers(
    step: &StepBase,
    kernel: &Kernel,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
) -> Helpers {
    // A scatter-add's kernel is fixed: its tensors are contiguous, as it reads no strided inputs.
    if let Some(Template::ScatterAdd { .. }) = kernel.template {
        return Helpers {
            functions: Vec::new(),
            launches: Launches::default(),
        };
    }

    // A scan's kernel is fixed too, its tensors loaded and stored through the helpers below.
    if let Some(template) = kernel.template.filter(|t| !matches!(t, Template::Scan(_))) {
        let owned = |t: &Tensor| (t.shape().to_vec(), t.strides().to_vec());

        let layouts = codegen::Tensors {
            inputs: inputs
                .iter()
                .zip(&kernel.by_value)
                .map(|(t, &scalar)| (!scalar).then(|| owned(t)))
                .collect(),
            outputs: std::iter::once(out).chain(outs).map(owned).collect(),
        };

        let generated = match (template, &step.primitive) {
            (Template::Elementwise, Primitive::Fusion { body, .. }) => {
                codegen::Triton::elementwise(body, &kernel.by_value, &HashMap::new(), &layouts)
                    .expect("a Triton kernel of every elementwise body")
            }
            (Template::Transpose { .. }, Primitive::Fusion { body, .. }) => {
                codegen::Triton::transpose(body, &kernel.by_value, &layouts)
                    .expect("a Triton kernel of every tiled transpose")
            }
            (Template::DynamicUpdateSlice, Primitive::Fusion { body, .. }) => {
                codegen::Triton::updates(body, &layouts)
                    .expect("a Triton kernel of every updates body")
            }
            (Template::DynamicUpdateSlice, _) => codegen::Triton::update(
                &step.inputs[0].1,
                &step.inputs[1].1,
                &kernel.by_value,
                &layouts,
            )
            .expect("a Triton kernel of every dynamic_update_slice"),
            _ => unreachable!("a templated kernel's fusion"),
        };

        return Helpers {
            functions: vec![Function::new("BODY", generated)],
            launches: Launches::default(),
        };
    }

    let mut functions: Vec<Function> = inputs
        .iter()
        .enumerate()
        .filter(|&(k, _)| !kernel.by_value[k])
        .map(|(k, t)| Function::new(format!("LOAD{k}"), load(t.shape(), t.strides())))
        .collect();

    functions.push(Function::new("STORE", store(out.shape(), out.strides())));

    for (k, t) in outs.iter().enumerate() {
        functions.push(Function::new(
            format!("STORE{}", k + 1),
            store(t.shape(), t.strides()),
        ));
    }

    Helpers {
        functions,
        launches: Launches::default(),
    }
}

/// A hash of `tensors`' dtypes, shapes and strides: the key of what a step generated for those layouts.
pub(crate) fn layouts<'a>(tensors: impl IntoIterator<Item = &'a Tensor>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    for t in tensors {
        (t.dtype(), t.shape(), t.strides()).hash(&mut hasher);
    }

    hasher.finish()
}
