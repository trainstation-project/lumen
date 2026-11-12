//! MPS codegen: once the plan is laid out, generates Metal source for every requested kernel and gives each
//! fusion step its kernel name; and the kernel shape (GEMV row, GEMM, row reduction, reduction, transpose or
//! elementwise) each fused body lowers to.

use crate::autotune::mps::Tunable;
use crate::codegen::attention::AttentionCodegen;
use crate::codegen::dot_general::mps::{gemm_kernel, gemm_row_kernel, gemv_config, row_dot};
use crate::codegen::dynamic_slice::DynamicSliceCodegen;
use crate::codegen::elementwise::ElementwiseCodegen;
use crate::codegen::elementwise::mps::{Tuning, elementwise_tuning, elementwise_tunings};
use crate::codegen::kinds::{gemm_dot, reduction_root, row_reductions};
use crate::codegen::layout::{LayoutCodegen, transpose_tiling};
use crate::codegen::reduction::ReductionCodegen;
use crate::codegen::row::RowCodegen;
use crate::codegen::utils::mps::Mps;
use crate::graph::{Graph, Primitive, TensorType};
use crate::graph_compiler::device::{Device, DeviceModule};
use crate::graph_compiler::passes::attention::{self, Strides};
use crate::graph_compiler::passes::place_views;
use crate::graph_compiler::{CompilerConfig, Options};
use crate::plan::{Buffer, CpuPlan, MpsPlan, MpsStep, Plan, PlanOptions, Step as _, StepBase};
use crate::runtime::kernel::mps::encode::{compile_routed, record_launch};
use std::collections::{BTreeMap, HashMap};

/// The MPS plan of `plan`: generates and compiles each pending fusion step's kernel, writing its outputs through
/// the step's output views and reading its by-value inputs as scalars, and names the step after it. An
/// elementwise fusion keeps what the autotuner needs to regenerate it as its step's kernel.
pub(crate) fn codegen(mut plan: CpuPlan, options: &Options) -> Result<MpsPlan, String> {
    let config = &options.config;
    let mut sources = BTreeMap::new();
    let mut kernels: Vec<Option<Tunable>> = plan.steps().iter().map(|_| None).collect();

    for (at, step) in plan.steps_mut().iter_mut().map(|s| &mut s.0).enumerate() {
        // Output views, keyed by output index, of a fusion writing straight into a placed buffer.
        let mut views: HashMap<usize, (usize, Vec<usize>)> = step
            .output_views
            .iter()
            .enumerate()
            .filter_map(|(k, v)| v.as_ref().map(|v| (k, (v.offset, v.strides.clone()))))
            .collect();

        let by_value: Vec<bool> = step
            .inputs
            .iter()
            .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
            .collect();

        let Primitive::Fusion { name, body, .. } = &mut step.primitive else {
            continue;
        };

        if name == crate::graph::UPDATES {
            let (kernel, source) =
                Mps::updates(body, &()).map_err(|e| format!("{}: {e}", step.label))?;

            record_launch(&kernel, body, None);
            sources.insert(kernel.clone(), source);
            *name = kernel;
            continue;
        }

        let Some(requested) = Kernel::of(name) else {
            continue;
        };

        let (kernel, source) = match requested {
            Kernel::Fusion => {
                let (kernel, source) = kernel(body, &by_value, config, &views, None)
                    .map_err(|e| format!("{}: {e}", step.label))?;
                kernels[at] = Some(Tunable { by_value, views });
                (kernel, source)
            }
            Kernel::Attention => {
                let a = attention::of_body(body).expect("an attention");
                Mps::forward(body, &a, &Strides::default())?
            }
            Kernel::AttentionBackward => {
                let strides = Strides {
                    outputs: std::mem::take(&mut views),
                    ..Strides::default()
                };

                let b = attention::backward_of_body_strided(body, &strides)
                    .expect("an attention backward");

                Mps::backward(body, &b, &strides)?
            }
        };

        record_launch(&kernel, body, tuning_of(body));
        sources.insert(kernel.clone(), source);
        *name = kernel;
    }

    compile_routed(sources)?;
    let mut kernels = kernels.into_iter();
    plan.map(|step| Ok(MpsStep::new(step.into_base(), kernels.next().flatten())))
}

/// Default launch tuning when `body` lowers to the elementwise kernel, else `None`.
pub(crate) fn tuning_of(body: &Graph) -> Option<Tuning> {
    is_elementwise(body).then(|| elementwise_tuning(body))
}

/// Candidate launch tunings to autotune when `body` lowers to the elementwise kernel.
pub(crate) fn tunings_of(body: &Graph) -> Vec<Tuning> {
    match is_elementwise(body) {
        true => elementwise_tunings(body),
        false => Vec::new(),
    }
}

/// Whether `body` matches no specialized kernel shape and lowers to the elementwise kernel.
fn is_elementwise(body: &Graph) -> bool {
    row_dot(body).is_none()
        && gemm_dot(body).is_none()
        && row_reductions(body).is_empty()
        && reduction_root(body).is_none()
        && transpose_tiling(body).is_none()
}

/// Generates the kernel for a fused `body`, returning its name and Metal source.
/// `by_value[k]` passes input `k` as a constant scalar; `views` maps output index to the
/// (element offset, strides) it writes through. Fails on a body no Metal kernel takes.
pub(crate) fn kernel(
    body: &Graph,
    by_value: &[bool],
    config: &CompilerConfig,
    views: &HashMap<usize, (usize, Vec<usize>)>,
    tuning: Option<Tuning>,
) -> Result<(String, String), String> {
    if let Some(dot) = row_dot(body) {
        return Ok(gemm_row_kernel(
            body,
            by_value,
            dot,
            config,
            gemv_config(body),
        ));
    }

    if let Some(dot) = gemm_dot(body) {
        return Ok(gemm_kernel(body, by_value, dot));
    }

    if !row_reductions(body).is_empty() {
        return Mps::row(body, by_value, config);
    }

    if let Some(root) = reduction_root(body) {
        return Mps::reduction(body, by_value, root);
    }

    if transpose_tiling(body).is_some() {
        return Mps::transpose(body, by_value, views);
    }

    let tuning = tuning.unwrap_or_else(|| elementwise_tuning(body));
    Mps::elementwise(body, by_value, views, &tuning)
}

/// Whether the kernel chosen for `body` can write its outputs through strided views.
pub(crate) fn writes_views(body: &Graph) -> bool {
    gemm_dot(body).is_none() && row_reductions(body).is_empty() && reduction_root(body).is_none()
}

/// Whether fusion input `k` may be bound as a strided view: it is not attention and no dot reads it.
pub(crate) fn loads_strided(body: &Graph, k: usize) -> bool {
    let Some(&v) = body.inputs().get(k) else {
        return false;
    };

    let template = |p: &Primitive| matches!(p, Primitive::DotGeneral { .. });

    attention::of_body(body).is_none()
        && attention::backward_of_body(body).is_none()
        && !body
            .nodes()
            .iter()
            .any(|n| template(&n.primitive) && n.inputs.contains(&v))
}

/// Whether some input of `step` is a view its kernel loads element by element (a strided variant of the
/// kernel), which tuned kernels don't; views a GEMM's dot strides through need none.
pub(crate) fn loads_views(step: &StepBase) -> bool {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        return step.views.iter().any(Option::is_some);
    };

    step.views
        .iter()
        .enumerate()
        .any(|(k, view)| view.is_some() && loads_strided(body, k))
}

/// Whether an attention fusion reads its inputs at `strides`: any, with a contiguous last dim (its
/// kernel is generated for the layout when it runs).
pub(crate) fn attention_reads_strided(body: &Graph, strides: &[usize]) -> bool {
    strides.last().is_none_or(|&s| s == 1)
        && (attention::of_body(body).is_some() || attention::backward_of_body(body).is_some())
}

/// Whether GEMM fusion input `k` is read only by the dot, as one operand, in a layout MPS's matmul
/// kernels stride through: an attention's dots are its kernel's, not a GEMM's.
pub(crate) fn fusion_reads_strided(body: &Graph, k: usize, strides: &[usize]) -> bool {
    attention::of_body(body).is_none()
        && attention::backward_of_body(body).is_none()
        && place_views::fusion_reads_strided(
            body,
            k,
            strides,
            crate::codegen::dot_general::mps::reads_strided,
        )
}

/// Scratch bytes a step's kernels need beyond its output, for primitives that use any.
pub(crate) fn scratch_bytes(p: &Primitive, inputs: &[&TensorType], _output: &TensorType) -> usize {
    match p {
        Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => {
            let accum = crate::codegen::reduction::mps::accum_dtype(p, inputs[0].dtype);
            crate::codegen::reduction::mps::scratch_bytes(inputs[0], axes, accum)
        }
        Primitive::Cumsum {
            axis, accum_dtype, ..
        } => crate::codegen::scan::mps::scratch_bytes(inputs[0], *axis, *accum_dtype),
        Primitive::Fusion { body, .. } => {
            crate::runtime::kernel::mps::encode::fusion_scratch_bytes(body)
        }
        Primitive::LinearCrossEntropy => {
            crate::codegen::dot_general::mps::linear_cross_entropy_scratch(inputs[0], inputs[1])
        }
        Primitive::ScatterAdd { axis } => {
            crate::codegen::scatter_add::mps::scratch_bytes(inputs[0], inputs[1], *axis)
        }
        _ => 0,
    }
}

/// Kind of kernel a fusion requests, generated later by codegen.
pub(crate) enum Kernel {
    /// A fusion's kernel generated from its body.
    Fusion,
    /// Flash attention forward.
    Attention,
    /// Flash attention backward.
    AttentionBackward,
}

impl Kernel {
    /// The name a fusion awaiting this kind of kernel goes by until codegen.
    fn pending(&self) -> &'static str {
        match self {
            Kernel::Fusion => "pending#fusion",
            Kernel::Attention => "pending#attention",
            Kernel::AttentionBackward => "pending#attention_backward",
        }
    }

    /// The kind of kernel a fusion named `name` awaits, or `None` once it has one.
    pub(crate) fn of(name: &str) -> Option<Kernel> {
        [Kernel::Fusion, Kernel::Attention, Kernel::AttentionBackward]
            .into_iter()
            .find(|k| k.pending() == name)
    }
}

/// Requests a kernel, returning the placeholder name a fusion uses until codegen generates it.
pub(crate) fn request(kernel: Kernel) -> String {
    kernel.pending().to_owned()
}

/// MPS rejects ops Metal has no kernel for; it plans its passed graph with scratch, generates and compiles the
/// kernels, optionally autotunes them, and labels the dots reading packed weight blocks.
impl Device for Mps {
    const NAME: &'static str = "mps";

    /// Plans `module`'s graph, generates its kernels and autotunes them.
    fn lower(module: DeviceModule<'_, Self>, _: crate::Device) -> Result<Box<dyn Plan>, String> {
        Ok(Box::new(lower(module)?))
    }
}

/// Compiles `graph` for MPS as `device::compile` does, keeping the plan's MPS type for tests to inspect.
#[cfg(test)]
pub(crate) fn compile(graph: &Graph, options: &Options) -> Result<MpsPlan, String> {
    let mut module = DeviceModule::<Mps>::new(graph, options);
    crate::graph_compiler::pipeline::pipeline!(Mps, &options.config).run(&mut module)?;
    lower(module)
}

/// Plans MPS `module`'s passed graph with scratch, generates and compiles the kernels, optionally autotunes, and
/// labels the dots reading packed weight blocks.
fn lower(module: DeviceModule<'_, Mps>) -> Result<MpsPlan, String> {
    let options = module.options;
    let config = &options.config;

    let DeviceModule {
        graph: fused,
        packed,
        interleaved,
        scalars,
        ..
    } = module;

    let plan = PlanOptions {
        scratch: Some(scratch_bytes),
        ..crate::graph_compiler::plan_options(&fused, options, config, &packed, scalars)
    };

    let mut plan = codegen(CpuPlan::compile_with(&fused, &plan), options)?;

    if config.autotune || crate::autotune::forcing() {
        let tuner = crate::autotune::Autotuner {
            timer: crate::autotune::mps::Mps,
            autotune: config.autotune,
        };

        let mut elementwise = crate::autotune::mps::Elementwise {
            config,
            sources: BTreeMap::new(),
        };

        tuner.tune(&mut elementwise, &mut plan)?;
        tuner.tune(&mut crate::autotune::mps::Gemms::new(config), &mut plan)?;
        tuner.tune(&mut crate::autotune::mps::Attentions::default(), &mut plan)?;
    }

    // Relabel dots reading packed weight blocks (directly or through same-sized elementwise steps) as `Nx dot_general`.
    let first_block = fused.inputs().len() - packed.len();
    let mut derived: Vec<(Buffer, usize)> = Vec::new();

    for step in plan.steps_mut() {
        let whole = step
            .inputs
            .iter()
            .zip(&step.views)
            .filter(|(_, v)| v.is_none());

        let block = whole.map(|(input, _)| input).find_map(|(b, _)| match *b {
            Buffer::Input(i) if i >= first_block => Some(i - first_block),
            b => derived.iter().find(|(d, _)| *d == b).map(|&(_, k)| k),
        });

        let outputs: Vec<Buffer> = std::iter::once(&step.output)
            .chain(&step.extra_outputs)
            .map(|(b, _)| *b)
            .collect();

        derived.retain(|(d, _)| !outputs.contains(d));
        let gemm =
            matches!(&step.primitive, Primitive::Fusion { body, .. } if gemm_dot(body).is_some());

        if matches!(step.primitive, Primitive::DotGeneral { .. }) || gemm {
            if let Some((k, rest)) = block.zip(step.label.strip_prefix("dot_general")) {
                step.label =
                    crate::graph::intern(format!("{}x dot_general{rest}", packed[k].0.len()));
            }
        } else if block.is_some() {
            // Each output (a horizontal fusion's of several sizes too) derives from the block
            // when the inputs it is computed from are the block or same-sized.
            let types = std::iter::once(&step.output).chain(&step.extra_outputs);
            let mut found = Vec::new();

            for (p, (buffer, ty)) in types.enumerate() {
                let reads = feeding(&step.primitive, p, step.inputs.len());
                let n = ty.numel();

                let of_block = reads.iter().all(|&i| {
                    let m = step.inputs[i].1.numel();
                    m == n || m == 1
                });

                let block = reads
                    .iter()
                    .filter(|&&i| step.views[i].is_none())
                    .find_map(|&i| match step.inputs[i].0 {
                        Buffer::Input(i) if i >= first_block => Some(i - first_block),
                        b => derived.iter().find(|(d, _)| *d == b).map(|&(_, k)| k),
                    });

                if let Some(k) = block.filter(|_| of_block) {
                    found.push((*buffer, k));
                }
            }

            derived.extend(found);
        }
    }

    plan.packed = packed;
    plan.interleaved = interleaved;
    Ok(plan)
}

/// Positions of the step inputs that output `p` of `primitive` is computed from: those its fusion
/// body reads for that output, or every input of another primitive.
fn feeding(primitive: &Primitive, p: usize, inputs: usize) -> Vec<usize> {
    let Primitive::Fusion { body, .. } = primitive else {
        return (0..inputs).collect();
    };

    let mut producer = vec![None; body.types.len()];

    for node in body.nodes() {
        producer[node.output] = Some(node);
    }

    let (mut seen, mut stack) = (vec![false; body.types.len()], vec![body.outputs()[p]]);
    let mut reads = Vec::new();

    while let Some(v) = stack.pop() {
        if std::mem::replace(&mut seen[v], true) {
            continue;
        }

        match producer[v] {
            Some(node) => stack.extend(&node.inputs),
            None => reads.extend(body.inputs().iter().position(|&i| i == v)),
        }
    }

    reads
}
