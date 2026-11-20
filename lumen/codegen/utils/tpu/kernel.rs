//! TPU codegen: each planned step's kernel, generated once when the plan compiles and kept in the step's
//! memo; the runtime lowers it once per device and launches it.

use super::Tpu;
use super::lowering::lowering;
use crate::codegen::elementwise::tpu::elementwise;
use crate::codegen::gather::tpu::Gather;
use crate::codegen::kernel::Runner;
use crate::graph::Primitive;
use crate::graph_compiler::Options;
use crate::plan::step::tpu::Pointers;
use crate::plan::{CpuPlan, Step as _, StepBase, TpuPlan, TpuStep};
use crate::runtime::kernel::Launch as _;
use crate::runtime::kernel::tpu::TpuLaunch;
use std::sync::Arc;

/// The `module` of a launch whose `spec` is the generated source of its program rather than a spec for a
/// module's.
pub(crate) const GENERATED: &str = "generated";

/// How a kernel's buffers are ordered among a step's input and output buffers.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Rows {
    /// The step's inputs, its output, then its extra buffers.
    Split,
    /// Every input buffer, then the output.
    Last,
    /// The step's inputs, then its output.
    Step,
}

/// A step's TPU kernel: generated once, when its plan compiles, and lowered once per device.
#[derive(Debug, Clone)]
pub(crate) struct Kernel {
    /// Python module whose `program(spec)` lowers the kernel, or `GENERATED` for a generated program.
    pub(crate) module: &'static str,
    /// Name shown in profiles.
    pub(crate) kernel: String,
    /// JSON object fields describing the kernel (without braces), or a generated program's source.
    pub(crate) spec: String,
    /// The spec for when the step's first input isn't its output buffer, so the kernel copies it first.
    pub(crate) copied: Option<String>,
    /// Extra `i32` scalars after the row offsets.
    pub(crate) scalars: Vec<i32>,
    /// How its buffers are ordered.
    pub(crate) rows: Rows,
    /// What a failed launch's error starts with.
    pub(crate) context: Option<String>,
}

impl Kernel {
    /// A kernel of `module` named `kernel` with `spec`, its buffers ordered by `rows`.
    pub(crate) fn new(module: &'static str, kernel: String, spec: String, rows: Rows) -> Kernel {
        Kernel {
            module,
            kernel,
            spec,
            copied: None,
            scalars: Vec::new(),
            rows,
            context: None,
        }
    }
}

/// Plan `step`'s TPU kernel, by its primitive or fusion kind; `None` when it has nothing to launch.
pub(crate) fn kernel(step: &StepBase) -> Result<Option<Kernel>, String> {
    match step.primitive {
        Primitive::Fusion { ref name, .. } if name.starts_with("flash_attention") => {
            crate::codegen::attention::tpu::step_kernel(step).map(Some)
        }
        Primitive::Fusion { ref name, .. } if name == "reduction" => {
            crate::codegen::reduction::tpu::kernel(step).map(Some)
        }
        Primitive::Fusion { ref name, .. } if name == "dot_row" => {
            crate::codegen::dot_general::tpu::row::kernel(step).map(Some)
        }
        Primitive::Fusion { ref name, .. } if name == "row" => {
            crate::codegen::row::tpu::kernel(step).map(Some)
        }
        Primitive::Fusion { ref name, .. } if name == crate::graph::UPDATES => {
            crate::codegen::dynamic_slice::tpu::kernel(step)
        }
        Primitive::Fusion { ref name, .. } if name == "elementwise" => elementwise(step),
        Primitive::DotGeneral { .. } | Primitive::Fusion { .. } => {
            crate::codegen::dot_general::tpu::kernel(step)
        }
        Primitive::Reshape { .. } => crate::codegen::layout::tpu::kernel(step).map(Some),
        Primitive::Gather { .. } => crate::codegen::gather::tpu::kernel(step).map(Some),
        Primitive::DynamicUpdateSlice => crate::codegen::dynamic_slice::tpu::kernel(step),
        Primitive::PallasCall { kernel, .. } => {
            // Pallas kernels see their inputs, then the output, then any scratch buffers.
            let mut pallas = Kernel::new(
                "lumen.kernels.pallas",
                step.primitive.name().to_owned(),
                format!("\"kernel\": {kernel}"),
                Rows::Split,
            );

            pallas.context = Some(step.label.to_string());
            Ok(Some(pallas))
        }
        // Custom calls run their registered kernels, not a TPU program.
        Primitive::CustomCall { .. } => Ok(None),
        _ => Err(format!("{} does not run on TPU yet", step.primitive.name())),
    }
}

/// The TPU plan of `plan`: each step with its kernel, generated now so runs only launch them; fails on a step no
/// TPU kernel takes (a gather, only once its launch finds the SparseCore can't run it either).
pub(crate) fn codegen(plan: CpuPlan, _: &Options) -> Result<TpuPlan, String> {
    plan.map(|step| {
        let kernel = match kernel(&step) {
            Ok(kernel) => kernel,
            // A gather its TensorCore kernel can't take may still run on the SparseCore, which its launch decides.
            Err(_) if matches!(step.primitive, Primitive::Gather { .. }) => None,
            Err(e) => return Err(e),
        };

        let base = step.into_base();

        let runner: Arc<dyn Runner<TpuStep, Pointers>> = match (&base.primitive, &kernel) {
            (Primitive::Gather { .. }, _) => Arc::new(Gather),
            (_, Some(_)) => Arc::new(Pallas),
            (_, None) => Arc::new(Nothing),
        };

        Ok(TpuStep::new(base, kernel, runner))
    })
}

/// A step's generated Pallas kernel: lowered once per device (kept in the step's memo) and launched into the
/// device's batch.
pub(crate) struct Pallas;

impl crate::codegen::kernel::Kernel for Pallas {
    type Step = TpuStep;
    type Buffers = Pointers;
    type Config = ();

    /// Launches the step's kernel on its buffers.
    fn run(&self, step: &TpuStep, _: Option<&()>, p: &Pointers) -> Result<(), String> {
        let (index, inputs, output) = (p.index, &p.inputs, p.output);
        let kernel = step.code.as_ref().expect("a step's generated kernel");

        let n = step.inputs.len();
        let output = output.cast_const();

        let rows: Vec<*const u8> = match kernel.rows {
            Rows::Split => inputs[..n]
                .iter()
                .copied()
                .chain([output])
                .chain(inputs[n..].iter().copied())
                .collect(),
            Rows::Last => inputs.iter().copied().chain([output]).collect(),
            Rows::Step => inputs[..n].iter().copied().chain([output]).collect(),
        };

        let copy = kernel.copied.is_some() && inputs[0] != output;
        let scalars = rows.len() + kernel.scalars.len();
        let device = crate::allocator::tpu::rows(index)?;

        let key = {
            use std::hash::{Hash, Hasher};

            let mut h = std::collections::hash_map::DefaultHasher::new();
            (index, device, copy).hash(&mut h);
            h.finish()
        };

        let launched = || -> Result<(), String> {
            let (mosaic, spec) = &*step.memo.get(key, || {
                let fields = match (copy, &kernel.copied) {
                    (true, Some(copied)) => copied,
                    _ => &kernel.spec,
                };

                lowering(index, kernel.module, fields, scalars, None)
                    .map_err(|e| format!("{}: {e}", kernel.kernel))
            })?;

            Tpu::launch(TpuLaunch {
                device: index,
                module: kernel.module,
                kernel: &kernel.kernel,
                spec: String::new(),
                rows: &rows,
                scalars: &kernel.scalars,
                host: &[],
                sparse: None,
                lowered: Some((Arc::clone(mosaic), Arc::clone(spec))),
            })
        };

        launched().map_err(|e| match &kernel.context {
            Some(context) => format!("{context}: {e}"),
            None => e,
        })
    }
}

/// A step with nothing to launch on the TPU (a custom call runs its registered kernel).
pub(crate) struct Nothing;

impl crate::codegen::kernel::Kernel for Nothing {
    type Step = TpuStep;
    type Buffers = Pointers;
    type Config = ();

    /// Does nothing.
    fn run(&self, _: &TpuStep, _: Option<&()>, _: &Pointers) -> Result<(), String> {
        Ok(())
    }
}
