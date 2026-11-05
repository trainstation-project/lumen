//! TPU codegen: each planned step's kernel, generated once when the plan compiles and kept in the step's
//! memo; the runtime lowers it once per device and launches it.

use std::collections::HashMap;

use crate::codegen::elementwise::ElementwiseCodegen;
use crate::codegen::utils::tpu::Tpu;
use crate::graph::Primitive;
use crate::plan::{Plan, Step};
use crate::runtime::tpu::GENERATED;

/// Memo key of a step's TPU kernel.
pub(crate) const KERNEL: u64 = u64::MAX;

/// How a kernel's buffers are ordered among a step's input and output buffers.
#[derive(Clone, Copy)]
pub(crate) enum Rows {
    /// The step's inputs, its output, then its extra buffers.
    Split,
    /// Every input buffer, then the output.
    Last,
    /// The step's inputs, then its output.
    Step,
}

/// A step's TPU kernel: generated once, when its plan compiles, and lowered once per device.
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

/// The elementwise fusion `step`'s kernel; `None` when it writes nothing.
fn elementwise(step: &Step) -> Result<Option<Kernel>, String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("an elementwise fusion")
    };

    if step.output.1.numel() == 0 {
        return Ok(None);
    }

    let views: HashMap<usize, (usize, Vec<usize>)> = step
        .output_views
        .iter()
        .enumerate()
        .filter_map(|(k, v)| v.as_ref().map(|v| (k, (v.offset, v.strides.clone()))))
        .collect();

    let code =
        Tpu::elementwise(body, &[], &views, &()).map_err(|e| format!("{}: {e}", step.label))?;

    // An indexed elementwise kernel takes its inputs, then its output; a plain one its output after the step's own inputs.
    let rows = match code.indexed {
        true => Rows::Last,
        false => Rows::Split,
    };

    Ok(Some(Kernel::new(GENERATED, code.kernel, code.source, rows)))
}

/// Plan `step`'s TPU kernel, by its primitive or fusion kind; `None` when it has nothing to launch.
pub(crate) fn kernel(step: &Step) -> Result<Option<Kernel>, String> {
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
                "lumen.pallas",
                step.primitive.name().to_owned(),
                format!("\"kernel\": {kernel}"),
                Rows::Split,
            );

            pallas.context = Some(step.label.to_string());
            Ok(Some(pallas))
        }
        _ => Err(format!("{} does not run on TPU yet", step.primitive.name())),
    }
}

/// Generates the kernel of each step of `plan` (and of its device stages), so runs only launch them.
pub(crate) fn codegen(plan: &Plan) {
    let staged = plan.staged.iter().flat_map(|s| &s.stages);

    for step in plan.steps() {
        // A step its kernel fails on keeps failing, with the same error, when it runs.
        let _ = step.memo.get(KERNEL, || kernel(step));
    }

    for stage in staged.filter(|s| !s.host) {
        codegen(&stage.plan);
    }
}
