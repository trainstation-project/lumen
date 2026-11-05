//! Running plan steps on TPU: each step's kernel, generated when its plan compiles, lowered once per device and
//! launched into the device's batch.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use super::{unsupported, viewed};
use crate::Device;
use crate::codegen::tpu::Kernel;
use crate::codegen::tpu::{Rows, kernel};
use crate::codegen::utils::tpu::Tpu;
use crate::codegen::utils::tpu::lowering::lowering;
use crate::graph::Primitive;
use crate::plan::run::Buffers;
use crate::plan::{Buffer, Step, StepBase};
use crate::runtime::kernel::Launch as _;
use crate::runtime::kernel::tpu::TpuLaunch;

/// A step of a Pallas kernel on a TPU.
#[derive(Debug, Clone)]
pub struct TpuStep {
    base: StepBase,
    /// The kernel codegen generated for the step; none when it has nothing to launch, or is a gather only the
    /// SparseCore may take.
    pub(crate) kernel: Option<Kernel>,
}

impl TpuStep {
    /// `base` with the kernel codegen generated for it.
    pub(crate) fn new(base: StepBase, kernel: Option<Kernel>) -> Self {
        TpuStep { base, kernel }
    }
}

impl Deref for TpuStep {
    type Target = StepBase;

    /// The step's primitive and buffers.
    fn deref(&self) -> &StepBase {
        &self.base
    }
}

impl DerefMut for TpuStep {
    /// The step's primitive and buffers.
    fn deref_mut(&mut self) -> &mut StepBase {
        &mut self.base
    }
}

impl Step for TpuStep {
    /// Runs the step on its TPU: its kernel lowered once per device and launched into the device's batch.
    fn run(&self, buffers: &Buffers) -> Result<(), String> {
        let Device::Tpu(index) = buffers.device else {
            unreachable!("a TPU plan runs on a TPU")
        };

        let step = &self.base;
        let inputs = &buffers.args(step);
        let output = buffers.pointer(&step.output);

        // A gather runs on the SparseCore when its buffers' rows allow, known only at launch.
        if let Primitive::Gather { .. } = step.primitive
            && crate::codegen::gather::tpu::sparse_core(index, step, inputs, output)?
        {
            return Ok(());
        }

        // A step without a kernel has nothing to launch, or is a gather neither kernel takes, which fails with its
        // TensorCore kernel's error.
        let Some(kernel) = &self.kernel else {
            return match step.primitive {
                Primitive::Gather { .. } => kernel(step).map(|_| ()),
                _ => Ok(()),
            };
        };

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

    /// Errors unless `device` is a TPU, no step reads runtime scalars or an operand through a view.
    fn check(steps: &[Self], device: Device) -> Result<(), String> {
        let runs = |d| matches!(d, Device::Tpu(_));

        if let Some(e) = unsupported(steps, device, runs, "a TPU").or_else(|| viewed(steps)) {
            return Err(e);
        }

        let scalar = steps.iter().find(|s| {
            !matches!(s.primitive, Primitive::CustomCall { .. })
                && s.inputs.iter().any(|(b, _)| matches!(b, Buffer::Scalar(_)))
        });

        match scalar {
            Some(step) => Err(format!(
                "{}: runtime scalars do not run on TPU yet",
                step.label
            )),
            None => Ok(()),
        }
    }

    /// Runs `run` in a batch on the TPU, then the batch's launches as one program.
    fn batched(device: Device, run: &mut dyn FnMut() -> Result<(), String>) -> Result<(), String> {
        let Device::Tpu(index) = device else {
            unreachable!("a TPU plan runs on a TPU")
        };

        let batch = crate::runtime::stream::tpu::Batch::new(index);
        run()?;
        batch.run()
    }

    /// The step without its kernel.
    fn into_base(self) -> StepBase {
        self.base
    }
}
