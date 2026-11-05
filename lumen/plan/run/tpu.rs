//! Running plan steps on TPU.

use std::sync::Arc;

use crate::codegen::tpu::{Kernel, Rows, kernel};
use crate::codegen::utils::tpu::Tpu;
use crate::graph::Primitive;
use crate::plan::Step;
use crate::runtime::kernel::Launch as _;
use crate::runtime::kernel::tpu::TpuLaunch;
use crate::runtime::tpu::lowering;

/// Runs plan `step` on TPU device `index`: its kernel generated once, lowered once per device.
pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    // A gather runs on the SparseCore when its buffers' rows allow, known only at launch.
    if let Primitive::Gather { .. } = step.primitive
        && crate::codegen::gather::tpu::sparse_core(index, step, inputs, output)?
    {
        return Ok(());
    }

    // A step keeps the kernel codegen generated for it; one without has nothing to launch, or is a gather
    // neither kernel takes, which fails with its TensorCore kernel's error.
    let Some(kernel) = step
        .kernel
        .as_deref()
        .and_then(|k| k.downcast_ref::<Kernel>())
    else {
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
