//! Launching a TPU kernel.

use super::Launch;
use crate::codegen::utils::tpu::Tpu;
use std::sync::Arc;

use crate::runtime::tpu::{BATCHES, GENERATED, Lowered, SparseCore, lowering, run};

/// One TPU launch: the device it runs on, the kernel's lowering module (or generated program) and spec, its
/// buffers, extra scalars and host data.
pub(crate) struct TpuLaunch<'a> {
    /// The TPU it runs on.
    pub(crate) device: usize,
    /// Python module whose `program(spec)` lowers the kernel, or `GENERATED` for a generated program.
    pub(crate) module: &'static str,
    /// Name shown in profiles.
    pub(crate) kernel: &'a str,
    /// JSON object fields describing the kernel (without braces), to which the device fields are added; a
    /// generated program's source.
    pub(crate) spec: String,
    /// Buffers the kernel reads and writes, passed as their row offsets.
    pub(crate) rows: &'a [*const u8],
    /// Extra `i32` scalars after the row offsets.
    pub(crate) scalars: &'a [i32],
    /// Host bytes passed as whole extra rows.
    pub(crate) host: &'a [u8],
    /// Runs on the SparseCore when set.
    pub(crate) sparse: Option<SparseCore>,
    /// Its Mosaic and full spec, when the caller lowered it already (a step's kernel, lowered once per device);
    /// `spec` is then unused.
    pub(crate) lowered: Option<(Arc<str>, Arc<str>)>,
}

/// A TPU lowers each kernel to Mosaic once and batches it into the program it runs on flush.
impl Launch for Tpu {
    type Args<'a> = TpuLaunch<'a>;

    /// Lowers `launch` (unless its caller did) and adds it to the open batch on its device, or runs it now. A launch
    /// with host data runs on its own after flushing the batch.
    fn launch(launch: TpuLaunch<'_>) -> Result<(), String> {
        let index = launch.device;
        let scalars = launch.rows.len() + launch.scalars.len();
        let host =
            (!launch.host.is_empty()).then_some(launch.host.len() / crate::allocator::tpu::ROW);

        let (mosaic, spec) = match launch.lowered {
            Some(lowered) => lowered,
            None => lowering(index, launch.module, &launch.spec, scalars, host)
                .map_err(|e| format!("{}: {e}", launch.kernel))?,
        };

        let lowered = Lowered {
            mosaic,
            module: launch.module,
            spec,
            scalars: [
                crate::allocator::tpu::rows_of(index, launch.rows)?.as_slice(),
                launch.scalars,
            ]
            .concat(),
            sparse: launch.sparse.map(Box::new),
        };

        let (kernel, host) = (launch.kernel, launch.host);
        crate::profiler::tpu::launched(index, kernel);

        crate::profiler::launched::templated(kernel, || {
            // A generated program's script is its source, run on this device's memory.
            let template = match lowered.module {
                GENERATED => {
                    let rows = crate::allocator::tpu::rows(index).unwrap_or_default();
                    let host =
                        (!host.is_empty()).then_some(host.len() / crate::allocator::tpu::ROW);
                    let host = host.map_or("None".to_string(), |h| h.to_string());
                    let scalars = lowered.scalars.len();

                    let device = format!(
                        "{{\"scalars\": {scalars}, \"host_rows\": {host}, \"rows\": {rows}}}"
                    );

                    let script = format!("{}\nmosaic = program({device})\n", lowered.spec);
                    (GENERATED.to_owned(), script)
                }
                _ => (lowered.module.to_owned(), lowered.spec.to_string()),
            };

            (Some(lowered.mosaic.to_string()), Some(template))
        });

        let batched = BATCHES.with(|b| match b.borrow_mut().last_mut() {
            Some((i, launches)) if *i == index => match host.is_empty() {
                true => {
                    launches.push(lowered);
                    Ok(None)
                }
                false => Ok(Some((std::mem::take(launches), lowered))),
            },
            _ => Err(lowered),
        });

        match batched {
            Ok(None) => Ok(()),
            Ok(Some((before, lowered))) => {
                run(index, &before, &[])?;
                run(index, std::slice::from_ref(&lowered), host)
            }
            Err(lowered) => run(index, std::slice::from_ref(&lowered), host),
        }
    }
}
