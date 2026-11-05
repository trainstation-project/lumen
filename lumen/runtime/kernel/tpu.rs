//! Launching a TPU kernel.

use super::Launch;
use crate::codegen::utils::tpu::Tpu;
use crate::runtime::tpu::{Lowered, SparseCore, enqueue, lowering};

/// One TPU launch: the device it runs on, the kernel's lowering module (or generated program) and spec, its
/// buffers, extra scalars and host data.
pub(crate) struct TpuLaunch<'a> {
    /// The TPU it runs on.
    pub(crate) device: usize,
    /// Python module whose `program(spec)` lowers the kernel, or `GENERATED` for a generated program.
    pub(crate) module: &'static str,
    /// Name shown in profiles.
    pub(crate) kernel: String,
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
}

/// A TPU lowers each kernel to Mosaic once and batches it into the program it runs on flush.
impl Launch for Tpu {
    type Args<'a> = TpuLaunch<'a>;

    /// Lowers `launch` and adds it to the open batch on device `index`, or runs it now. A launch with host
    /// data runs on its own after flushing the batch.
    fn launch(launch: TpuLaunch<'_>) -> Result<(), String> {
        let index = launch.device;
        let scalars = launch.rows.len() + launch.scalars.len();
        let host =
            (!launch.host.is_empty()).then_some(launch.host.len() / crate::allocator::tpu::ROW);

        let (mosaic, spec) = lowering(index, launch.module, &launch.spec, scalars, host)
            .map_err(|e| format!("{}: {e}", launch.kernel))?;

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

        enqueue(index, &launch.kernel, lowered, launch.host)
    }
}
