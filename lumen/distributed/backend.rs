//! Per-device mesh setup: how each backend finds this process's rank and world size, and TPU slice env wiring.
use crate::Device;
use crate::device::DeviceBackend;
use crate::device::cpu::Cpu;
use crate::device::cuda::Cuda;
use crate::device::meta::Meta;
use crate::device::mps::Mps;
use crate::device::tpu::Tpu;

/// Env var holding the TPU host bounds, e.g. `2,2,1`.
pub const TPU_TOPOLOGY: &str = "LUMEN_TPU_TOPOLOGY";
/// Env var holding the comma-separated `host:port` slicebuilder address of every worker.
pub const TPU_SLICEBUILDER_ADDRESSES: &str = "LUMEN_TPU_SLICEBUILDER_ADDRESSES";

/// Joins the env-configured process group when WORLD_SIZE > 1 and returns `(rank, world_size)`.
fn default_world() -> Result<(usize, usize), String> {
    if !super::is_initialized() && super::get_world_size()? > 1 {
        let (store, rank, world_size) =
            super::rendezvous("env://", None, None, super::DEFAULT_TIMEOUT)?;

        super::init_process_group(store, rank, world_size)?;
    }

    Ok((super::get_rank()?, super::get_world_size()?))
}

/// Device-specific hook that resolves this process's place in a mesh.
pub trait MeshBackend: DeviceBackend {
    /// Returns `(rank, world_size)` for a mesh of `_mesh_size` devices, initialising the process group if needed.
    fn world(&self, _mesh_size: usize) -> Result<(usize, usize), String> {
        default_world()
    }
}

impl MeshBackend for Cpu {}
impl MeshBackend for Cuda {}
impl MeshBackend for Meta {}

/// Builds the libtpu env vars that make worker `rank` drive local chip `local_rank` of a multi-process slice.
pub fn tpu_slice_env(
    topology: &str,
    addresses: &str,
    rank: usize,
    local_rank: usize,
    world_size: usize,
    libtpu_init_args: &str,
) -> Result<Vec<(&'static str, String)>, String> {
    let addresses: Vec<&str> = addresses.split(',').filter(|a| !a.is_empty()).collect();

    if addresses.is_empty() {
        return Err(format!("{TPU_SLICEBUILDER_ADDRESSES} is empty"));
    }

    if world_size > addresses.len() {
        return Err(format!(
            "a world of {world_size} processes spans several TPU slices, but {TPU_SLICEBUILDER_ADDRESSES} lists one of {} processes",
            addresses.len()
        ));
    }

    // workers past one slice's address list wrap around; the size check above means this is only the identity today
    let task = rank % addresses.len();

    let port = addresses[task]
        .rsplit_once(':')
        .map(|(_, port)| port.to_owned())
        .ok_or_else(|| {
            format!(
                "{TPU_SLICEBUILDER_ADDRESSES}: '{}' has no port",
                addresses[task]
            )
        })?;

    // each process owns one chip, so per-process chip bounds are all ones in the topology's rank
    let chip = match topology.split(',').count() {
        4 => "1,1,1,1",
        _ => "1,1,1",
    };

    let insecure = "--slicebuilder_use_insecure_grpc";

    let init_args = match libtpu_init_args.contains(insecure) {
        true => libtpu_init_args.to_owned(),
        false => format!("{libtpu_init_args} {insecure}=true")
            .trim()
            .to_owned(),
    };

    Ok(vec![
        ("CLOUD_TPU_TASK_ID", task.to_string()),
        ("TPU_VISIBLE_DEVICES", local_rank.to_string()),
        ("TPU_VISIBLE_CHIPS", local_rank.to_string()),
        ("TPU_HOST_BOUNDS", topology.to_owned()),
        ("TPU_CHIPS_PER_HOST_BOUNDS", chip.to_owned()),
        ("TPU_PROCESS_BOUNDS", topology.to_owned()),
        ("TPU_CHIPS_PER_PROCESS_BOUNDS", chip.to_owned()),
        ("TPU_PROCESS_PORT", port),
        ("TPU_PROCESS_ADDRESSES", addresses.join(",")),
        ("ALLOW_MULTIPLE_LIBTPU_LOAD", "1".to_owned()),
        ("LIBTPU_INIT_ARGS", init_args),
    ])
}

/// Exports the slice env for this process before libtpu opens, so it claims only its own chip.
fn join_tpu_slice() -> Result<(), String> {
    if Tpu.opened() {
        return Err(
            "the TPU was opened before its mesh: build the mesh before anything uses the TPU, so this process joins the slice with its one chip".into(),
        );
    }

    let required = |name: &str| {
        std::env::var(name).map_err(|_| {
            format!("{name} is not set: run `eval $(python -m lumen.distributed | sed 's/^/export /')` or lumen.distributed.set_tpu_launch_env() before launching the workers")
        })
    };

    let rank = super::get_rank()?;
    let local_rank = super::env_index("LOCAL_RANK", rank)?;

    let env = tpu_slice_env(
        &required(TPU_TOPOLOGY)?,
        &required(TPU_SLICEBUILDER_ADDRESSES)?,
        rank,
        local_rank,
        super::get_world_size()?,
        &std::env::var("LIBTPU_INIT_ARGS").unwrap_or_default(),
    )?;

    for (name, value) in env {
        unsafe { std::env::set_var(name, value) };
    }

    Ok(())
}

impl MeshBackend for Tpu {
    /// Joins the multi-process TPU slice before the process group when the world has several processes.
    fn world(&self, _mesh_size: usize) -> Result<(usize, usize), String> {
        if !super::is_initialized() && super::get_world_size()? > 1 {
            join_tpu_slice()?;
        }

        default_world()
    }
}

impl MeshBackend for Mps {
    /// Accepts only a mesh of the single MPS device in a single-process world.
    fn world(&self, mesh_size: usize) -> Result<(usize, usize), String> {
        self.check(0)?;
        let devices = self.device_count();

        if mesh_size != devices {
            return Err(format!(
                "an MPS mesh holds the {devices} MPS device, got {mesh_size} devices"
            ));
        }

        let world_size = super::get_world_size()?;

        if world_size != 1 {
            return Err(format!(
                "MPS runs in one process, but WORLD_SIZE is {world_size}"
            ));
        }

        Ok((0, 1))
    }
}

/// Returns the mesh backend for a device type name such as `"tpu"`.
pub fn mesh_backend(device_type: &str) -> Result<&'static dyn MeshBackend, String> {
    let device = Device::of_type(device_type, 0).ok_or_else(|| {
        format!("device_type must be one of cpu, mps, cuda, tpu, meta, got '{device_type}'")
    })?;

    Ok(match device {
        Device::Cpu(_) => &Cpu,
        Device::Mps(_) => &Mps,
        Device::Cuda(_) => &Cuda,
        Device::Tpu(_) => &Tpu,
        Device::Meta(_) => &Meta,
    })
}
