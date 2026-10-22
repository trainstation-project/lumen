use crate::Device;
use crate::device::DeviceBackend;
use crate::device::cpu::Cpu;
use crate::device::cuda::Cuda;
use crate::device::meta::Meta;
use crate::device::mps::Mps;
use crate::device::tpu::Tpu;

pub trait MeshBackend: DeviceBackend {
    fn world(&self, _mesh_size: usize) -> Result<(usize, usize), String> {
        if !super::is_initialized() && super::get_world_size()? > 1 {
            let (store, rank, world_size) =
                super::rendezvous("env://", None, None, super::DEFAULT_TIMEOUT)?;
            super::init_process_group(store, rank, world_size)?;
        }
        Ok((super::get_rank()?, super::get_world_size()?))
    }
}

impl MeshBackend for Cpu {}
impl MeshBackend for Cuda {}
impl MeshBackend for Tpu {}
impl MeshBackend for Meta {}

impl MeshBackend for Mps {
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
