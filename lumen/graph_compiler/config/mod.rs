use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{PoisonError, RwLock};

use crate::Device;

mod cpu;
mod cuda;
mod mps;
mod tpu;

pub use cpu::CpuConfig;
pub use cuda::CudaConfig;
pub use mps::MpsConfig;
pub use tpu::TpuConfig;

#[cfg(feature = "python")]
pub(crate) use cpu::PyCpuConfig;
#[cfg(feature = "python")]
pub(crate) use cuda::PyCudaConfig;
#[cfg(feature = "python")]
pub(crate) use mps::PyMpsConfig;
#[cfg(feature = "python")]
pub(crate) use tpu::PyTpuConfig;

pub trait CompilerConfig: std::fmt::Debug + Send + Sync {
    fn merge_dots(&self) -> bool;
    fn flash_attention(&self) -> bool;
    fn fused_linear_cross_entropy_chunk_size(&self) -> Option<usize>;
    fn memory_limit(&self) -> Option<usize>;
    fn dump_to(&self) -> Option<&str>;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Configs {
    pub cpu: CpuConfig,
    pub cuda: CudaConfig,
    pub mps: MpsConfig,
    pub tpu: TpuConfig,
}

impl Configs {
    pub fn of(&self, device: Device) -> &dyn CompilerConfig {
        match device {
            Device::Cpu(_) | Device::Meta(_) => &self.cpu,
            Device::Cuda(_) => &self.cuda,
            Device::Mps(_) => &self.mps,
            Device::Tpu(_) => &self.tpu,
        }
    }
}

macro_rules! device_config {
    ($name:ident, $py:ident, $label:literal, $field:ident, $device:expr, {
        $($flag:ident, $set:ident: $ty:ty = $default:expr;)*
    }) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name {
            pub merge_dots: bool,
            pub flash_attention: bool,
            pub fused_linear_cross_entropy_chunk_size: Option<usize>,
            pub memory_limit: usize,
            pub dump_to: Option<String>,
            $(pub $flag: $ty,)*
        }

        impl Default for $name {
            fn default() -> Self {
                $name {
                    merge_dots: true,
                    flash_attention: true,
                    fused_linear_cross_entropy_chunk_size: None,
                    memory_limit: 0,
                    dump_to: None,
                    $($flag: $default,)*
                }
            }
        }

        #[cfg(feature = "python")]
        #[pyo3::pyclass(name = $label, module = "lumen", extends = super::PyCompilerConfig)]
        pub(crate) struct $py;

        #[cfg(feature = "python")]
        impl $py {
            pub(crate) fn object(py: pyo3::Python<'_>) -> pyo3::PyResult<pyo3::Py<$py>> {
                let base = super::PyCompilerConfig { device: $device };
                pyo3::Py::new(py, pyo3::PyClassInitializer::from(base).add_subclass($py))
            }
        }

        #[cfg(feature = "python")]
        #[pyo3::pymethods]
        impl $py {
            $(
                #[getter]
                fn $flag(&self) -> $ty {
                    super::configs().$field.$flag
                }

                #[setter]
                fn $set(&self, value: $ty) {
                    super::update(|c| c.$field.$flag = value);
                }
            )*

            pub(crate) fn __repr__(&self) -> String {
                let c = super::configs().$field;
                let flags: Vec<String> = vec![
                    format!("merge_dots={}", super::py_value(&c.merge_dots)),
                    format!("flash_attention={}", super::py_value(&c.flash_attention)),
                    format!(
                        "fused_linear_cross_entropy_chunk_size={}",
                        super::py_value(&c.fused_linear_cross_entropy_chunk_size)
                    ),
                    format!("memory_limit={}", super::py_value(&c.memory_limit)),
                    format!("dump_to={}", super::py_value(&c.dump_to)),
                    $(format!("{}={}", stringify!($flag), super::py_value(&c.$flag))),*
                ];
                format!(
                    "{}.config.compiler.{}({})",
                    crate::LIBRARY_NAME,
                    stringify!($field),
                    flags.join(", ")
                )
            }
        }
    };
}

pub(crate) use device_config;

#[cfg(feature = "python")]
#[pyo3::pyclass(name = "CompilerConfig", module = "lumen", subclass)]
pub(crate) struct PyCompilerConfig {
    device: Device,
}

#[cfg(feature = "python")]
macro_rules! shared_flags {
    ($($flag:ident, $set:ident: $ty:ty;)*) => {
        #[pyo3::pymethods]
        impl PyCompilerConfig {
            $(
                #[getter]
                fn $flag(&self) -> $ty {
                    let c = configs();
                    match self.device {
                        Device::Cpu(_) | Device::Meta(_) => c.cpu.$flag,
                        Device::Cuda(_) => c.cuda.$flag,
                        Device::Mps(_) => c.mps.$flag,
                        Device::Tpu(_) => c.tpu.$flag,
                    }
                }

                #[setter]
                fn $set(&self, value: $ty) {
                    update(|c| match self.device {
                        Device::Cpu(_) | Device::Meta(_) => c.cpu.$flag = value,
                        Device::Cuda(_) => c.cuda.$flag = value,
                        Device::Mps(_) => c.mps.$flag = value,
                        Device::Tpu(_) => c.tpu.$flag = value,
                    });
                }
            )*

            fn reset(&self) {
                update(|c| match self.device {
                    Device::Cpu(_) | Device::Meta(_) => c.cpu = CpuConfig::default(),
                    Device::Cuda(_) => c.cuda = CudaConfig::default(),
                    Device::Mps(_) => c.mps = MpsConfig::default(),
                    Device::Tpu(_) => c.tpu = TpuConfig::default(),
                });
            }
        }
    };
}

#[cfg(feature = "python")]
shared_flags! {
    merge_dots, set_merge_dots: bool;
    flash_attention, set_flash_attention: bool;
    fused_linear_cross_entropy_chunk_size, set_fused_linear_cross_entropy_chunk_size: Option<usize>;
    memory_limit, set_memory_limit: usize;
    dump_to, set_dump_to: Option<String>;
}

#[cfg(feature = "python")]
fn py_value(value: &dyn std::fmt::Debug) -> String {
    match format!("{value:?}").as_str() {
        "true" => "True".into(),
        "false" => "False".into(),
        v => v
            .strip_prefix("Some(")
            .and_then(|v| v.strip_suffix(')'))
            .unwrap_or(v)
            .into(),
    }
}

static CONFIGS: RwLock<Option<Configs>> = RwLock::new(None);

static VERSION: AtomicU64 = AtomicU64::new(0);

pub fn version() -> u64 {
    VERSION.load(Ordering::Acquire)
}

pub fn configs() -> Configs {
    let configs = CONFIGS.read().unwrap_or_else(PoisonError::into_inner);
    configs.clone().unwrap_or_default()
}

pub fn update(change: impl FnOnce(&mut Configs)) {
    let mut configs = CONFIGS.write().unwrap_or_else(PoisonError::into_inner);
    let mut next = configs.clone().unwrap_or_default();
    change(&mut next);
    *configs = Some(next);
    VERSION.fetch_add(1, Ordering::Release);
}
