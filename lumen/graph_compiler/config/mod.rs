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
    ($name:ident, $py:ident, $label:literal, $field:ident {
        $($flag:ident, $set:ident: $ty:ty = $default:expr;)*
    }) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name {
            $(pub $flag: $ty,)*
        }

        impl Default for $name {
            fn default() -> Self {
                $name {
                    $($flag: $default,)*
                }
            }
        }

        #[cfg(feature = "python")]
        #[pyo3::pyclass(name = $label, module = "lumen")]
        pub(crate) struct $py;

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

            fn reset(&self) {
                super::update(|c| c.$field = $name::default());
            }

            pub(crate) fn __repr__(&self) -> String {
                let c = super::configs().$field;
                let flags: Vec<String> = vec![
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
