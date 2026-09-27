//! Options for tensor factories (PyTorch: `c10::TensorOptions`).
//!
//! As in C++, a bare [`DType`] or [`Device`] converts into options, so
//! factories read like PyTorch's:
//!
//! ```
//! use lumen::{DType, Device, Tensor, TensorOptions};
//!
//! let a = Tensor::zeros(&[2, 3], DType::F64); // at::zeros({2, 3}, at::kDouble)
//! let b = Tensor::zeros(&[2, 3], Device::Cpu); // at::zeros({2, 3}, at::kCPU)
//! let c = Tensor::zeros(&[2, 3], TensorOptions::new().dtype(DType::I32).device(Device::Cpu));
//! let d = Tensor::zeros(&[2, 3], TensorOptions::new()); // at::zeros({2, 3})
//! assert_eq!((a.dtype(), b.dtype(), c.dtype(), d.dtype()), (DType::F64, DType::F32, DType::I32, DType::F32));
//! ```

use crate::device::Device;
use crate::dtype::DType;

/// The dtype factories use when neither the options nor a value decide it
/// (PyTorch: `torch.get_default_dtype()`).
pub const DEFAULT_DTYPE: DType = DType::F32;

/// Dtype and device for a new tensor; unset fields take the factory's
/// default (PyTorch: `TensorOptions`, whose fields are likewise optional).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TensorOptions {
    dtype: Option<DType>,
    device: Option<Device>,
}

impl TensorOptions {
    pub const fn new() -> Self {
        TensorOptions {
            dtype: None,
            device: None,
        }
    }

    /// These options with `dtype` set (PyTorch: `TensorOptions::dtype`).
    pub const fn dtype(self, dtype: DType) -> Self {
        TensorOptions {
            dtype: Some(dtype),
            ..self
        }
    }

    /// These options with `device` set (PyTorch: `TensorOptions::device`).
    pub const fn device(self, device: Device) -> Self {
        TensorOptions {
            device: Some(device),
            ..self
        }
    }

    /// The dtype, if set (PyTorch: `TensorOptions::dtype_opt`).
    pub const fn dtype_opt(&self) -> Option<DType> {
        self.dtype
    }

    /// The device, if set (PyTorch: `TensorOptions::device_opt`).
    pub const fn device_opt(&self) -> Option<Device> {
        self.device
    }
}

impl From<DType> for TensorOptions {
    fn from(dtype: DType) -> Self {
        TensorOptions::new().dtype(dtype)
    }
}

impl From<Device> for TensorOptions {
    fn from(device: Device) -> Self {
        TensorOptions::new().device(device)
    }
}
