//! `TensorOptions`, the optional dtype and device a tensor is created with.
use super::dtype::DType;
use crate::device::Device;

/// Dtype used when none is given.
pub const DEFAULT_DTYPE: DType = DType::F32;

/// Optional dtype and device for creating a tensor; unset fields fall back to defaults.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TensorOptions {
    dtype: Option<DType>,
    device: Option<Device>,
}

impl TensorOptions {
    /// Options with neither dtype nor device set.
    pub const fn new() -> Self {
        TensorOptions {
            dtype: None,
            device: None,
        }
    }

    /// These options with `dtype` set.
    pub const fn dtype(self, dtype: DType) -> Self {
        TensorOptions {
            dtype: Some(dtype),
            ..self
        }
    }

    /// These options with `device` set.
    pub const fn device(self, device: Device) -> Self {
        TensorOptions {
            device: Some(device),
            ..self
        }
    }

    /// The dtype, if set.
    pub const fn dtype_opt(&self) -> Option<DType> {
        self.dtype
    }

    /// The device, if set.
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
