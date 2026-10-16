use super::dtype::DType;
use crate::device::Device;

pub const DEFAULT_DTYPE: DType = DType::F32;

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

    pub const fn dtype(self, dtype: DType) -> Self {
        TensorOptions {
            dtype: Some(dtype),
            ..self
        }
    }

    pub const fn device(self, device: Device) -> Self {
        TensorOptions {
            device: Some(device),
            ..self
        }
    }

    pub const fn dtype_opt(&self) -> Option<DType> {
        self.dtype
    }

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
