//! Kernels written by hand against a device's kernel language: `lumen.kernels.pallas`, the Pallas-style TPU kernels
//! a program traces (its Rust half; the CuTe DSL helpers in `cuda/` are Python only).

pub mod pallas;
