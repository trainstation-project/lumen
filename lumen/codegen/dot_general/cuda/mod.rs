//! The CUDA (CuTe DSL) GEMM's launch configs; the kernel and its launcher are Python (`ampere.py`, `launch.py`).

mod config;
pub(crate) use config::*;
