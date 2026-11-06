//! The CUDA (CuTe DSL) GEMM's launch configs; the kernel and its launcher are Python (`gemm_sm80.py`, `launch.py`).

mod config;

pub(crate) use config::*;
