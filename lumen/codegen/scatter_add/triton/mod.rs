//! Triton scatter-adds: the fixed kernels in `kernel.py` (atomic by default; deterministic, a counting sort
//! of the picks by position, then a row's updates added in pick order) and their launch configs. Nothing
//! fuses into a scatter-add, so there is no code to generate.

mod config;

pub(crate) use config::*;
