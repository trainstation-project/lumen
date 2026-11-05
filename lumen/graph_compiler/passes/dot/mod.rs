//! Passes rewriting dots (`DotGeneral`): operand layouts, strength reduction, split-K, and fusing a matmul
//! into the row kernel reading its output. `split_k` and `rows` hold each backend's version in their
//! `mps`, `tpu` and `triton` files; `tpu` here fuses TPU matmuls with their epilogues, and `triton` runs
//! GEMM steps through the Triton GEMM kernel.

pub(crate) mod canonicalize;
pub(crate) mod fold_reshapes;
pub(crate) mod rows;
pub(crate) mod split_k;
pub(crate) mod strength;
pub(crate) mod tpu;
