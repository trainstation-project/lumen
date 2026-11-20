//! Passes rewriting dots (`DotGeneral`): operand layouts, strength reduction, split-K, fusing a matmul into the row
//! kernel reading its output (`rows`, `wide_rows`), and fusing matmuls with their slices and epilogues (`fuse`). Each
//! runs on the devices whose codegen implements its trait.

pub(crate) mod canonicalize;
pub(crate) mod fold_reshapes;
pub(crate) mod fuse;
pub(crate) mod rows;
pub(crate) mod split_k;
pub(crate) mod strength;
pub(crate) mod wide_rows;
