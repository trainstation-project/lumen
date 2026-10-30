//! Triton layout: none generated, layout ops are views the elementwise loops read through.

use std::collections::HashMap;

use super::LayoutCodegen;
use crate::codegen::triton::Triton;
use crate::graph::{Graph, TensorType};

/// Triton layout: transposes and reshapes are views folded into the elementwise loops.
impl LayoutCodegen for Triton {
    /// A kernel's name and source.
    type Code = (String, String);

    /// None: a transpose is a view the elementwise loops read through.
    fn transpose(
        _: &Graph,
        _: &[bool],
        _: &HashMap<usize, (usize, Vec<usize>)>,
    ) -> Result<(String, String), String> {
        Err("a transpose on Triton is a view the elementwise loops read through".into())
    }

    /// None: a reshape is a view the elementwise loops read through.
    fn reshape(from: &TensorType, to: &TensorType) -> Result<(String, String), String> {
        Err(format!(
            "a reshape of {from} to {to} on Triton is a view the elementwise loops read through"
        ))
    }
}
