//! The type of a graph value and the index naming it.

use std::fmt;

use crate::DType;

/// The dtype and shape of a graph value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl TensorType {
    /// Builds a type from a dtype and a shape slice.
    pub fn new(dtype: DType, shape: &[usize]) -> Self {
        TensorType {
            dtype,
            shape: shape.to_vec(),
        }
    }

    /// Number of elements, the product of the shape.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims: Vec<String> = self.shape.iter().map(usize::to_string).collect();
        write!(f, "{}[{}]", self.dtype, dims.join(","))
    }
}

/// Index of a value in a graph's type table.
pub type Var = usize;
