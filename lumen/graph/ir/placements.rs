//! Values stored inside another value's buffer (view slices, operands of in-place concatenates) and where.

use super::{Graph, Node, Primitive, Var};
use crate::tensor::contiguous_strides;

/// Where a value lives inside another value's buffer: element offset and strides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub offset: usize,
    pub strides: Vec<usize>,
}

impl Graph {
    /// View of the `k`-th operand of concatenate `node` inside its output buffer.
    /// Panics if `node` is not a concatenate.
    pub fn slot(&self, node: &Node, k: usize) -> View {
        let Primitive::Concatenate { dimension, .. } = node.primitive else {
            panic!("{}: not a concatenate", node.primitive)
        };

        let strides = contiguous_strides(&self.types[node.output].shape);

        // The slot starts after the earlier operands' extents along the concatenated dimension.
        let before: usize = node.inputs[..k]
            .iter()
            .map(|&v| self.types[v].shape[dimension])
            .sum();

        View {
            offset: before * strides[dimension],
            strides,
        }
    }

    /// For each value stored inside another's buffer (view slices, operands of in-place concatenates),
    /// the host value and the view; `None` for values with their own storage.
    pub fn placements(&self) -> Vec<Option<(Var, View)>> {
        let mut placed = vec![None; self.types.len()];

        for node in &self.nodes {
            match &node.primitive {
                Primitive::Slice {
                    start_indices,
                    view: true,
                    ..
                } => {
                    let x = node.inputs[0];
                    let strides = contiguous_strides(&self.types[x].shape);
                    let offset = start_indices.iter().zip(&strides).map(|(s, t)| s * t).sum();
                    placed[node.output] = Some((x, View { offset, strides }));
                }
                Primitive::Concatenate { in_place: true, .. } => {
                    for (k, &v) in node.inputs.iter().enumerate() {
                        placed[v] = Some((node.output, self.slot(node, k)));
                    }
                }
                _ => {}
            }
        }

        placed
    }
}
