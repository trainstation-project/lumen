//! TPU's answers to the dot passes.

use crate::codegen::devices::tpu::Tpu;

/// TPU's matmul kernel reads its batch dims leading, the lhs as batch, free dims then one contracting one (or batch,
/// contracting, free), the rhs as batch, then one contracting and one free dim in either order.
impl super::canonicalize::DotLayouts for Tpu {
    /// Whether the TPU matmul kernel reads operand `k` of `dot` over `operands` as laid out.
    fn readable(
        dot: &crate::graph::Primitive,
        operands: [&crate::graph::TensorType; 2],
        k: usize,
    ) -> bool {
        let crate::graph::Primitive::DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            rhs_batch,
            ..
        } = dot
        else {
            unreachable!("a dot_general")
        };

        let leads = |batch: &[usize]| batch.iter().enumerate().all(|(i, &d)| d == i);
        let nb = lhs_batch.len();
        let rank = operands[k].shape.len();

        match k {
            0 => {
                let last = *lhs_contracting == [rank - 1];
                let transposed = rank == nb + 2 && *lhs_contracting == [nb];
                leads(lhs_batch) && rank >= nb + 2 && (last || transposed)
            }
            _ => leads(rhs_batch) && rank == nb + 2 && rhs_contracting.len() == 1,
        }
    }
}

/// TPU fuses matmuls with their slices, stacked lhs and epilogues, in whole vector lanes, a producer writing a
/// concatenation's slot as `place_concats::tpu::slot_writes` accepts.
impl super::fuse::FusesDots for Tpu {
    const LANES: usize = crate::codegen::devices::tpu::LANES;

    /// `place_concats::tpu::slot_writes`.
    fn slot_writes(
        graph: &crate::graph::Graph,
        concat: &crate::graph::Node,
        k: usize,
        producer: &crate::graph::Node,
    ) -> bool {
        crate::graph_compiler::passes::place_concats::tpu::slot_writes(graph, concat, k, producer)
    }
}

/// TPU's row kernel reads the inputs it classes `Kind::Full` as whole rows.
impl super::rows::FullRows for Tpu {
    /// Whether `codegen::row::tpu::kinds` classes input `at` of `body` as full rows.
    fn full(body: &crate::graph::Graph, at: usize) -> bool {
        use crate::codegen::row::tpu::{Kind, kinds};
        kinds(body).is_some_and(|kinds| kinds[body.inputs()[at]] == Kind::Full)
    }
}
