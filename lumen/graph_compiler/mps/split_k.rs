//! MPS split-K: the shared pass (`common::split_k`) with MPS's tiles, its GEMV kept whole, and
//! atomic accumulation unless deterministic.

use super::fusion::fusible;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::common::split_k::Rules;
use crate::graph_compiler::pass::{CompilerPass, Module};
use crate::kernels::dot_general::mps::{float_groups, gemv};

/// Splits `graph`'s matmuls by MPS's rules; `atomic` and `fuse` as in `Rules`.
pub(super) fn split_k(graph: &Graph, atomic: bool, fuse: bool) -> Graph {
    let rules = Rules {
        tiles: &|m, n| float_groups(1, m, n),
        target: 256,
        unsplit: &|dot, lhs, rhs, out| gemv(dot, lhs, rhs, out).is_some(),
        fusible: &fusible,
        atomic,
        fuse,
    };
    crate::graph_compiler::common::split_k::split_k(graph, &rules)
}

/// The dot of a `split_k` fusion body (a dot followed by a sum over its leading axis).
pub(super) fn atomic_dot(body: &Graph) -> Option<&Node> {
    match body.nodes() {
        [dot, sum] => {
            let split = matches!(dot.primitive, Primitive::DotGeneral { .. })
                && matches!(&sum.primitive, Primitive::ReduceSum { axes, .. } if axes == &[0])
                && sum.inputs == [dot.output];
            split.then_some(dot)
        }
        _ => None,
    }
}

/// Compiler pass running `split_k`, atomically unless deterministic mode is on.
pub(super) struct SplitK;

impl<M: Module> CompilerPass<M> for SplitK {
    fn name(&self) -> &'static str {
        "split_k"
    }

    /// Splits the module's matmuls.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let config = &m.options().config.mps;
        *m.graph_mut() = split_k(m.graph(), !config.deterministic, config.fuse);
        Ok(())
    }
}
