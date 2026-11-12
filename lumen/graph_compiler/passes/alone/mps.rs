//! What MPS runs for this pass.

use crate::DType;
use crate::codegen::mps::{Kernel, request};
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::alone::{Alone, Lone};
use crate::graph_compiler::pipeline::Pass;

impl Lone for DeviceModule<'_, Mps> {
    /// Whether `n` needs its own kernel: a float GEMM, a copying concatenation, a gather, a layout op that
    /// copies (a broadcast, transpose or slice not placed as a view) or an elementwise op.
    fn alone(graph: &Graph, n: &Node) -> bool {
        let float = |v: Var| {
            matches!(
                graph.type_of(v).dtype,
                DType::F16 | DType::BF16 | DType::F32
            )
        };

        let gemm = matches!(n.primitive, Primitive::DotGeneral { .. })
            && n.inputs.iter().all(|&v| float(v))
            && float(n.output);

        gemm || matches!(
            n.primitive,
            Primitive::Concatenate {
                in_place: false,
                ..
            } | Primitive::Gather { .. }
                | Primitive::BroadcastInDim { .. }
                | Primitive::Transpose { .. }
                | Primitive::Slice { view: false, .. }
        ) || crate::graph_compiler::elementwise(&n.primitive)
    }

    /// Graph input indices passed to kernels by value.
    fn scalars(&mut self) -> Option<&mut Vec<usize>> {
        Some(&mut self.scalars)
    }

    /// Requests a fusion kernel, returning its pending name; codegen reads its by-value inputs off the plan.
    fn kernel(&mut self, _: &Graph, _: &[bool]) -> String {
        request(Kernel::Fusion)
    }
}

/// Always.
impl DevicePass<Mps> for Alone {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Mps>> {
        Box::new(Alone)
    }
}
