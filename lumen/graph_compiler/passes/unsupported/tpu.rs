//! TPU's answer to the `unsupported` pass: which ops it has no kernel for.

use crate::codegen::devices::tpu::Tpu;
use crate::graph::Primitive;

/// What TPU kernels run: no float16 (Mosaic's TPU kernels have no float16), and no cumsum, scatter-add or random
/// bits (no TPU kernel computes them yet).
impl super::Supports for Tpu {
    /// Rejects graphs with a float16 value or an op no TPU kernel computes.
    fn check(graph: &crate::graph::Graph) -> Result<(), String> {
        use crate::DType;

        for node in graph.nodes() {
            let half = node
                .inputs
                .iter()
                .chain([&node.output])
                .any(|&v| graph.type_of(v).dtype == DType::F16);

            if half {
                return Err(format!(
                    "{}: float16 does not run on TPU yet",
                    node.primitive.name()
                ));
            }

            if matches!(
                node.primitive,
                Primitive::Cumsum { .. }
                    | Primitive::ScatterAdd { .. }
                    | Primitive::RandomBits { .. }
            ) {
                return Err(format!("{} does not run on TPU yet", node.primitive.name()));
            }
        }

        Ok(())
    }
}
