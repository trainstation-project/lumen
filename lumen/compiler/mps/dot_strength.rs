//! Dot strength reduction (XLA: `DotStrengthReduction`): a dot with no free
//! dimension on either side, a dot product per batch index (attention's
//! backward `D = rowsum(dO * O)`), is a product and a sum,
//! `reduce_sum(a * b)` over the contracted dimensions, not a matmul (whose
//! tiles would be all padding). The operands are cast to the accumulation
//! dtype (exactly: a product of two floats of half its width is exact in
//! it), the sum accumulates in it, and the result is cast to the dot's.

use crate::graph::{Graph, Primitive, Var};

/// `graph` with each dot of no free dimensions a product and a sum.
pub(crate) fn reduce_vector_dots(graph: &Graph) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in graph.nodes() {
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let Primitive::DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            rhs_batch,
            accum_dtype,
            output_dtype,
        } = &node.primitive
        else {
            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("a node of the graph");
            continue;
        };
        let (lhs, rhs) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));
        let no_free = lhs.shape.len() == lhs_batch.len() + lhs_contracting.len()
            && rhs.shape.len() == rhs_batch.len() + rhs_contracting.len();
        if !no_free || !accum_dtype.is_float() {
            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("a node of the graph");
            continue;
        }
        // Each operand as [batch..., contracting...], in the accumulation dtype.
        let a = aligned(
            &mut out,
            inputs[0],
            lhs_batch,
            lhs_contracting,
            *accum_dtype,
        );
        let b = aligned(
            &mut out,
            inputs[1],
            rhs_batch,
            rhs_contracting,
            *accum_dtype,
        );
        let mut apply = |p: Primitive, ins: &[Var]| out.apply(p, ins).expect("well typed");
        let product = apply(Primitive::Mul, &[a, b]);
        let axes: Vec<usize> = (lhs_batch.len()..lhs.shape.len()).collect();
        let mut sum = apply(
            Primitive::ReduceSum {
                axes,
                accum_dtype: *accum_dtype,
            },
            &[product],
        );
        if output_dtype != accum_dtype {
            sum = apply(
                Primitive::Cast {
                    new_dtype: *output_dtype,
                },
                &[sum],
            );
        }
        map[node.output] = sum;
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Operand `v` of `out` with its `batch` dimensions first, then its
/// `contracting` ones, in `dtype`.
fn aligned(
    out: &mut Graph,
    v: Var,
    batch: &[usize],
    contracting: &[usize],
    dtype: crate::DType,
) -> Var {
    let permutation: Vec<usize> = batch.iter().chain(contracting).copied().collect();
    let mut v = v;
    if permutation.iter().enumerate().any(|(i, &d)| i != d) {
        v = out
            .apply(Primitive::Transpose { permutation }, &[v])
            .expect("a permutation");
    }
    if out.type_of(v).dtype != dtype {
        v = out
            .apply(Primitive::Cast { new_dtype: dtype }, &[v])
            .expect("a cast");
    }
    v
}
