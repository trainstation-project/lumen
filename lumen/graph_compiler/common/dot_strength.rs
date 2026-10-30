//! Strength reduction of dots between vectors into elementwise multiply and reduce-sum.

use crate::DType;
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};

/// Rewrites float dots whose operands both have only size-1 free dims (batched inner products)
/// as `reduce_sum(x * y)`, cast to the output dtype when it differs from the accumulator.
pub(crate) fn reduce_vector_dots(graph: &Graph) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    for node in graph.nodes() {
        out.set_origin(node);
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

        map[node.output] = match &node.primitive {
            Primitive::DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                accum_dtype,
                output_dtype,
            } if accum_dtype.is_float() => {
                let operands = [
                    (inputs[0], lhs_batch, lhs_contracting),
                    (inputs[1], rhs_batch, rhs_contracting),
                ];

                let vector = |&(v, b, c): &(Var, &Vec<usize>, &Vec<usize>)| {
                    let shape = &out.type_of(v).shape;
                    let free = (0..shape.len()).filter(|d| !b.contains(d) && !c.contains(d));
                    free.map(|d| shape[d]).product::<usize>() == 1
                };

                let accum = *accum_dtype;

                (vector(&operands[0]) && vector(&operands[1]))
                    .then(|| product(&mut out, operands[0], operands[1], accum))
                    .map(|sum| {
                        let sum = reshaped(&mut out, sum, &graph.type_of(node.output).shape);

                        match accum == *output_dtype {
                            true => sum,
                            false => out
                                .apply(
                                    Primitive::Cast {
                                        new_dtype: *output_dtype,
                                    },
                                    &[sum],
                                )
                                .expect("a cast"),
                        }
                    })
            }
            _ => None,
        }
        .unwrap_or_else(|| {
            let v = out
                .apply(node.primitive.clone(), &inputs)
                .expect("a node of the graph");

            out.set_label(v, node.label);
            v
        });
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// `reduce_sum(v * m)` over `m`'s contracting dims, with `v` laid out and broadcast onto `m`'s shape;
/// the result is in dot output order (batch dims, then free dims).
fn product(
    out: &mut Graph,
    (v, v_batch, v_contracting): (Var, &Vec<usize>, &Vec<usize>),
    (m, m_batch, m_contracting): (Var, &Vec<usize>, &Vec<usize>),
    accum: DType,
) -> Var {
    let (v_ty, m_ty) = (out.type_of(v).clone(), out.type_of(m).clone());

    // Pairs of matching (v dim, m dim) for batch and contracting dims, in m's dim order.
    let mut kept: Vec<(usize, usize)> = v_batch
        .iter()
        .zip(m_batch)
        .chain(v_contracting.iter().zip(m_contracting))
        .map(|(&a, &b)| (a, b))
        .collect();

    kept.sort_by_key(|&(_, d)| d);
    // Drop v's size-1 free dims, then permute its kept dims into m's order.
    let mut by_vector = kept.clone();
    by_vector.sort_unstable();
    let new_sizes: Vec<usize> = by_vector.iter().map(|&(d, _)| v_ty.shape[d]).collect();
    let mut x = reshaped(out, v, &new_sizes);
    let mut apply = |p: Primitive, ins: &[Var]| out.apply(p, ins).expect("well typed");

    let permutation: Vec<usize> = kept
        .iter()
        .map(|k| by_vector.iter().position(|b| b == k).expect("kept"))
        .collect();

    if permutation.iter().enumerate().any(|(i, &d)| i != d) {
        x = apply(Primitive::Transpose { permutation }, &[x]);
    }

    let x = cast_with(&mut apply, x, v_ty.dtype, accum);

    let x = match kept.len() == m_ty.shape.len() {
        true => x,
        false => apply(
            Primitive::BroadcastInDim {
                shape: m_ty.shape.clone(),
                broadcast_dimensions: kept.iter().map(|&(_, d)| d).collect(),
            },
            &[x],
        ),
    };

    let y = cast_with(&mut apply, m, m_ty.dtype, accum);
    let product = apply(Primitive::Mul, &[x, y]);
    let mut axes = m_contracting.clone();
    axes.sort_unstable();

    let sum = apply(
        Primitive::ReduceSum {
            axes: axes.clone(),
            accum_dtype: accum,
        },
        &[product],
    );

    let rest: Vec<usize> = (0..m_ty.shape.len())
        .filter(|d| !axes.contains(d))
        .collect();

    let free = rest.iter().copied().filter(|d| !m_batch.contains(d));
    let order: Vec<usize> = m_batch.iter().copied().chain(free).collect();

    let permutation: Vec<usize> = order
        .iter()
        .map(|d| rest.iter().position(|r| r == d).expect("a dimension left"))
        .collect();

    match permutation.iter().enumerate().any(|(i, &d)| i != d) {
        true => apply(Primitive::Transpose { permutation }, &[sum]),
        false => sum,
    }
}

/// `v` cast from `from` to `to`, unchanged when they match.
fn cast_with(
    apply: &mut impl FnMut(Primitive, &[Var]) -> Var,
    v: Var,
    from: DType,
    to: DType,
) -> Var {
    match from == to {
        true => v,
        false => apply(Primitive::Cast { new_dtype: to }, &[v]),
    }
}

/// `v` reshaped to `shape`, unchanged when it already has it.
fn reshaped(out: &mut Graph, v: Var, shape: &[usize]) -> Var {
    match out.type_of(v).shape == shape {
        true => v,
        false => out
            .apply(
                Primitive::Reshape {
                    new_sizes: shape.to_vec(),
                },
                &[v],
            )
            .expect("the same elements"),
    }
}

/// Compiler pass running [`reduce_vector_dots`].
pub(crate) struct DotStrength;

impl<M: Module> CompilerPass<M> for DotStrength {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "dot_strength"
    }

    /// Replaces the module's graph with its strength-reduced version.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = reduce_vector_dots(m.graph());
        Ok(())
    }
}
