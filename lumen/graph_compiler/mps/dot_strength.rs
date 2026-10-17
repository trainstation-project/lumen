//! Dot strength reduction (XLA: `DotStrengthReduction`): a dot with one
//! operand a vector (its free dimensions of size 1, or none: a decode
//! step's `x @ W`, attention's backward `D = rowsum(dO * O)`) is a product
//! and a sum, `reduce_sum(v * m)` over the contracted dimensions, the
//! vector broadcast into the matrix's layout (read as it is, once), not a
//! matmul (whose tiles would be mostly padding). The operands are cast to
//! the accumulation dtype (exactly: a product of two floats of half its
//! width is exact in it), the sum accumulates in it, and the result is cast
//! to the dot's.

use crate::DType;
use crate::graph::{Graph, Primitive, Var};

/// `graph` with each dot of a vector a product and a sum.
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
                // The matrix: the rhs, unless it is the vector (the lhs's
                // free dimensions have more than one element).
                let vector = |&(v, b, c): &(Var, &Vec<usize>, &Vec<usize>)| {
                    let shape = &out.type_of(v).shape;
                    let free = (0..shape.len()).filter(|d| !b.contains(d) && !c.contains(d));
                    free.map(|d| shape[d]).product::<usize>() == 1
                };
                let accum = *accum_dtype;
                match (vector(&operands[0]), vector(&operands[1])) {
                    (true, _) => Some(product(&mut out, operands[0], operands[1], accum)),
                    (false, true) => Some(product(&mut out, operands[1], operands[0], accum)),
                    (false, false) => None,
                }
                .map(|sum| {
                    // The vector's free dimensions (of size 1) back.
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

/// `sum_k vector * matrix` in `accum`, each operand `(value, batch,
/// contracting)`: the vector broadcast into the matrix's dimensions, the
/// product summed over its contracting ones; its dimensions in the dot's
/// order but for the vector's free ones (of size 1).
fn product(
    out: &mut Graph,
    (v, v_batch, v_contracting): (Var, &Vec<usize>, &Vec<usize>),
    (m, m_batch, m_contracting): (Var, &Vec<usize>, &Vec<usize>),
    accum: DType,
) -> Var {
    let (v_ty, m_ty) = (out.type_of(v).clone(), out.type_of(m).clone());
    // The vector's batch and contracting dimensions, each with the
    // matrix's it is, in the matrix's order.
    let mut kept: Vec<(usize, usize)> = v_batch
        .iter()
        .zip(m_batch)
        .chain(v_contracting.iter().zip(m_contracting))
        .map(|(&a, &b)| (a, b))
        .collect();
    kept.sort_by_key(|&(_, d)| d);
    // Its free dimensions (of size 1) dropped, then in that order.
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
    // Over the matrix's free dimensions (none: as it is).
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
    // The sum's dimensions (the matrix's others, in its order) in the
    // dot's: batch, then the lhs's free ones, then the rhs's.
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

/// `v` (of `from`) as `to`.
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

/// `v` of `out` reshaped to `shape`, if it is not of it (a reshape to its
/// own shape is a copy).
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
