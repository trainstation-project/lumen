//! Pass that transposes `DotGeneral` operands a backend's matmul kernels can't read as laid out into
//! batch, free, contracting order (the lhs) and batch, contracting, free order (the rhs).

use crate::graph::primitive::free_dims;
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};

/// Whether a backend's matmul kernels read operand `k` (0 the lhs, 1 the rhs) of `dot` over `operands`
/// as laid out, contiguously.
pub(crate) type Readable = fn(&Primitive, [&TensorType; 2], usize) -> bool;

/// Operand dimension orders putting each operand in `[batch, M, K]` / `[batch, K, N]` form.
pub(crate) struct MatmulOrder {
    /// Lhs dims ordered batch, free (M), contracting (K).
    pub lhs: Vec<usize>,
    /// Rhs dims ordered batch, contracting (K), free (N).
    pub rhs: Vec<usize>,
    /// Where the batch and M groups end in `lhs`.
    pub lhs_split: (usize, usize),
    /// Where the batch and K groups end in `rhs`.
    pub rhs_split: (usize, usize),
}

/// The batch/free/contracting dimension orders of a `dot_general`'s operands.
pub(crate) fn matmul_order(p: &Primitive, lhs_rank: usize, rhs_rank: usize) -> MatmulOrder {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
        ..
    } = p
    else {
        unreachable!("a dot_general")
    };

    let lhs_used = [lhs_batch.as_slice(), lhs_contracting].concat();
    let rhs_used = [rhs_batch.as_slice(), rhs_contracting].concat();
    let lhs_free: Vec<usize> = free_dims(lhs_rank, &lhs_used).collect();
    let rhs_free: Vec<usize> = free_dims(rhs_rank, &rhs_used).collect();
    let (nb, nk) = (lhs_batch.len(), lhs_contracting.len());

    MatmulOrder {
        lhs: [lhs_batch.as_slice(), &lhs_free, lhs_contracting].concat(),
        rhs: [rhs_batch.as_slice(), rhs_contracting, &rhs_free].concat(),
        lhs_split: (nb, nb + lhs_free.len()),
        rhs_split: (nb, nb + nk),
    }
}

/// Rebuilds `graph` with explicit transposes in front of dot operands the backend's kernels can't read as laid
/// out (`readable`), rewriting the dimension numbers to match the transposed layout.
pub(crate) fn canonicalize_dots(graph: &Graph, readable: Readable) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    for node in graph.nodes() {
        out.set_origin(node);
        let mut inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

        let primitive = match &node.primitive {
            p @ Primitive::DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                accum_dtype,
                output_dtype,
            } => {
                let (lhs, rhs) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));
                let order = matmul_order(p, lhs.shape.len(), rhs.shape.len());
                let (mut lc, mut rc) = (lhs_contracting.clone(), rhs_contracting.clone());
                let (mut lb, mut rb) = (lhs_batch.clone(), rhs_batch.clone());

                let mut transposed = |input: &mut Var, permutation: &[usize]| {
                    let t = Primitive::Transpose {
                        permutation: permutation.to_vec(),
                    };

                    *input = out
                        .apply(t, &[*input])
                        .expect("a permutation of its dimensions");
                };

                // A transposed lhs is laid out as batch, free, contracting dims in `order.lhs`.
                if !readable(p, [lhs, rhs], 0) {
                    transposed(&mut inputs[0], &order.lhs);
                    let (i, j) = order.lhs_split;
                    (lb, lc) = ((0..i).collect(), (j..order.lhs.len()).collect());
                }

                // A transposed rhs is laid out as batch, contracting, free dims in `order.rhs`.
                if !readable(p, [lhs, rhs], 1) {
                    transposed(&mut inputs[1], &order.rhs);
                    let (i, j) = order.rhs_split;
                    (rb, rc) = ((0..i).collect(), (i..j).collect());
                }

                Primitive::DotGeneral {
                    lhs_contracting: lc,
                    rhs_contracting: rc,
                    lhs_batch: lb,
                    rhs_batch: rb,
                    accum_dtype: *accum_dtype,
                    output_dtype: *output_dtype,
                }
            }
            p => p.clone(),
        };

        map[node.output] = out
            .apply(primitive, &inputs)
            .expect("a rewrite keeps the node's type");

        out.set_label(map[node.output], node.label);
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Compiler pass running `canonicalize_dots` on the module's graph with the backend's `readable`.
pub(crate) struct CanonicalizeDots {
    pub readable: Readable,
}

impl<M: Module> CompilerPass<M> for CanonicalizeDots {
    fn name(&self) -> &'static str {
        "canonicalize_dots"
    }

    /// Replaces the module's graph with its canonicalized form.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = canonicalize_dots(m.graph(), self.readable);
        Ok(())
    }
}
