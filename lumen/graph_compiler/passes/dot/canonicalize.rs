//! `CanonicalizeDots` (pass name `canonicalize_dots`) puts in an explicit `Transpose` before each dot operand
//! a backend's matmul kernel can't read as laid out, and rewrites the dot's dimension numbers to match. It
//! runs after `split_k` and before the diamonds passes, so later passes and codegen see only operands the
//! kernel reads directly.
//!
//! ## Rewrite
//!
//! - `CanonicalizeDots { readable }` holds the backend's rule. `Readable` is
//!   `fn(&Primitive, [&TensorType; 2], operand index) -> bool` (0 = lhs, 1 = rhs).
//! - `matmul_order(dot, lhs_rank, rhs_rank)` gives the permutations that put the lhs in
//!   batch, free (M), contracting (K) order and the rhs in batch, contracting (K), free (N) order, plus where
//!   each group ends (`lhs_split`, `rhs_split`).
//! - An unreadable lhs is transposed by `order.lhs`; its batch dims become `0..i` and its contracting dims the
//!   tail. An unreadable rhs is transposed by `order.rhs`; its batch dims become `0..i` and contracting dims
//!   `i..j`. A readable operand is left alone, so only the operands that need it are copied.
//! - Dtypes are unchanged; every other node is copied with its label.
//!
//! `matmul_order` is also used by MPS's GEMM codegen (`codegen::dot_general::mps`) to read operands in the
//! same `[batch, M, K]` / `[batch, K, N]` form. Tests construct the pass directly with `mps::reads_contiguous`.
//!
//! Devices: it runs, always, on each device implementing `DotLayouts` (which operand layouts its
//! matmul kernels read as laid out): MPS (what its GEMM reads at contiguous strides) and TPU (batch dims leading, one
//! contracting dim placed as its kernel reads it).
use crate::graph::ir::primitive::free_dims;
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::{CompilerPass, Module};

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

/// A backend telling the `canonicalize_dots` pass which matmul operands its kernels read as laid out;
/// the others are transposed into a form they can first.
pub(crate) trait DotLayouts {
    /// Whether the matmul kernels read operand `k` of `dot` (over `operands`) as laid out.
    fn readable(
        dot: &crate::graph::Primitive,
        operands: [&crate::graph::TensorType; 2],
        k: usize,
    ) -> bool;
}

/// Compiler pass running `canonicalize_dots` on the module's graph with the backend's `readable`.
pub(crate) struct CanonicalizeDots {
    pub readable: Readable,
}

impl CanonicalizeDots {
    /// The pass, the matmul layouts device `D` reads.
    pub(crate) fn of<D: DotLayouts>() -> Self {
        CanonicalizeDots {
            readable: D::readable,
        }
    }
}

impl<D: DotLayouts> CompilerPass<DeviceModule<'_, D>> for CanonicalizeDots {
    fn name(&self) -> &'static str {
        "canonicalize_dots"
    }

    /// Rebuilds `graph` with explicit transposes in front of dot operands the backend's kernels can't read as laid
    /// out (`readable`), rewriting the dimension numbers to match the transposed layout.
    fn apply(&mut self, m: &mut DeviceModule<'_, D>) -> Result<(), String> {
        let graph = m.graph();
        let readable = self.readable;

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

        *m.graph_mut() = out;
        Ok(())
    }
}
