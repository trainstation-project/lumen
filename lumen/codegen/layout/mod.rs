//! Layout code generation and kernels: copies between strided and contiguous forms, reshapes of tiled
//! tensors, and transposes fused with elementwise ops.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
pub(crate) mod triton;

use crate::codegen::kinds::{gemm_dot, reduction_root, row_reductions};
use crate::graph::{Graph, Node, Primitive, TensorType, Var};
use crate::tensor::contiguous_strides;

/// A backend's layout code generator: the code moving elements between layouts. Backends whose layout
/// ops run as views or fixed kernels instead fail on them.
pub(crate) trait LayoutCodegen {
    /// The generated code: a Metal kernel's name and source, a Triton `BODY` function's, or a TPU program's
    /// kernel and spec.
    type Code;
    /// What a transpose's code is generated for beyond the body: the output views (offset, strides) MPS
    /// writes outputs through, the tensors' layouts Triton's loads and stores are generated for, or `()`.
    type Options: ?Sized;

    /// Code for the fusion `body` whose output is computed elementwise from transposes, staged through
    /// tiles; the inputs flagged in `by_value` are passed as scalars.
    fn transpose(
        body: &Graph,
        by_value: &[bool],
        options: &Self::Options,
    ) -> Result<Self::Code, String>;

    /// Code for a reshape of `from` into `to`.
    #[cfg_attr(not(lumen_tpu_linked), allow(dead_code))]
    fn reshape(from: &TensorType, to: &TensorType) -> Result<Self::Code, String>;
}

/// How a body's output is tiled for a transpose kernel.
pub(crate) struct TransposeTiling {
    /// Transpose nodes read into threadgroup tiles, all sharing the same tiling.
    pub(crate) heroes: Vec<Var>,
    /// Output elements along the merged dims that are contiguous in the transposed input.
    pub(crate) a: usize,
    /// Output elements along the merged innermost dims, contiguous in the output.
    pub(crate) b: usize,
    /// Output stride of the `a` dims.
    pub(crate) a_stride: usize,
    /// `(size, output stride)` of each remaining dim, flattened into the grid's z axis.
    pub(crate) batch: Vec<(usize, usize)>,
    /// The output dims `first..=last` merged into `a`, then those merged into `b`.
    pub(crate) runs: [(usize, usize); 2],
    /// The output dims of `batch`.
    pub(crate) batch_dims: Vec<usize>,
}

impl TransposeTiling {
    /// Tiles of `TILE` x `TILE` output elements across every batch: one program (threadgroup) each.
    pub(crate) fn tiles(&self) -> usize {
        let batches: usize = self.batch.iter().map(|d| d.0).product();
        self.a.div_ceil(TILE) * self.b.div_ceil(TILE) * batches
    }
}

/// Side of the square tile one program (a threadgroup on Metal) transposes.
pub(crate) const TILE: usize = 32;

/// Smallest extent on both tiled sides worth staging through a tile.
const MIN_TILED: usize = 16;

/// Most transposes one kernel stages, each needing its own threadgroup tile.
pub(crate) const MAX_HEROES: usize = 4;

/// Tiling for a body whose output is computed elementwise from transposes that swap a contiguous
/// input axis with the output's innermost axis, or `None` if no transpose qualifies.
pub(crate) fn transpose_tiling(body: &Graph) -> Option<TransposeTiling> {
    use Primitive::*;

    if gemm_dot(body).is_some()
        || reduction_root(body).is_some()
        || !row_reductions(body).is_empty()
    {
        return None;
    }

    let out = body.type_of(body.outputs()[0]);

    if out.numel() > u32::MAX as usize {
        return None;
    }

    let nodes = body.nodes();
    let mut used = vec![false; body.types.len()];
    let mut elementwise_only = vec![true; body.types.len()];

    for &v in body.outputs() {
        used[v] = true;
    }

    // A node is at the output when it has the output's shape and reaches it only through
    // elementwise ops, so the tile can be read back at output positions.
    let mut at_output = vec![false; body.types.len()];

    for node in nodes.iter().rev() {
        let v = node.output;
        at_output[v] = used[v] && elementwise_only[v] && body.type_of(v).shape == out.shape;

        let elementwise = matches!(
            node.primitive,
            Add | Sub
                | Mul
                | Div
                | Max
                | Eq
                | Lt
                | Neg
                | Exp
                | Log
                | Sqrt
                | Tanh
                | Logistic
                | Cast { .. }
                | Select
        );

        for &x in &node.inputs {
            used[x] = true;
            elementwise_only[x] &= elementwise && at_output[v];
        }
    }

    let tile = |node: &Node| -> Option<[(usize, usize); 2]> {
        let Transpose { permutation } = &node.primitive else {
            return None;
        };

        if !at_output[node.output] {
            return None;
        }

        let xs = contiguous_strides(&body.type_of(node.inputs[0]).shape);
        // Merge adjacent output dims that stay contiguous in the input into runs of
        // (first dim, last dim, input stride).
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();

        for d in (0..out.shape.len()).filter(|&d| out.shape[d] != 1) {
            let stride = xs[permutation[d]];

            match runs.last_mut() {
                Some(run) if run.2 == stride * out.shape[d] => *run = (run.0, d, stride),
                _ => runs.push((d, d, stride)),
            }
        }

        let a = runs.iter().position(|r| r.2 == 1)?;
        let b = runs.len() - 1;
        let size = |r: (usize, usize, usize)| out.shape[r.0..=r.1].iter().product::<usize>();

        (a < b && size(runs[a]).min(size(runs[b])) >= MIN_TILED)
            .then(|| [(runs[a].0, runs[a].1), (runs[b].0, runs[b].1)])
    };

    let mut heroes = Vec::new();
    let mut tiled = None;

    for node in nodes {
        let Some(t) = tile(node) else { continue };

        if *tiled.get_or_insert(t) == t && heroes.len() < MAX_HEROES {
            heroes.push(node.output);
        }
    }

    let [(a0, a1), (b0, b1)] = tiled?;
    let out_strides = contiguous_strides(&out.shape);
    let size = |first: usize, last: usize| out.shape[first..=last].iter().product();

    let batch_dims: Vec<usize> = (0..out.shape.len())
        .filter(|&d| out.shape[d] != 1 && !(a0..=a1).contains(&d) && !(b0..=b1).contains(&d))
        .collect();

    let batch = batch_dims
        .iter()
        .map(|&d| (out.shape[d], out_strides[d]))
        .collect();

    Some(TransposeTiling {
        heroes,
        a: size(a0, a1),
        b: size(b0, b1),
        a_stride: out_strides[a1],
        batch,
        runs: [(a0, a1), (b0, b1)],
        batch_dims,
    })
}
