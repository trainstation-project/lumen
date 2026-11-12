//! Flash attention: pattern matching of attention (forward and backward) in traced graphs and the rewrites
//! replacing each match with a fusion kernel. The matcher and operand layouts both directions share live
//! here; each direction's matching, rewrite and pass live in `forward` and `backward`, with each backend's
//! answers to that pass in their `mps`, `tpu` and `triton` files. `tpu` and `triton` here hold what those
//! backends do with the fusions after matching: TPU's layout folding and Triton's kernel launches.

#[cfg(lumen_mps_linked)]
mod mps;
mod triton;

use crate::DType;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::identity::Identity;
use crate::graph_compiler::pipeline::Pass;
use crate::graph_compiler::{CompilerConfig, constant};
use crate::tensor::contiguous_strides;
use std::collections::HashMap;

mod backward;
mod forward;
pub(crate) mod tpu;

#[cfg(lumen_mps_linked)]
pub(crate) use backward::backward_of_body_into;
pub(crate) use backward::{
    AttentionBackwards, Backward, FlashAttentionBackward, backward_of_body,
    backward_of_body_strided,
};
#[cfg(feature = "python")]
pub(crate) use forward::traced;
pub(crate) use forward::{
    Attention, Attentions, Epilogues, FlashAttention, of_body, of_body_strided,
};

/// Name of a forward flash-attention fusion.
pub(crate) const FORWARD: &str = "flash_attention";

/// Name of a backward flash-attention fusion.
pub(crate) const BACKWARD: &str = "flash_attention_backward";

/// How a kernel reads (or writes) a `[batch.., row, col]` operand as a strided view of `base`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Access {
    /// Graph value whose buffer holds the data.
    pub base: Var,
    /// Element offset of the view in `base`.
    pub offset: usize,
    /// `(stride, divisor)` per batch dim: index `i` addresses `i / divisor * stride`.
    pub batch: Vec<(usize, usize)>,
    /// Element stride of the row dim.
    pub row: usize,
    /// Element stride of the column dim.
    pub col: usize,
}

/// The batch extents as the attention kernels' grid launches them: the dims before the last merged into
/// one, and the last (the heads); 1 for an absent one.
pub(crate) fn grid_batch(batch: &[usize]) -> (usize, usize) {
    match batch.split_last() {
        Some((&heads, outer)) => (outer.iter().product(), heads),
        None => (1, 1),
    }
}

/// A term of an operand's element offset on the attention grid, each kernel writing it in its own syntax.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Term {
    /// The view's first element.
    Offset(usize),
    /// The outer batch index times this stride.
    Outer(usize),
    /// The head index, divided by `divisor` (heads sharing a KV head), times `stride`.
    Heads { stride: usize, divisor: usize },
}

impl Access {
    /// The non-zero batch terms of the access's element offset on the grid `grid_batch` gives `batch`: its
    /// offset, the merged outer dims' stride and the heads' stride with its divisor. Fails when the outer
    /// dims don't step as one dim (each over the whole of the one inside it) or carry a divisor.
    pub(crate) fn batch_terms(&self, batch: &[usize]) -> Result<Vec<Term>, String> {
        let mut terms = Vec::new();

        if self.offset != 0 {
            terms.push(Term::Offset(self.offset));
        }

        let Some((&(stride, divisor), outer)) = self.batch.split_last() else {
            return Ok(terms);
        };

        // The outer dims of more than one element, outermost first.
        let dims: Vec<(usize, (usize, usize))> = batch
            .iter()
            .copied()
            .zip(outer.iter().copied())
            .filter(|&(n, _)| n != 1)
            .collect();

        let steps = dims.windows(2).all(|w| w[0].1.0 == w[1].1.0 * w[1].0);

        if !steps || dims.iter().any(|&(_, (_, divisor))| divisor != 1) {
            return Err(
                "flash attention: batch dims before the heads that don't step as one dim".into(),
            );
        }

        if let Some(&(_, (outer, _))) = dims.last().filter(|&&(_, (s, _))| s != 0) {
            terms.push(Term::Outer(outer));
        }

        if stride != 0 {
            terms.push(Term::Heads { stride, divisor });
        }

        Ok(terms)
    }
}

/// Dropout on the attention probabilities, from `random_bits` compared against a threshold.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Dropout {
    /// Seed operand of `random_bits`.
    pub seed: Var,
    /// Counter start operand of `random_bits`.
    pub start: Var,
    /// Static counter offset of `random_bits`.
    pub offset: u64,
    /// Probabilities whose random bits are below this are dropped.
    pub threshold: u32,
    /// Scale applied to the kept probabilities.
    pub scale: f64,
    /// The boolean drop mask value in the graph.
    pub mask: Var,
}

/// One op applied to the attention scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Score {
    /// Rounds to the dtype (a cast).
    Round(DType),
    /// Multiplies by a constant, computing in the dtype.
    Mul(f64, DType),
}

/// Where a fusion body's operands live: element strides of its inputs read through views and
/// `(offset, strides)` of its outputs written into views, by input and output index; others are
/// contiguous in their own buffers.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Strides {
    pub(crate) inputs: HashMap<usize, Vec<usize>>,
    pub(crate) outputs: HashMap<usize, (usize, Vec<usize>)>,
}

impl Strides {
    /// The layouts of `body`'s inputs and outputs as `inputs` and `outputs` strides give them: an
    /// input or output with contiguous strides (and no offset) is left out.
    pub(crate) fn of(body: &Graph, inputs: &[Vec<usize>], outputs: &[(usize, Vec<usize>)]) -> Self {
        let contiguous =
            |v: Var, strides: &[usize]| contiguous_strides(&body.type_of(v).shape) == strides;

        Strides {
            inputs: body
                .inputs()
                .iter()
                .zip(inputs)
                .enumerate()
                .filter(|(_, (v, s))| !contiguous(**v, s))
                .map(|(k, (_, s))| (k, s.clone()))
                .collect(),
            outputs: body
                .outputs()
                .iter()
                .zip(outputs)
                .enumerate()
                .filter(|(_, (v, (o, s)))| *o != 0 || !contiguous(**v, s))
                .map(|(k, (_, view))| (k, view.clone()))
                .collect(),
        }
    }
}

/// Extracts the subgraph computing `outputs` from `bases` as a standalone body with `bases` as inputs.
pub(crate) fn body_of(
    graph: &Graph,
    producer: &[Option<usize>],
    outputs: &[Var],
    bases: &[Var],
) -> Graph {
    let nodes = graph.nodes();
    let mut members = Vec::new();
    let mut stack: Vec<Var> = outputs.to_vec();

    while let Some(v) = stack.pop() {
        if bases.contains(&v) {
            continue;
        }

        let n = producer[v].expect("a value the fusion computes");

        if !members.contains(&n) {
            members.push(n);
            stack.extend(&nodes[n].inputs);
        }
    }

    members.sort_unstable();
    let mut body = Graph::new();
    let mut var: HashMap<Var, Var> = HashMap::new();

    for &b in bases {
        var.insert(b, body.input(graph.type_of(b).clone()));
    }

    for &n in &members {
        let node = &nodes[n];
        let ins: Vec<Var> = node.inputs.iter().map(|v| var[v]).collect();
        body.set_origin(node);

        let v = body
            .apply(node.primitive.clone(), &ins)
            .expect("a node of the graph");

        var.insert(node.output, v);
    }

    let outputs: Vec<Var> = outputs.iter().map(|v| var[v]).collect();
    body.set_outputs(&outputs).expect("the fusion's outputs");
    body
}

/// Whether every element offset of a `rows x cols` view fits in 32 bits.
fn fits32(acc: &Access, rows: usize, cols: usize) -> bool {
    (acc.row * (rows - 1) + acc.col * (cols - 1)) < (1 << 32)
}

/// Graph indices used to match attention patterns.
struct Matcher<'a> {
    graph: &'a Graph,
    /// Node producing each value.
    producer: Vec<Option<usize>>,
    /// Live nodes reading each value, in order.
    readers: Vec<Vec<usize>>,
    /// Whether each value is a graph output.
    output: Vec<bool>,
    /// Whether each node feeds a graph output (all nodes while still tracing).
    live: Vec<bool>,
    /// Match dQ only as the dot or a single transpose of it (for a kernel writing all three gradients).
    plain_dq: bool,
    /// Allow a causal offset from a rank-0 integer input.
    runtime_offsets: bool,
    /// Body outputs written into a view: `(offset, strides)` of their buffer.
    views: HashMap<Var, (usize, Vec<usize>)>,
    /// Body inputs read through a view: their element strides.
    strided: HashMap<Var, Vec<usize>>,
}

impl<'a> Matcher<'a> {
    /// Matcher over `graph` with producers, readers and liveness computed.
    fn new(graph: &'a Graph) -> Self {
        let nodes = graph.nodes();
        let n = graph.types.len();
        let mut producer = vec![None; n];

        for (i, node) in nodes.iter().enumerate() {
            producer[node.output] = Some(i);
        }

        let mut output = vec![false; n];
        graph.outputs().iter().for_each(|&v| output[v] = true);
        let tracing = graph.outputs().is_empty();
        let mut live = vec![tracing; nodes.len()];
        let mut live_var = output.clone();

        for (i, node) in nodes.iter().enumerate().rev() {
            if live_var[node.output] {
                live[i] = true;
                node.inputs.iter().for_each(|&v| live_var[v] = true);
            }
        }

        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (i, node) in nodes.iter().enumerate().filter(|&(i, _)| live[i]) {
            for &v in &node.inputs {
                if readers[v].last() != Some(&i) {
                    readers[v].push(i);
                }
            }
        }

        Matcher {
            plain_dq: false,
            runtime_offsets: false,
            views: HashMap::new(),
            strided: HashMap::new(),
            graph,
            producer,
            readers,
            output,
            live,
        }
    }

    /// The matcher with the body's inputs and outputs laid out as `strides` gives them.
    fn laid_out(mut self, strides: &Strides) -> Self {
        let graph = self.graph;

        self.views = strides
            .outputs
            .iter()
            .map(|(&k, view)| (graph.outputs()[k], view.clone()))
            .collect();

        self.strided = strides
            .inputs
            .iter()
            .map(|(&k, s)| (graph.inputs()[k], s.clone()))
            .collect();

        self
    }

    /// Node producing `v`, if any.
    fn node(&self, v: Var) -> Option<&'a Node> {
        self.producer[v].map(|i| &self.graph.nodes()[i])
    }

    /// Whether `v` has exactly one reader and isn't a graph output.
    fn single(&self, v: Var) -> bool {
        !self.output[v] && self.readers[v].len() == 1
    }

    /// The constant `v` holds when it is a broadcast, reshape or cast of a `full`.
    fn scalar(&self, v: Var) -> Option<f64> {
        if !constant(self.graph, &self.producer, v) {
            return None;
        }

        let mut v = v;

        loop {
            let n = self.node(v)?;

            match &n.primitive {
                Primitive::Full { fill_value, .. } => return Some(fill_value.to_f64()),
                Primitive::BroadcastInDim { .. }
                | Primitive::Reshape { .. }
                | Primitive::Cast { .. } => v = n.inputs[0],
                _ => return None,
            }
        }
    }

    /// Matches `select(mask, 0, x * scale)` dropout on a `shape`d tensor. Returns `x`, the dropout,
    /// and the dtype the scaling computes in.
    fn dropout(&self, v: Var, shape: &[usize], transposed: bool) -> Option<(Var, Dropout, DType)> {
        use Primitive::*;

        let n = self
            .node(v)
            .filter(|n| matches!(n.primitive, Select) && self.single(v))?;

        if self.scalar(n.inputs[1]) != Some(0.0) {
            return None;
        }

        let m = n.inputs[2];

        let mul = self
            .node(m)
            .filter(|n| matches!(n.primitive, Mul) && self.single(m))?;

        let (a, b) = (mul.inputs[0], mul.inputs[1]);

        let (scale, x) = match self.scalar(b) {
            Some(c) => (c, a),
            None => (self.scalar(a)?, b),
        };

        let mut dropout = self.mask(n.inputs[0], shape, transposed)?;
        dropout.scale = scale;
        Some((x, dropout, self.graph.type_of(m).dtype))
    }

    /// Matches a drop mask `random_bits < threshold` of `shape`, transposed in its last two dims when
    /// `transposed`.
    fn mask(&self, v: Var, shape: &[usize], transposed: bool) -> Option<Dropout> {
        use Primitive::*;
        let rank = shape.len();

        let swaps = |p: &[usize]| {
            let mut swapped: Vec<usize> = (0..rank).collect();
            swapped.swap(rank - 2, rank - 1);
            p == swapped
        };

        let (mut x, mut flipped, mut threshold) = (v, false, None);

        loop {
            let n = self.node(x)?;

            match &n.primitive {
                Transpose { permutation } if swaps(permutation) => flipped = !flipped,
                Lt if threshold.is_none() => threshold = Some(self.scalar(n.inputs[1])?),
                RandomBits { shape: s, offset } => {
                    let t = threshold?;
                    let whole = t.fract() == 0.0 && (0.0..4294967296.0).contains(&t);

                    return (s == shape && flipped == transposed && whole).then(|| Dropout {
                        seed: n.inputs[0],
                        start: n.inputs[1],
                        offset: *offset,
                        threshold: t as u32,
                        scale: 1.0,
                        mask: v,
                    });
                }
                _ => return None,
            }

            x = n.inputs[0];
        }
    }

    /// Follows dot `i` through transposes (and with `chain`, reshapes and one float cast) to where it
    /// is written; returns that node and the write as an [`Access`] over the dot's dims. With `shared`,
    /// the dot's one transpose is followed though others read the dot too.
    fn written(&self, i: usize, nb: usize, chain: bool, shared: bool) -> Option<(usize, Access)> {
        use Primitive::*;
        let nodes = self.graph.nodes();
        let mut path = vec![i];
        let mut cast = false;

        loop {
            let o = nodes[*path.last().expect("the dot")].output;

            let transposes: Vec<usize> = self.readers[o]
                .iter()
                .copied()
                .filter(|&r| matches!(nodes[r].primitive, Transpose { .. }))
                .collect();

            let r = match (&self.readers[o][..], &transposes[..]) {
                (&[r], _) if !self.output[o] && (chain || path.len() == 1) => r,
                (_, &[r]) if shared && !self.output[o] && path.len() == 1 => r,
                _ => break,
            };

            match &nodes[r].primitive {
                Transpose { .. } => {}
                Reshape { .. } if chain => {}
                Cast { new_dtype } if chain && !cast && new_dtype.is_float() => cast = true,
                _ => break,
            }

            path.push(r);
        }

        let root = *path.last().expect("the dot");
        let value = nodes[root].output;

        let (offset, strides) = match self.views.get(&value) {
            Some(view) => view.clone(),
            None => (0, contiguous_strides(&self.graph.type_of(value).shape)),
        };

        // Map the written buffer's (stride, divisor) dims back through the path to the dot's dims.
        let mut dims: Vec<(usize, usize)> = strides.into_iter().map(|s| (s, 1)).collect();

        for &k in path[1..].iter().rev() {
            let node = &nodes[k];

            dims = match &node.primitive {
                Transpose { permutation } => {
                    let mut operand = vec![(0, 1); dims.len()];

                    for (d, &p) in permutation.iter().enumerate() {
                        operand[p] = dims[d];
                    }

                    operand
                }
                Reshape { .. } => {
                    let from = &self.graph.type_of(node.output).shape;
                    let shape = &self.graph.type_of(node.inputs[0]).shape;
                    let operand = reshaped(from, &dims, shape)?;

                    operand
                        .iter()
                        .all(|&(_, div)| div == 1)
                        .then_some(operand)?
                }
                _ => dims,
            };
        }

        let out = Access {
            base: value,
            offset,
            batch: dims[..nb].to_vec(),
            row: dims[nb].0,
            col: dims[nb + 1].0,
        };

        Some((root, out))
    }

    /// Walks back from scores `x` through casts, one constant scale and one causal mask to the
    /// `Q K^T` dot. Returns the dot, the score ops in application order and the causal mask.
    fn scores(
        &self,
        x: Var,
        nb: usize,
        transposed: bool,
    ) -> Option<(usize, Vec<Score>, Option<Causal>)> {
        use Primitive::*;
        let ty = |v: Var| self.graph.type_of(v);
        let (mut v, mut scale, mut causal) = (x, None, None);
        let mut scores = Vec::new();

        let s = loop {
            let n = self.node(v)?;

            if v != x && !self.single(v) {
                return None;
            }

            let dtype = ty(n.output).dtype;

            match &n.primitive {
                DotGeneral { .. } => break self.producer[v]?,
                Cast { .. } => {
                    scores.push(Score::Round(dtype));
                    v = n.inputs[0];
                }
                Mul if scale.is_none() => {
                    let (a, b) = (n.inputs[0], n.inputs[1]);

                    let (c, rest) = match self.scalar(b) {
                        Some(c) => (c, a),
                        None => (self.scalar(a)?, b),
                    };

                    scale = Some(c);
                    scores.push(Score::Mul(c, dtype));
                    v = rest;
                }
                Select if causal.is_none() => {
                    if self.scalar(n.inputs[2]) != Some(f64::NEG_INFINITY) {
                        return None;
                    }

                    causal = Some(self.causal(n.inputs[0], nb, transposed)?);
                    v = n.inputs[1];
                }
                _ => return None,
            }
        };

        if scale.is_some_and(|c| !(c > 0.0 && c.is_finite())) {
            return None;
        }

        // A dot writing its float32 accumulator is rounded by the cast right after it, if any.
        let dtype = ty(self.graph.nodes()[s].output).dtype;

        if dtype != DType::F32 || !matches!(scores.last(), Some(Score::Round(_))) {
            scores.push(Score::Round(dtype));
        }

        scores.reverse();
        Some((s, scores, causal))
    }

    /// Access of `v` with batch dims `b` and the given row/col dims, which must be undivided.
    fn access(&self, v: Var, b: &[usize], row: usize, col: usize) -> Option<Access> {
        let (base, offset, dims) = self.view(v);
        let one = |d: usize| (dims[d].1 == 1).then_some(dims[d].0);

        Some(Access {
            base,
            offset,
            batch: b.iter().map(|&d| dims[d]).collect(),
            row: one(row)?,
            col: one(col)?,
        })
    }

    /// Matches a causal mask `key < query + c` (or `<=` built as `lt | eq`) on iota coordinates.
    /// Returns the diagonal offset and an optional runtime offset; `transposed` swaps query and key dims.
    fn causal(&self, v: Var, nb: usize, transposed: bool) -> Option<Causal> {
        use Primitive::*;
        let shape = self.graph.type_of(v).shape.clone();
        let (v, dims) = self.broadcast_dims(v);
        let n = self.node(v)?;

        let (a, b, inclusive) = match n.primitive {
            Lt => (n.inputs[0], n.inputs[1], false),
            Select => {
                let less = self
                    .node(n.inputs[0])
                    .filter(|l| matches!(l.primitive, Lt))?;

                let eq = self
                    .node(n.inputs[2])
                    .filter(|e| matches!(e.primitive, Eq))?;

                let swapped = [eq.inputs[1], eq.inputs[0]];

                if n.inputs[1] != n.inputs[0]
                    || (less.inputs != eq.inputs && less.inputs != swapped)
                {
                    return None;
                }

                (less.inputs[0], less.inputs[1], true)
            }
            _ => return None,
        };

        let (a, b) = (self.affine(a)?, self.affine(b)?);
        let mut outer = vec![0; nb + 2];

        for (k, (x, y)) in a.coefficients.iter().zip(&b.coefficients).enumerate() {
            match (dims[k], x - y) {
                (_, 0) => {}
                (Some(d), c) if d < outer.len() => outer[d] = c,
                _ => return None,
            }
        }

        // The mask must compare exactly `key - query`; size-1 dims accept either coefficient.
        let (query, key) = if transposed {
            (nb + 1, nb)
        } else {
            (nb, nb + 1)
        };

        for (d, c) in [(query, -1), (key, 1)] {
            if shape.get(d) == Some(&1) {
                outer[d] = c;
            }
        }

        if outer[..nb].iter().any(|&c| c != 0) || (outer[query], outer[key]) != (-1, 1) {
            return None;
        }

        let offset = match sum_terms(a.term, b.term.map(|(v, k)| (v, -k))) {
            None => None,
            Some((v, -1)) => Some(v),
            Some(_) => return None,
        };

        // Normalize to "kept when key <= query + c".
        let c0 = a.constant - b.constant;
        Some((if inclusive { -c0 } else { -c0 - 1 }, offset))
    }

    /// `v` with broadcasts and unit reshapes removed, and for each of its dims the outer dim it maps to.
    fn broadcast_dims(&self, mut v: Var) -> (Var, Vec<Option<usize>>) {
        let mut dims: Vec<Option<usize>> =
            (0..self.graph.type_of(v).shape.len()).map(Some).collect();

        while let Some(n) = self.node(v) {
            let inner = &self.graph.type_of(n.inputs[0]).shape;

            let mapped = match &n.primitive {
                Primitive::BroadcastInDim {
                    broadcast_dimensions,
                    ..
                } => broadcast_dimensions.iter().map(|&d| dims[d]).collect(),
                Primitive::Reshape { .. } => {
                    let outer = &self.graph.type_of(v).shape;

                    let Some(map) = unit_reshape(inner, outer) else {
                        break;
                    };

                    map.iter().map(|d| d.and_then(|d| dims[d])).collect()
                }
                _ => break,
            };

            dims = mapped;
            v = n.inputs[0];
        }

        (v, dims)
    }

    /// `v` as an affine function of its coordinates plus a constant and an optional runtime scalar.
    fn affine(&self, v: Var) -> Option<Affine> {
        use Primitive::*;
        let ty = self.graph.type_of(v);
        let rank = ty.shape.len();

        let constant = |c: i64| Affine {
            coefficients: vec![0; rank],
            constant: c,
            term: None,
        };

        let Some(n) = self.node(v) else {
            let runtime = self.runtime_offsets && rank == 0 && ty.dtype.is_int();

            return runtime.then_some(Affine {
                term: Some((v, 1)),
                ..constant(0)
            });
        };

        match &n.primitive {
            Iota { dimension, .. } => {
                let mut a = constant(0);
                a.coefficients[*dimension] = 1;
                Some(a)
            }
            Full { fill_value, .. } => {
                let c = fill_value.to_f64();
                (c.fract() == 0.0).then(|| constant(c as i64))
            }
            Cast { .. } => self.affine(n.inputs[0]),
            BroadcastInDim {
                broadcast_dimensions,
                ..
            } => {
                let a = self.affine(n.inputs[0])?;
                let inner = &self.graph.type_of(n.inputs[0]).shape;
                let mut out = constant(a.constant);
                out.term = a.term;

                for (k, &d) in broadcast_dimensions.iter().enumerate() {
                    if inner[k] == ty.shape[d] {
                        out.coefficients[d] = a.coefficients[k];
                    }
                }

                Some(out)
            }
            Reshape { .. } => {
                let a = self.affine(n.inputs[0])?;
                let map = unit_reshape(&self.graph.type_of(n.inputs[0]).shape, &ty.shape)?;
                let mut out = constant(a.constant);
                out.term = a.term;

                for (k, d) in map.into_iter().enumerate() {
                    if let Some(d) = d {
                        out.coefficients[d] = a.coefficients[k];
                    }
                }

                Some(out)
            }
            Add | Sub => {
                let (a, b) = (self.affine(n.inputs[0])?, self.affine(n.inputs[1])?);
                let s = if matches!(n.primitive, Add) { 1 } else { -1 };

                let term = match (a.term, b.term) {
                    (Some((x, _)), Some((y, _))) if x != y => return None,
                    (at, bt) => sum_terms(at, bt.map(|(v, k)| (v, s * k))),
                };

                let coefficients = (a.coefficients.iter().zip(&b.coefficients))
                    .map(|(x, y)| x + s * y)
                    .collect();

                Some(Affine {
                    coefficients,
                    constant: a.constant + s * b.constant,
                    term,
                })
            }
            _ => None,
        }
    }

    /// `v` as a view of a base value: the base, element offset and `(stride, divisor)` per dim.
    fn view(&self, v: Var) -> (Var, usize, Vec<(usize, usize)>) {
        use Primitive::*;

        let own = || {
            let shape = &self.graph.type_of(v).shape;

            let strides = match self.strided.get(&v) {
                Some(strides) => strides.clone(),
                None => contiguous_strides(shape),
            };

            (v, 0, strides.into_iter().map(|s| (s, 1)).collect())
        };

        let Some(n) = self.node(v) else {
            return own();
        };

        let layout = matches!(
            n.primitive,
            Transpose { .. } | Slice { .. } | BroadcastInDim { .. } | Reshape { .. }
        );

        if !layout {
            return own();
        }

        let x = n.inputs[0];
        let (base, mut offset, dims) = self.view(x);
        let xs = &self.graph.type_of(x).shape;
        let shape = &self.graph.type_of(v).shape;

        let dims = match &n.primitive {
            Transpose { permutation } => permutation.iter().map(|&p| dims[p]).collect(),
            Slice { start_indices, .. } => {
                for (d, &s) in start_indices.iter().enumerate() {
                    if s != 0 && dims[d].1 != 1 {
                        return own();
                    }

                    offset += s * dims[d].0;
                }

                dims
            }
            BroadcastInDim {
                broadcast_dimensions,
                ..
            } => {
                let mut out = vec![(0, 1); shape.len()];

                for (k, &d) in broadcast_dimensions.iter().enumerate() {
                    if xs[k] == shape[d] {
                        out[d] = dims[k];
                    }
                }

                out
            }
            Reshape { .. } => match reshaped(xs, &dims, shape) {
                Some(out) => out,
                None => return own(),
            },
            _ => unreachable!("a layout primitive"),
        };

        (base, offset, dims)
    }
}

/// Causal diagonal offset and optional runtime offset value.
type Causal = (i64, Option<Var>);

/// An integer tensor as `coefficients . coords + constant + coefficient * var`.
struct Affine {
    coefficients: Vec<i64>,
    constant: i64,
    term: Option<(Var, i64)>,
}

/// Sum of two runtime terms; `None` when they cancel or name different values.
fn sum_terms(a: Option<(Var, i64)>, b: Option<(Var, i64)>) -> Option<(Var, i64)> {
    match (a, b) {
        (Some((v, x)), Some((w, y))) if v == w => (x + y != 0).then_some((v, x + y)),
        (Some(_), Some(_)) => None,
        (t, None) | (None, t) => t,
    }
}

/// For a reshape that only adds or drops size-1 dims, the outer dim each inner dim maps to.
fn unit_reshape(inner: &[usize], outer: &[usize]) -> Option<Vec<Option<usize>>> {
    let kept =
        |shape: &[usize]| -> Vec<usize> { (0..shape.len()).filter(|&d| shape[d] != 1).collect() };

    let (a, b) = (kept(inner), kept(outer));

    if a.len() != b.len() || a.iter().zip(&b).any(|(&x, &y)| inner[x] != outer[y]) {
        return None;
    }

    let mut map = vec![None; inner.len()];

    for (&x, &y) in a.iter().zip(&b) {
        map[x] = Some(y);
    }

    Some(map)
}

/// Re-expresses per-dim `(stride, divisor)` of a view of shape `from` after reshaping it to `shape`,
/// or `None` when the result isn't a strided view.
fn reshaped(
    from: &[usize],
    dims: &[(usize, usize)],
    shape: &[usize],
) -> Option<Vec<(usize, usize)>> {
    let ins: Vec<(usize, (usize, usize))> = from
        .iter()
        .zip(dims)
        .filter(|(n, _)| **n != 1)
        .map(|(&n, &d)| (n, d))
        .collect();

    let outs: Vec<usize> = (0..shape.len()).filter(|&d| shape[d] != 1).collect();
    let mut out = vec![(0, 1); shape.len()];
    let (mut a, mut b) = (0, 0);

    // Pair up runs of input and output dims with equal products.
    while a < ins.len() || b < outs.len() {
        let (mut pa, mut pb) = (ins.get(a)?.0, shape[*outs.get(b)?]);
        let (a0, b0) = (a, b);

        while pa != pb {
            if pa < pb {
                a += 1;
                pa *= ins.get(a)?.0;
            } else {
                b += 1;
                pb *= shape[*outs.get(b)?];
            }
        }

        let group = &ins[a0..=a];
        let targets = &outs[b0..=b];

        match (group, targets) {
            ([(_, d)], [t]) => out[*t] = *d,
            // Splitting an undivided dim: strides follow the output shape.
            ([(_, (s, 1))], ts) => {
                let mut stride = *s;

                for &t in ts.iter().rev() {
                    out[t] = (stride, 1);
                    stride *= shape[t];
                }
            }
            // Merging dims: trailing broadcast dims become a divisor, otherwise they must be contiguous.
            (gs, [t]) => {
                let (_, (s0, d0)) = gs[0];
                let rest = &gs[1..];

                if rest.iter().all(|(_, (s, d))| *s == 0 && *d == 1) {
                    let inner: usize = rest.iter().map(|(n, _)| n).product();
                    out[*t] = (s0, d0 * inner);
                } else if gs
                    .windows(2)
                    .all(|w| w[0].1.1 == 1 && w[1].1.1 == 1 && w[0].1.0 == w[1].1.0 * w[1].0)
                {
                    out[*t] = (gs.last()?.1.0, 1);
                } else {
                    return None;
                }
            }
            _ => return None,
        }

        a += 1;
        b += 1;
    }

    Some(out)
}

/// What each device runs for this pass (each device's in its own file here); a device it doesn't run on runs the
/// identity pass.
pub(crate) trait AttentionDevice: Sized + 'static {
    /// Replaces each forward attention with a `flash_attention` fusion.
    fn flash_attention<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Identity)
    }

    /// Replaces attention's gradients with `flash_attention_backward` fusions.
    fn flash_attention_backward<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Identity)
    }

    /// Folds output layouts into the attention fusions, for the device's attention kernels.
    fn attention_layouts<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Identity)
    }
}
