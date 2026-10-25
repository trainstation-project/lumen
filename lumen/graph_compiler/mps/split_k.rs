use super::fusion::fusible;
use crate::DType;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};
use crate::kernels::dot_general::mps::float_groups;

const GROUPS: usize = 256;
const MAX_SPLIT: usize = 8;
const MIN_CHUNK: usize = 256;
const MIN_K: usize = 2048;

pub(super) fn split_k(graph: &Graph, atomic: bool, fuse: bool) -> Graph {
    let nodes = graph.nodes();
    let sliced = |v: Var| {
        nodes
            .iter()
            .any(|n| n.output == v && matches!(n.primitive, Primitive::Slice { .. }))
    };
    let fused = |v: Var| {
        nodes
            .iter()
            .any(|n| n.inputs.contains(&v) && fusible(graph, n))
    };
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in nodes {
        out.set_origin(node);
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let apply =
            |out: &mut Graph, p: Primitive, ins: &[Var]| out.apply(p, ins).expect("well typed");
        let Primitive::DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            accum_dtype,
            output_dtype,
            ..
        } = &node.primitive
        else {
            map[node.output] = apply(&mut out, node.primitive.clone(), &inputs);
            out.set_label(map[node.output], node.label);
            continue;
        };
        let (lhs, rhs) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));
        let split = match (&lhs_contracting[..], &rhs_contracting[..]) {
            (&[l], &[r])
                if lhs_batch.is_empty()
                    && *accum_dtype == DType::F32
                    && !node.inputs.iter().any(|&v| sliced(v)) =>
            {
                let m = lhs.numel() / lhs.shape[l];
                let n = rhs.numel() / rhs.shape[r];
                chunks(m, n, lhs.shape[l]).map(|s| (l, r, s))
            }
            _ => None,
        };
        let Some((l, r, s)) = split else {
            map[node.output] = apply(&mut out, node.primitive.clone(), &inputs);
            out.set_label(map[node.output], node.label);
            continue;
        };
        let reshaped = |shape: &[usize], d: usize| {
            let mut shape = shape.to_vec();
            let k = shape[d];
            shape.splice(d..=d, [s, k / s]);
            Primitive::Reshape { new_sizes: shape }
        };
        let a = apply(&mut out, reshaped(&lhs.shape, l), &[inputs[0]]);
        let b = apply(&mut out, reshaped(&rhs.shape, r), &[inputs[1]]);
        let dot = Primitive::DotGeneral {
            lhs_contracting: vec![l + 1],
            rhs_contracting: vec![r + 1],
            lhs_batch: vec![l],
            rhs_batch: vec![r],
            accum_dtype: DType::F32,
            output_dtype: DType::F32,
        };
        let sum = Primitive::ReduceSum {
            axes: vec![0],
            accum_dtype: DType::F32,
        };
        let partials = 2 * s * lhs.numel() / lhs.shape[l] * (rhs.numel() / rhs.shape[r]) * 4;
        let operands = (lhs.numel() + rhs.numel()) * lhs.dtype.size_of();
        let mut value = match atomic && partials > operands && !(fuse && fused(node.output)) {
            true => {
                let mut body = Graph::new();
                let ins = [a, b].map(|v| body.input(out.type_of(v).clone()));
                let partials = body.apply(dot, &ins).expect("well typed");
                let sum = body.apply(sum, &[partials]).expect("well typed");
                body.set_outputs(&[sum]).expect("its value");
                let fusion = Primitive::Fusion {
                    name: "split_k".into(),
                    label: "dot_general (split-K)",
                    body,
                };
                out.apply(fusion, &[a, b]).expect("well typed")
            }
            false => {
                let partials = apply(&mut out, dot, &[a, b]);
                out.set_label(partials, Some("dot_general (split-K)"));
                let sum = apply(&mut out, sum, &[partials]);
                out.set_label(sum, Some("reduce_sum (split-K)"));
                sum
            }
        };
        if *output_dtype != DType::F32 {
            value = apply(
                &mut out,
                Primitive::Cast {
                    new_dtype: *output_dtype,
                },
                &[value],
            );
        }
        map[node.output] = value;
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

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

fn chunks(m: usize, n: usize, k: usize) -> Option<usize> {
    let groups = float_groups(1, m, n);
    if groups == 0 || 2 * groups >= GROUPS || k < MIN_K {
        return None;
    }
    let mut s = GROUPS.div_ceil(groups).next_power_of_two().min(MAX_SPLIT);
    while s >= 2 && (!k.is_multiple_of(s) || k / s < MIN_CHUNK) {
        s /= 2;
    }
    (s >= 2).then_some(s)
}

pub(super) struct SplitK;

impl<M: Module> CompilerPass<M> for SplitK {
    fn name(&self) -> &'static str {
        "split_k"
    }

    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let config = &m.options().config.mps;
        *m.graph_mut() = split_k(m.graph(), !config.deterministic, config.fuse);
        Ok(())
    }
}
