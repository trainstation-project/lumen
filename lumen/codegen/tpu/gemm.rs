use std::collections::HashMap;

use crate::DType;
use crate::graph::{Graph, Node, Primitive, Var};

pub(crate) struct Epilogue {
    pub(crate) lhs: Vec<usize>,
    pub(crate) lhs_axis: usize,
    pub(crate) together: bool,
    pub(crate) rows: bool,
    pub(crate) widths: Vec<usize>,
    pub(crate) inputs: Vec<(DType, bool, usize)>,
    pub(crate) nodes: Vec<String>,
    pub(crate) outputs: Vec<(usize, usize, DType)>,
}

impl Epilogue {
    pub(crate) fn of(body: &Graph) -> (Epilogue, &Node) {
        let all = body.nodes();
        let at = all
            .iter()
            .position(|n| matches!(n.primitive, Primitive::DotGeneral { .. }))
            .expect("a dot");
        let dot = &all[at];
        let (lhs, lhs_axis) = match all[..at].iter().find(|n| n.output == dot.inputs[0]) {
            Some(Node {
                primitive: Primitive::Concatenate { dimension, .. },
                inputs,
                ..
            }) => (
                inputs
                    .iter()
                    .map(|&v| body.type_of(v).shape[*dimension])
                    .collect(),
                *dimension,
            ),
            _ => (Vec::new(), 0),
        };
        let first = lhs.len().max(1) + 1;
        let extras = &body.inputs()[first..];
        let shape = &body.type_of(dot.output).shape;
        let slices: Vec<&Node> = all[at + 1..]
            .iter()
            .filter(|n| matches!(n.primitive, Primitive::Slice { .. }) && n.inputs[0] == dot.output)
            .collect();
        let results = slices.len().max(1);
        let mut value: HashMap<Var, usize> = HashMap::from([(dot.output, 0)]);
        let mut member: HashMap<Var, usize> = HashMap::new();
        for (k, &v) in extras.iter().enumerate() {
            value.insert(v, results + k);
        }
        let axis = slices
            .first()
            .map_or(shape.len() - 1, |n| match &n.primitive {
                Primitive::Slice {
                    start_indices,
                    limit_indices,
                    ..
                } => (0..shape.len())
                    .find(|&d| limit_indices[d] - start_indices[d] != shape[d])
                    .unwrap_or(shape.len() - 1),
                _ => unreachable!("a slice"),
            });
        let (mut widths, mut nodes) = (Vec::new(), Vec::new());
        let mut readers: HashMap<Var, usize> = HashMap::new();
        let mut together = false;
        if slices.is_empty() {
            widths.push(shape[axis]);
            member.insert(dot.output, 0);
        }
        for node in &all[at + 1..] {
            if let Primitive::Slice {
                start_indices,
                limit_indices,
                ..
            } = &node.primitive
                && node.inputs[0] == dot.output
            {
                value.insert(node.output, widths.len());
                member.insert(node.output, widths.len());
                widths.push(limit_indices[axis] - start_indices[axis]);
                continue;
            }
            let args: Vec<usize> = node.inputs.iter().map(|v| value[v]).collect();
            nodes.push(super::node(&node.primitive, &args));
            value.insert(node.output, results + extras.len() + nodes.len() - 1);
            let mut of: Vec<usize> = node
                .inputs
                .iter()
                .filter_map(|v| member.get(v))
                .copied()
                .collect();
            of.sort_unstable();
            of.dedup();
            together |= of.len() > 1;
            if let Some(&p) = of.first() {
                member.insert(node.output, p);
                for &v in &node.inputs {
                    together |= readers.get(&v).is_some_and(|&q| q != p) && extras.contains(&v);
                    readers.entry(v).or_insert(p);
                }
            }
        }
        let epilogue = Epilogue {
            lhs,
            lhs_axis,
            together,
            rows: axis + 2 == shape.len(),
            widths,
            inputs: extras
                .iter()
                .map(|v| {
                    let ty = body.type_of(*v);
                    (
                        ty.dtype,
                        ty.shape.is_empty(),
                        readers.get(v).copied().unwrap_or(0),
                    )
                })
                .collect(),
            nodes,
            outputs: body
                .outputs()
                .iter()
                .map(|v| (value[v], member[v], body.type_of(*v).dtype))
                .collect(),
        };
        (epilogue, dot)
    }

    pub(crate) fn plain(n: usize, output: DType) -> Epilogue {
        Epilogue {
            lhs: Vec::new(),
            lhs_axis: 0,
            together: false,
            rows: false,
            widths: vec![n],
            inputs: Vec::new(),
            nodes: Vec::new(),
            outputs: vec![(0, 0, output)],
        }
    }
}
