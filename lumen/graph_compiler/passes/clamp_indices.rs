use crate::Scalar;
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};

fn clamp_indices(graph: &Graph) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in graph.nodes() {
        out.set_origin(node);
        let mut inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let index = match node.primitive {
            Primitive::Gather { axis } | Primitive::ScatterAdd { axis } => {
                Some((1, graph.type_of(node.inputs[0]).shape[axis]))
            }
            Primitive::LinearCrossEntropy => Some((2, graph.type_of(node.inputs[1]).shape[0])),
            _ => None,
        };
        if let Some((k, n)) = index {
            inputs[k] = clamped(&mut out, inputs[k], n);
        }
        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("well typed");
        out.set_label(map[node.output], node.label);
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

fn clamped(out: &mut Graph, i: Var, n: usize) -> Var {
    let ty = out.type_of(i).clone();
    let mut apply = |p: Primitive, ins: &[Var]| out.apply(p, ins).expect("well typed");
    let full = |c: usize| Primitive::Full {
        shape: ty.shape.clone(),
        fill_value: Scalar::Int(c as i64),
        dtype: ty.dtype,
    };
    let (zero, last) = (apply(full(0), &[]), apply(full(n.max(1) - 1), &[]));
    let low = apply(Primitive::Max, &[i, zero]);
    let past = apply(Primitive::Lt, &[last, low]);
    apply(Primitive::Select, &[past, last, low])
}

pub(crate) struct ClampIndices;

impl<M: Module> CompilerPass<M> for ClampIndices {
    fn name(&self) -> &'static str {
        "clamp_indices"
    }

    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = clamp_indices(m.graph());
        Ok(())
    }
}
