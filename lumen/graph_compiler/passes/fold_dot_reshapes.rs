use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};

pub(crate) fn fold_dot_reshapes(graph: &Graph) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    for (i, node) in graph.nodes().iter().enumerate() {
        producer[node.output] = Some(i);
    }
    for node in graph.nodes() {
        out.set_origin(node);
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let folded = match &node.primitive {
            Primitive::DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                accum_dtype,
                output_dtype,
            } if lhs_batch.is_empty() && *lhs_contracting == [1] => producer[node.inputs[0]]
                .map(|p| &graph.nodes()[p])
                .filter(|r| matches!(r.primitive, Primitive::Reshape { .. }))
                .map(|r| r.inputs[0])
                .filter(|&x| {
                    let (x, lhs) = (
                        &graph.type_of(x).shape,
                        &graph.type_of(node.inputs[0]).shape,
                    );
                    x.len() > 2 && lhs.len() == 2 && x.last() == lhs.last()
                })
                .map(|x| {
                    let rank = graph.type_of(x).shape.len();
                    let dot = Primitive::DotGeneral {
                        lhs_contracting: vec![rank - 1],
                        rhs_contracting: rhs_contracting.clone(),
                        lhs_batch: Vec::new(),
                        rhs_batch: rhs_batch.clone(),
                        accum_dtype: *accum_dtype,
                        output_dtype: *output_dtype,
                    };
                    let v = out
                        .apply(dot, &[map[x], inputs[1]])
                        .expect("a dot of the same elements");
                    let reshape = Primitive::Reshape {
                        new_sizes: graph.type_of(node.output).shape.clone(),
                    };
                    out.apply(reshape, &[v]).expect("the same elements")
                }),
            _ => None,
        };
        map[node.output] = folded.unwrap_or_else(|| {
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

pub(crate) struct FoldDotReshapes;

impl<M: Module> CompilerPass<M> for FoldDotReshapes {
    fn name(&self) -> &'static str {
        "fold_dot_reshapes"
    }

    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = fold_dot_reshapes(m.graph());
        Ok(())
    }
}
