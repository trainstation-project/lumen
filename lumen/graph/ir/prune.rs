//! Pruning: the graph without the nodes no output needs.

use super::{Graph, Var};

impl Graph {
    /// Copies the graph keeping only nodes that reach an output; also returns the old-to-new value map
    /// (`None` for dropped values). All inputs are kept.
    pub fn prune(&self) -> (Graph, Vec<Option<Var>>) {
        // Walk nodes backwards marking every value an output depends on.
        let mut live = vec![false; self.types.len()];

        for &v in &self.outputs {
            live[v] = true;
        }

        for node in self.nodes.iter().rev() {
            if live[node.output] {
                for &v in &node.inputs {
                    live[v] = true;
                }
            }
        }

        let mut pruned = Graph::new();
        let mut map: Vec<Option<Var>> = vec![None; self.types.len()];

        for &v in &self.inputs {
            map[v] = Some(pruned.input(self.types[v].clone()));
        }

        for node in self.nodes.iter().filter(|n| live[n.output]) {
            pruned.set_origin(node);

            let inputs: Vec<Var> = node
                .inputs
                .iter()
                .map(|&v| map[v].expect("an earlier value"))
                .collect();

            map[node.output] = Some(
                pruned
                    .apply(node.primitive.clone(), &inputs)
                    .expect("a node of the graph"),
            );

            pruned.set_label(map[node.output].expect("just made"), node.label);
        }

        let outputs: Vec<Var> = self
            .outputs
            .iter()
            .map(|&v| map[v].expect("a live value"))
            .collect();

        pruned.set_outputs(&outputs).expect("its values");
        (pruned, map)
    }
}
