//! Precision warnings: accumulations rounded to a narrow dtype and later cast back up.

use super::{FUSION_SEPARATOR, Graph, Node, Primitive, Var};

impl Graph {
    /// Finds accumulations (dots, sums) whose wide accumulator is rounded to a narrow output (a sum's own,
    /// a narrowing cast of a dot's) and then cast back up downstream; returns `(rounded value, cast,
    /// message)` for each, at its nearest such cast (a residual stream's later ones would repeat it once
    /// per layer).
    pub fn precision_warnings(&self) -> Vec<(Var, Var, String)> {
        use Primitive::*;
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); self.types.len()];
        let mut producer: Vec<Option<usize>> = vec![None; self.types.len()];

        for (i, node) in self.nodes.iter().enumerate() {
            producer[node.output] = Some(i);

            for &v in &node.inputs {
                readers[v].push(i);
            }
        }

        let mut warnings = Vec::new();

        for node in &self.nodes {
            // A dot's result is its accumulator: a cast that is its sole reader is where it is rounded (a
            // dot also read wide keeps its accumulator, the cast a later choice).
            let accumulation = match node.primitive {
                ReduceSum { .. } => node,
                Cast { .. } => match producer[node.inputs[0]].map(|p| &self.nodes[p]) {
                    Some(
                        dot @ Node {
                            primitive: DotGeneral { .. },
                            ..
                        },
                    ) if readers[dot.output].len() == 1 && !self.outputs.contains(&dot.output) => {
                        dot
                    }
                    _ => continue,
                },
                _ => continue,
            };

            let (DotGeneral { accum_dtype, .. } | ReduceSum { accum_dtype, .. }) =
                accumulation.primitive
            else {
                unreachable!("a dot or a sum")
            };

            let narrow = self.types[node.output].dtype;

            if accum_dtype.size_of() <= narrow.size_of() {
                continue;
            }

            // Follow same-dtype readers from the narrow output, breadth first, until a widening cast; another
            // reduction stops the search.
            let mut queue = std::collections::VecDeque::from([(node.output, Vec::new())]);
            let mut seen = vec![false; self.types.len()];

            'search: while let Some((v, path)) = queue.pop_front() {
                for &r in &readers[v] {
                    let (reader, wide) = (&self.nodes[r], self.types[self.nodes[r].output].dtype);
                    let mut path = path.clone();
                    path.push(self.label(reader));

                    match reader.primitive {
                        Cast { .. } if wide.size_of() > narrow.size_of() => {
                            let (name, fix) = match accumulation.primitive {
                                DotGeneral { .. } => (
                                    "dot_general",
                                    format!(
                                        "pass output_dtype={accum_dtype} (F.matmul(x, y, accum_dtype, output_dtype))"
                                    ),
                                ),
                                _ => (
                                    "reduce_sum",
                                    format!("cast its input to {accum_dtype} first"),
                                ),
                            };

                            let operands: Vec<String> = accumulation
                                .inputs
                                .iter()
                                .map(|&i| self.types[i].to_string())
                                .collect();

                            let message = format!(
                                "{name}({}) -> {} accumulates in {accum_dtype} but outputs {narrow}, then \
                                 {name} {sep} {} casts it back to {wide}: the rounding loses precision; {fix} \
                                 for a more accurate graph",
                                operands.join(", "),
                                self.types[node.output],
                                path.join(FUSION_SEPARATOR),
                                sep = FUSION_SEPARATOR.trim(),
                            );

                            warnings.push((node.output, reader.output, message));
                            break 'search;
                        }
                        DotGeneral { .. } | ReduceSum { .. } | ReduceMax { .. } | Fusion { .. } => {
                        }
                        _ if wide == narrow && !seen[reader.output] => {
                            seen[reader.output] = true;
                            queue.push_back((reader.output, path));
                        }
                        _ => {}
                    }
                }
            }
        }

        warnings
    }
}
