use super::MpsModule;
use super::fusion::fusible;
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::common::diamonds::Finder;
pub(super) use crate::graph_compiler::common::diamonds::Row;
use crate::graph_compiler::pass::CompilerPass;
use crate::kernels::reduce::mps::REDUCE_THREADS;

const MAX_GROUPS: usize = 64;

const MIN_GROUPS: usize = 16;

const MAX_COLUMNS: usize = 16;

pub(super) fn diamonds(graph: &Graph) -> Vec<Row> {
    crate::graph_compiler::common::diamonds::diamonds(graph, fusible)
}

fn partials(graph: &Graph, mut rows: Vec<Row>) -> (Graph, Vec<Row>) {
    let finder = Finder::new(graph, fusible);
    let nodes = graph.nodes();
    let mut row_of = vec![None; nodes.len()];
    for (r, row) in rows.iter().enumerate() {
        for &k in std::iter::once(&row.root)
            .chain(&row.inner)
            .chain(&row.reductions)
        {
            row_of[k] = Some(r);
        }
    }
    let depends = |v: Var, on: &dyn Fn(usize) -> bool, floor: usize| {
        let (mut seen, mut stack) = (vec![false; nodes.len()], vec![v]);
        while let Some(v) = stack.pop() {
            let Some(k) = finder.producer[v].filter(|&k| k >= floor && !seen[k]) else {
                continue;
            };
            if on(k) {
                return true;
            }
            seen[k] = true;
            stack.extend(&nodes[k].inputs);
        }
        false
    };
    let mut attach: Vec<Option<(usize, usize)>> = vec![None; nodes.len()];
    for (c, node) in nodes.iter().enumerate() {
        let Primitive::ReduceSum { axes, .. } = &node.primitive else {
            continue;
        };
        if !finder.live[c] || !fusible(graph, node) {
            continue;
        }
        let x = node.inputs[0];
        let ty = graph.type_of(x);
        let rank = ty.shape.len();
        let n = ty.shape.last().copied().unwrap_or(1);
        let over_rows = rank >= 2 && axes.iter().copied().eq(0..rank - 1);
        if !over_rows || n <= 1 || n.div_ceil(REDUCE_THREADS) > MAX_COLUMNS {
            continue;
        }
        let count = ty.numel() / n;
        let Some(groups) = (1..=MAX_GROUPS.min(count))
            .rev()
            .find(|&g| count.is_multiple_of(g))
            .filter(|&g| g >= MIN_GROUPS.min(count))
        else {
            continue;
        };
        let (mut reads, mut stack) = (Vec::new(), vec![x]);
        while let Some(v) = stack.pop() {
            match finder.producer[v] {
                Some(k)
                    if row_of[k].is_none()
                        && fusible(graph, &nodes[k])
                        && !matches!(
                            nodes[k].primitive,
                            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
                        ) =>
                {
                    stack.extend(&nodes[k].inputs)
                }
                _ if !reads.contains(&v) => reads.push(v),
                _ => {}
            }
        }
        attach[c] = rows
            .iter()
            .enumerate()
            .find(|&(r, row)| {
                let in_row = |k: usize| row_of[k] == Some(r);
                let root = graph.type_of(nodes[row.root].output);
                let first = std::iter::once(row.root)
                    .chain(row.inner.iter().copied())
                    .chain(row.reductions.iter().copied())
                    .min()
                    .unwrap_or(row.root);
                let shares = reads.iter().any(|&v| {
                    finder.producer[v].is_some_and(in_row)
                        || finder.readers[v].iter().any(|&u| in_row(u))
                });
                let before = reads.iter().all(|&v| match finder.producer[v] {
                    Some(k) if in_row(k) => k != row.root,
                    _ => !depends(v, &in_row, first),
                });
                let after = std::iter::once(&row.root)
                    .chain(&row.inner)
                    .chain(&row.reductions)
                    .all(|&k| !nodes[k].inputs.iter().any(|&v| depends(v, &|k| k == c, c)));
                root.numel() == ty.numel()
                    && root.shape.last() == Some(&n)
                    && shares
                    && before
                    && after
            })
            .map(|(r, _)| (r, groups));
    }
    if attach.iter().all(Option::is_none) {
        return (graph.clone(), rows);
    }
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    let (mut index, mut added) = (vec![0; nodes.len()], Vec::new());
    for (i, node) in nodes.iter().enumerate() {
        out.set_origin(node);
        let mut inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let mut primitive = node.primitive.clone();
        if let Some((r, groups)) = attach[i] {
            let Primitive::ReduceSum { accum_dtype, .. } = node.primitive else {
                unreachable!("a sum")
            };
            let ty = out.type_of(inputs[0]);
            let n = *ty.shape.last().expect("a dimension");
            let new_sizes = vec![groups, ty.numel() / n / groups, n];
            let blocks = out.apply(Primitive::Reshape { new_sizes }, &inputs);
            let sum = |axes| Primitive::ReduceSum { axes, accum_dtype };
            let blocks = blocks.expect("a reshape of the rows");
            inputs = vec![out.apply(sum(vec![1]), &[blocks]).expect("their sums")];
            let at = out.nodes().len();
            added.push((r, at - 2, at - 1));
            primitive = sum(vec![0]);
        }
        index[i] = out.nodes().len();
        map[node.output] = out
            .apply(primitive, &inputs)
            .expect("a graph is typed as the original");
        out.set_label(map[node.output], node.label);
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs)
        .expect("outputs are values of the graph");
    for row in &mut rows {
        row.root = index[row.root];
        for k in row
            .inner
            .iter_mut()
            .chain(&mut row.reductions)
            .chain(&mut row.outputs)
        {
            *k = index[*k];
        }
    }
    for (r, reshape, partial) in added {
        rows[r].inner.push(reshape);
        rows[r].inner.sort_unstable();
        rows[r].partials.push(partial);
    }
    (out, rows)
}

pub(super) struct Diamonds;

impl<'a> CompilerPass<MpsModule<'a>> for Diamonds {
    fn name(&self) -> &'static str {
        "normalization_diamonds"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        (m.graph, m.rows) = partials(&m.graph, diamonds(&m.graph));
        Ok(())
    }
}
