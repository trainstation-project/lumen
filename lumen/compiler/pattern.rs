use crate::graph::{Graph, Node, Primitive, Var};

/// A test of a value: what defines it, and what defines its operands.
pub(crate) enum Pattern {
    /// Any value, captured as `.0`; one captured before must be the same.
    Bind(usize),
    /// A value any of these patterns matches: the first that does.
    OneOf(Vec<Pattern>),
    /// A value a node defines whose primitive passes `test`, its operands
    /// matching `operands`, in order (or, if `either`, in either order: a
    /// commutative primitive's), its value captured as `bind` if given.
    Node {
        test: fn(&Primitive) -> bool,
        operands: Vec<Pattern>,
        either: bool,
        bind: Option<usize>,
    },
}

/// Any value, captured as `k`.
pub(crate) fn bind(k: usize) -> Pattern {
    Pattern::Bind(k)
}

/// A node whose primitive passes `test`, its operands matching `operands`
/// in order.
pub(crate) fn op<const N: usize>(test: fn(&Primitive) -> bool, operands: [Pattern; N]) -> Pattern {
    Pattern::Node {
        test,
        operands: operands.into(),
        either: false,
        bind: None,
    }
}

/// A value any of `patterns` matches (the first that does).
pub(crate) fn one_of<const N: usize>(patterns: [Pattern; N]) -> Pattern {
    Pattern::OneOf(patterns.into())
}

/// As [`op`], two operands matching in either order.
pub(crate) fn either(test: fn(&Primitive) -> bool, operands: [Pattern; 2]) -> Pattern {
    Pattern::Node {
        test,
        operands: operands.into(),
        either: true,
        bind: None,
    }
}

impl Pattern {
    /// This node pattern, its value captured as `k` too.
    pub(crate) fn bind(self, k: usize) -> Pattern {
        match self {
            Pattern::Node {
                test,
                operands,
                either,
                ..
            } => Pattern::Node {
                test,
                operands,
                either,
                bind: Some(k),
            },
            p => p,
        }
    }
}

/// A match: each capture's value, and the nodes matched (indices into the
/// graph's nodes), the root's first.
#[derive(Debug, Clone, Default)]
pub(crate) struct Match {
    pub captures: Vec<Option<Var>>,
    pub nodes: Vec<usize>,
}

impl Match {
    /// Capture `k`'s value.
    pub(crate) fn get(&self, k: usize) -> Var {
        self.captures[k].expect("a captured value")
    }
}

/// Matches patterns in `graph`.
pub(crate) struct Matcher<'a> {
    graph: &'a Graph,
    producer: Vec<Option<usize>>,
    /// The reads of each value: by live nodes (once per operand; dead ones,
    /// which earlier rewrites leave, read nothing) and as outputs.
    readers: Vec<usize>,
}

impl<'a> Matcher<'a> {
    pub(crate) fn new(graph: &'a Graph) -> Self {
        let mut producer = vec![None; graph.types.len()];
        let mut live = vec![false; graph.types.len()];
        graph.outputs().iter().for_each(|&v| live[v] = true);
        for (i, node) in graph.nodes().iter().enumerate().rev() {
            producer[node.output] = Some(i);
            if live[node.output] {
                node.inputs.iter().for_each(|&v| live[v] = true);
            }
        }
        let mut readers = vec![0; graph.types.len()];
        for node in graph.nodes().iter().filter(|n| live[n.output]) {
            node.inputs.iter().for_each(|&v| readers[v] += 1);
        }
        graph.outputs().iter().for_each(|&v| readers[v] += 1);
        Matcher {
            graph,
            producer,
            readers,
        }
    }

    /// The node defining `v`, if a node does (not an input).
    pub(crate) fn node(&self, v: Var) -> Option<&'a Node> {
        self.producer[v].map(|i| &self.graph.nodes()[i])
    }

    /// `pattern` matched at `v`, with `captures` slots.
    pub(crate) fn find(&self, pattern: &Pattern, v: Var, captures: usize) -> Option<Match> {
        let mut m = Match {
            captures: vec![None; captures],
            nodes: Vec::new(),
        };
        self.matches(pattern, v, &mut m).then_some(m)
    }

    fn matches(&self, pattern: &Pattern, v: Var, m: &mut Match) -> bool {
        match pattern {
            Pattern::OneOf(patterns) => patterns.iter().any(|p| {
                let saved = m.clone();
                let found = self.matches(p, v, m);
                if !found {
                    *m = saved;
                }
                found
            }),
            Pattern::Bind(k) => match m.captures[*k] {
                Some(bound) => bound == v,
                None => {
                    m.captures[*k] = Some(v);
                    true
                }
            },
            Pattern::Node {
                test,
                operands,
                either,
                bind,
            } => {
                let Some(i) = self.producer[v] else {
                    return false;
                };
                let node = &self.graph.nodes()[i];
                if !test(&node.primitive) || node.inputs.len() != operands.len() {
                    return false;
                }
                if let Some(k) = bind {
                    match m.captures[*k] {
                        Some(bound) if bound != v => return false,
                        _ => m.captures[*k] = Some(v),
                    }
                }
                m.nodes.push(i);
                let orders: &[[usize; 2]] = if *either {
                    &[[0, 1], [1, 0]]
                } else {
                    &[[0, 1]]
                };
                for order in orders {
                    let saved = m.clone();
                    let all = operands.iter().enumerate().all(|(k, p)| {
                        let operand = if operands.len() == 2 { order[k] } else { k };
                        self.matches(p, node.inputs[operand], m)
                    });
                    if all {
                        return true;
                    }
                    *m = saved;
                }
                false
            }
        }
    }

    /// Whether replacing match `m` by a computation of its root alone
    /// leaves nothing reading a value it drops: every matched node but the
    /// root (the first) is read only by matched nodes, and is no output.
    pub(crate) fn exclusive(&self, m: &Match) -> bool {
        let nodes = self.graph.nodes();
        let mut matched = m.nodes.clone();
        matched.sort_unstable();
        matched.dedup();
        matched.iter().filter(|&&i| i != m.nodes[0]).all(|&i| {
            let v = nodes[i].output;
            let inside: usize = matched
                .iter()
                .map(|&j| nodes[j].inputs.iter().filter(|&&u| u == v).count())
                .sum();
            inside == self.readers[v]
        })
    }

    /// The value of the scalar constant `v` (a `full` of shape `()`,
    /// perhaps broadcast), if it is one.
    pub(crate) fn scalar(&self, v: Var) -> Option<f64> {
        let mut node = self.node(v)?;
        if let Primitive::BroadcastInDim {
            broadcast_dimensions,
            ..
        } = &node.primitive
            && broadcast_dimensions.is_empty()
        {
            node = self.node(node.inputs[0])?;
        }
        match &node.primitive {
            Primitive::Full {
                shape, fill_value, ..
            } if shape.is_empty() => Some(fill_value.to_f64()),
            _ => None,
        }
    }
}
