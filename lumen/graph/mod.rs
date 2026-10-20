pub(crate) mod custom;
pub mod primitive;
#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
pub(crate) mod tests;

use std::collections::BTreeSet;
use std::fmt;
use std::sync::{Mutex, PoisonError};

pub use primitive::{BinaryOp, FUSION_SEPARATOR, NEURAL_ENGINE, Primitive, UnaryOp};

use crate::DType;
use crate::tensor::contiguous_strides;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl TensorType {
    pub fn new(dtype: DType, shape: &[usize]) -> Self {
        TensorType {
            dtype,
            shape: shape.to_vec(),
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims: Vec<String> = self.shape.iter().map(usize::to_string).collect();
        write!(f, "{}[{}]", self.dtype, dims.join(","))
    }
}

pub type Var = usize;

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub primitive: Primitive,
    pub inputs: Vec<Var>,
    pub output: Var,
    pub scope: Scope,
    pub sources: Lines,
    pub label: Option<&'static str>,
}

pub type Scope = &'static [Range];

pub type Lines = &'static [(&'static str, u32)];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Range {
    pub name: &'static str,
    pub call: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    pub(crate) types: Vec<TensorType>,
    pub(crate) inputs: Vec<Var>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) outputs: Vec<Var>,
    pub(crate) scope: Scope,
    pub(crate) sources: Lines,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub offset: usize,
    pub strides: Vec<usize>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn input(&mut self, ty: TensorType) -> Var {
        self.types.push(ty);
        self.inputs.push(self.types.len() - 1);
        self.types.len() - 1
    }

    pub fn set_scope(&mut self, scope: Scope) {
        self.scope = scope;
    }

    pub fn set_origin(&mut self, node: &Node) {
        self.scope = node.scope;
        self.sources = node.sources;
    }

    pub fn set_origins(&mut self, nodes: &[&Node]) {
        self.scope = nodes.first().map_or(&[], |n| n.scope);
        let mut lines = Vec::new();
        for &line in nodes.iter().flat_map(|n| n.sources) {
            if !lines.contains(&line) {
                lines.push(line);
            }
        }
        self.sources = intern_lines(&lines);
    }

    pub fn apply(&mut self, primitive: Primitive, inputs: &[Var]) -> Result<Var, String> {
        self.check_vars(inputs)?;
        let args: Vec<&TensorType> = inputs.iter().map(|&v| &self.types[v]).collect();
        let ty = primitive.infer(&args)?;
        self.types.push(ty);
        let output = self.types.len() - 1;
        self.nodes.push(Node {
            primitive,
            inputs: inputs.to_vec(),
            output,
            scope: self.scope,
            sources: self.sources,
            label: None,
        });
        Ok(output)
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn set_sources(&mut self, v: Var, sources: Lines) {
        if let Some(node) = self.nodes.iter_mut().rev().find(|n| n.output == v) {
            node.sources = sources;
        }
    }

    pub(crate) fn set_label(&mut self, v: Var, label: Option<&'static str>) {
        if let Some(node) = self.nodes.iter_mut().rev().find(|n| n.output == v) {
            node.label = label;
        }
    }

    pub fn set_outputs(&mut self, outputs: &[Var]) -> Result<(), String> {
        self.check_vars(outputs)?;
        self.outputs = outputs.to_vec();
        Ok(())
    }

    pub fn prune(&self) -> (Graph, Vec<Option<Var>>) {
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

    fn check_vars(&self, vars: &[Var]) -> Result<(), String> {
        match vars.iter().find(|&&v| v >= self.types.len()) {
            Some(v) => Err(format!("%{v} is not a value of this graph")),
            None => Ok(()),
        }
    }

    pub fn type_of(&self, var: Var) -> &TensorType {
        &self.types[var]
    }

    pub fn slot(&self, node: &Node, k: usize) -> View {
        let Primitive::Concatenate { dimension, .. } = node.primitive else {
            panic!("{}: not a concatenate", node.primitive)
        };
        let strides = contiguous_strides(&self.types[node.output].shape);
        let before: usize = node.inputs[..k]
            .iter()
            .map(|&v| self.types[v].shape[dimension])
            .sum();
        View {
            offset: before * strides[dimension],
            strides,
        }
    }

    pub fn placements(&self) -> Vec<Option<(Var, View)>> {
        let mut placed = vec![None; self.types.len()];
        for node in &self.nodes {
            match &node.primitive {
                Primitive::Slice {
                    start_indices,
                    view: true,
                    ..
                } => {
                    let x = node.inputs[0];
                    let strides = contiguous_strides(&self.types[x].shape);
                    let offset = start_indices.iter().zip(&strides).map(|(s, t)| s * t).sum();
                    placed[node.output] = Some((x, View { offset, strides }));
                }
                Primitive::Concatenate { in_place: true, .. } => {
                    for (k, &v) in node.inputs.iter().enumerate() {
                        placed[v] = Some((node.output, self.slot(node, k)));
                    }
                }
                _ => {}
            }
        }
        placed
    }

    pub fn inputs(&self) -> &[Var] {
        &self.inputs
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }

    pub fn label(&self, node: &Node) -> &'static str {
        if let Some(label) = node.label {
            return label;
        }
        match node.primitive {
            Primitive::Cast { new_dtype } => intern(format!(
                "cast({} -> {})",
                self.types[node.inputs[0]].dtype.full_name(),
                new_dtype.full_name()
            )),
            Primitive::DotGeneral {
                accum_dtype,
                output_dtype,
                ..
            } if accum_dtype != output_dtype => intern(format!(
                "dot_general{FUSION_SEPARATOR}cast({} -> {})",
                accum_dtype.full_name(),
                output_dtype.full_name()
            )),
            ref p => p.name(),
        }
    }

    pub fn precision_warnings(&self) -> Vec<(Var, Var, String)> {
        use Primitive::*;
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); self.types.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            for &v in &node.inputs {
                readers[v].push(i);
            }
        }
        let mut warnings = Vec::new();
        for node in &self.nodes {
            let (DotGeneral { accum_dtype, .. } | ReduceSum { accum_dtype, .. }) = node.primitive
            else {
                continue;
            };
            let narrow = self.types[node.output].dtype;
            if accum_dtype.size_of() <= narrow.size_of() {
                continue;
            }
            let mut stack = vec![(node.output, Vec::new())];
            let mut seen = vec![false; self.types.len()];
            while let Some((v, path)) = stack.pop() {
                for &r in &readers[v] {
                    let (reader, wide) = (&self.nodes[r], self.types[self.nodes[r].output].dtype);
                    let mut path = path.clone();
                    path.push(self.label(reader));
                    match reader.primitive {
                        Cast { .. } if wide.size_of() > narrow.size_of() => {
                            let (name, fix) = match node.primitive {
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
                            let operands: Vec<String> = node
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
                        }
                        DotGeneral { .. } | ReduceSum { .. } | ReduceMax { .. } | Fusion { .. } => {
                        }
                        _ if wide == narrow && !seen[reader.output] => {
                            seen[reader.output] = true;
                            stack.push((reader.output, path));
                        }
                        _ => {}
                    }
                }
            }
        }
        warnings
    }
}

pub(crate) fn intern(label: String) -> &'static str {
    static LABELS: Mutex<BTreeSet<&'static str>> = Mutex::new(BTreeSet::new());
    let mut labels = LABELS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(&interned) = labels.get(label.as_str()) {
        return interned;
    }
    let interned: &'static str = Box::leak(label.into_boxed_str());
    labels.insert(interned);
    interned
}

pub(crate) fn intern_lines(lines: &[(&'static str, u32)]) -> Lines {
    static LINES: Mutex<BTreeSet<Lines>> = Mutex::new(BTreeSet::new());
    let mut interned = LINES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(&lines) = interned.get(lines) {
        return lines;
    }
    let lines: Lines = Box::leak(lines.to_vec().into_boxed_slice());
    interned.insert(lines);
    lines
}

#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn intern_scope(ranges: &[Range]) -> Scope {
    static SCOPES: Mutex<BTreeSet<Scope>> = Mutex::new(BTreeSet::new());
    let mut scopes = SCOPES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(&interned) = scopes.get(ranges) {
        return interned;
    }
    let interned: Scope = Box::leak(ranges.to_vec().into_boxed_slice());
    scopes.insert(interned);
    interned
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let typed = |v: Var| format!("%{v}:{}", self.types[v]);
        let names = |vars: &[Var]| {
            let names: Vec<String> = vars.iter().map(|v| format!("%{v}")).collect();
            names.join(" ")
        };
        let inputs: Vec<String> = self.inputs.iter().map(|&v| typed(v)).collect();
        writeln!(f, "{{ lambda {}. let", inputs.join(" "))?;
        for node in &self.nodes {
            write!(f, "    {} = {}", typed(node.output), node.primitive)?;
            if !node.inputs.is_empty() {
                write!(f, " {}", names(&node.inputs))?;
            }
            writeln!(f)?;
        }
        let outputs: Vec<String> = self.outputs.iter().map(|v| format!("%{v}")).collect();
        write!(f, "  in ({}) }}", outputs.join(", "))
    }
}
