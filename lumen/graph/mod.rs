//! The traced dataflow graph: typed values, nodes applying primitives, scopes and source lines,
//! plus pruning, in-place placement views and precision warnings.

pub(crate) mod custom;
pub mod primitive;
#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
pub(crate) mod tests;

use std::collections::BTreeSet;
use std::fmt;
use std::sync::{Mutex, PoisonError};

pub use primitive::{BinaryOp, FUSION_SEPARATOR, NEURAL_ENGINE, Primitive, UPDATES, UnaryOp};

use crate::DType;
use crate::tensor::contiguous_strides;

/// The dtype and shape of a graph value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl TensorType {
    /// Builds a type from a dtype and a shape slice.
    pub fn new(dtype: DType, shape: &[usize]) -> Self {
        TensorType {
            dtype,
            shape: shape.to_vec(),
        }
    }

    /// Number of elements, the product of the shape.
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

/// Index of a value in a graph's type table.
pub type Var = usize;

/// One primitive application producing a single output value.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub primitive: Primitive,
    /// Values the primitive reads, in operand order.
    pub inputs: Vec<Var>,
    pub output: Var,
    /// Named ranges the node was traced inside, outermost first.
    pub scope: Scope,
    /// Python source lines the node came from.
    pub sources: Lines,
    /// Display name overriding the primitive's own, set for fused or renamed nodes.
    pub label: Option<&'static str>,
}

/// Interned stack of named ranges a node was traced in.
pub type Scope = &'static [Range];

/// Interned list of `(file, line)` source locations.
pub type Lines = &'static [(&'static str, u32)];

/// One entry of a scope: a range name and which call of it this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Range {
    pub name: &'static str,
    /// Distinguishes separate invocations of the same named range.
    pub call: u64,
}

/// A traced program: value types, inputs, nodes in topological order and outputs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    /// Type of every value, indexed by `Var`.
    pub(crate) types: Vec<TensorType>,
    pub(crate) inputs: Vec<Var>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) outputs: Vec<Var>,
    /// Scope stamped on nodes appended next.
    pub(crate) scope: Scope,
    /// Source lines stamped on nodes appended next.
    pub(crate) sources: Lines,
}

/// Where a value lives inside another value's buffer: element offset and strides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub offset: usize,
    pub strides: Vec<usize>,
}

impl Graph {
    /// Creates an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a graph input of type `ty` and returns its value.
    pub fn input(&mut self, ty: TensorType) -> Var {
        self.types.push(ty);
        self.inputs.push(self.types.len() - 1);
        self.types.len() - 1
    }

    /// Sets the scope stamped on nodes appended from now on.
    pub fn set_scope(&mut self, scope: Scope) {
        self.scope = scope;
    }

    /// Makes nodes appended from now on inherit `node`'s scope and source lines.
    pub fn set_origin(&mut self, node: &Node) {
        self.scope = node.scope;
        self.sources = node.sources;
    }

    /// Makes nodes appended from now on inherit the first node's scope and the union of all their source lines.
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

    /// Appends a node applying `primitive` to `inputs`, inferring and returning its output value.
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

    /// Overrides the source lines of the node that produces `v`.
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn set_sources(&mut self, v: Var, sources: Lines) {
        if let Some(node) = self.nodes.iter_mut().rev().find(|n| n.output == v) {
            node.sources = sources;
        }
    }

    /// Overrides the display label of the node that produces `v`.
    pub(crate) fn set_label(&mut self, v: Var, label: Option<&'static str>) {
        if let Some(node) = self.nodes.iter_mut().rev().find(|n| n.output == v) {
            node.label = label;
        }
    }

    /// Marks `outputs` as the graph's results.
    pub fn set_outputs(&mut self, outputs: &[Var]) -> Result<(), String> {
        self.check_vars(outputs)?;
        self.outputs = outputs.to_vec();
        Ok(())
    }

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

    /// Errors if any of `vars` is not a value of this graph.
    fn check_vars(&self, vars: &[Var]) -> Result<(), String> {
        match vars.iter().find(|&&v| v >= self.types.len()) {
            Some(v) => Err(format!("%{v} is not a value of this graph")),
            None => Ok(()),
        }
    }

    /// Type of value `var`.
    pub fn type_of(&self, var: Var) -> &TensorType {
        &self.types[var]
    }

    /// View of the `k`-th operand of concatenate `node` inside its output buffer.
    /// Panics if `node` is not a concatenate.
    pub fn slot(&self, node: &Node, k: usize) -> View {
        let Primitive::Concatenate { dimension, .. } = node.primitive else {
            panic!("{}: not a concatenate", node.primitive)
        };
        let strides = contiguous_strides(&self.types[node.output].shape);
        // The slot starts after the earlier operands' extents along the concatenated dimension.
        let before: usize = node.inputs[..k]
            .iter()
            .map(|&v| self.types[v].shape[dimension])
            .sum();
        View {
            offset: before * strides[dimension],
            strides,
        }
    }

    /// For each value stored inside another's buffer (view slices, operands of in-place concatenates),
    /// the host value and the view; `None` for values with their own storage.
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

    /// The graph's input values.
    pub fn inputs(&self) -> &[Var] {
        &self.inputs
    }

    /// The graph's nodes in topological order.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// The graph's output values.
    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }

    /// Display name of `node`: its explicit label, else a name spelling out casts and mixed-precision dots.
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

    /// Finds accumulations (dots, sums) whose wide accumulator is rounded to a narrow output and then
    /// cast back up downstream; returns `(accumulation, cast, message)` for each.
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
            // Follow same-dtype readers from the narrow output until a widening cast; another reduction stops the search.
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

/// Returns a `'static` copy of `label`, leaking each distinct string once.
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

/// Returns a `'static` copy of `lines`, leaking each distinct list once.
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

/// Returns a `'static` copy of `ranges`, leaking each distinct scope once.
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
