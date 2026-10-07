//! Graphs of [`Primitive`]s, the program representation of compiled
//! execution (JAX: a jaxpr). `lumen.compile` traces a Python function once
//! per input signature into a [`Graph`]; the graph is then run as a whole.
//! [`plan`] compiles a graph into a static [`Plan`] for execution;
//! [`reference`](crate::ops::reference), a CPU interpreter, defines what
//! each primitive computes.

pub(crate) mod custom;
pub mod plan;
pub mod primitive;
#[cfg(feature = "python")]
pub(crate) mod python;
mod remat;
mod schedule;
#[cfg(test)]
pub(crate) mod tests;

use std::collections::BTreeSet;
use std::fmt;
use std::sync::{Mutex, PoisonError};

pub use plan::{Plan, PlanOptions};
pub(crate) use plan::{Source, Stage, Staged};
pub use primitive::{FUSION_SEPARATOR, NEURAL_ENGINE, Primitive};

use crate::DType;

/// The static type of a graph value: a dtype and a shape (JAX:
/// `ShapedArray`). Graph values have no strides or device.
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

/// `f32[2,3]`, as in a jaxpr.
impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims: Vec<String> = self.shape.iter().map(usize::to_string).collect();
        write!(f, "{}[{}]", self.dtype, dims.join(","))
    }
}

/// A value in a graph: an index into its types.
pub type Var = usize;

/// `output = primitive(inputs...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub primitive: Primitive,
    pub inputs: Vec<Var>,
    pub output: Var,
    /// The `record_function` ranges open where it was traced, outermost
    /// first (a pass gives the nodes it makes of one its scope, as XLA's
    /// passes keep an instruction's metadata): its step's ranges when a
    /// plan runs it.
    pub scope: Scope,
    /// The lines of the traced program that computed it (each the first
    /// frame outside lumen), in order: the tracer's one; a pass's nodes,
    /// those of the node it rewrites, or of all it merges into one
    /// ([`Graph::set_origin`], [`Graph::set_origins`]).
    pub sources: Lines,
    /// Its label, if not its primitive's ([`Graph::label`]): what a pass
    /// made it for (a split-K dot's partials and their sum; a gated pair's
    /// merged weights' gradients, `2x dot_general`), as profiled. The
    /// passes after the one setting it keep it.
    pub label: Option<&'static str>,
}

/// A nesting of `record_function` ranges, outermost first ([`intern_scope`]).
pub type Scope = &'static [Range];

/// Lines of the traced program, each its file and line number
/// ([`intern_lines`]).
pub type Lines = &'static [(&'static str, u32)];

/// A `record_function` range a node was traced in: its name, and which call
/// of it (each time the traced code entered it; a backward range, the
/// forward call's). Steps of one call share its range when a plan runs
/// them, those of another call open their own (reordered steps too).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Range {
    pub name: &'static str,
    pub call: u64,
}

/// A function from typed inputs to outputs, in SSA form: each node
/// defines one new value from earlier ones. Built by [`input`](Self::input)
/// and [`apply`](Self::apply), which checks every node's types.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    pub(crate) types: Vec<TensorType>,
    inputs: Vec<Var>,
    nodes: Vec<Node>,
    outputs: Vec<Var>,
    /// The scope [`apply`](Self::apply) gives the nodes it adds.
    scope: Scope,
    /// The source lines it gives them.
    sources: Lines,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// A new input of type `ty`.
    pub fn input(&mut self, ty: TensorType) -> Var {
        self.types.push(ty);
        self.inputs.push(self.types.len() - 1);
        self.types.len() - 1
    }

    /// Give the nodes added from now on `scope` (a pass: the scope of the
    /// node it is rewriting).
    pub fn set_scope(&mut self, scope: Scope) {
        self.scope = scope;
    }

    /// Give the nodes added from now on `node`'s scope and source lines (a
    /// pass: the node it is rewriting's).
    pub fn set_origin(&mut self, node: &Node) {
        self.scope = node.scope;
        self.sources = node.sources;
    }

    /// Give the nodes added from now on the first of `nodes`' scope and
    /// all their source lines, each once (a pass: the nodes it merges into
    /// one, `merge_dots`' dots).
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

    /// The value `primitive(inputs...)`, or why the types do not allow it.
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

    /// Give the node defining `v` the source lines `sources` (the
    /// tracer's, known once it is applied).
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn set_sources(&mut self, v: Var, sources: Lines) {
        if let Some(node) = self.nodes.iter_mut().rev().find(|n| n.output == v) {
            node.sources = sources;
        }
    }

    /// Label the node defining `v` ([`Node::label`]; `None`: its
    /// primitive's).
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

    /// The graph without the nodes no output depends on (XLA's DCE: the
    /// tangents a linearized program computes and its transpose does not
    /// read), its inputs kept; and each value's new number, if kept.
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

    pub fn inputs(&self) -> &[Var] {
        &self.inputs
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }

    /// What `node` is profiled as: its primitive's name; a cast's, with
    /// its dtypes: `cast(float32 -> bfloat16)`.
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
            // A dot rounding its accumulator to a narrower output as it
            // writes it: that cast, its write-out's epilogue, named too.
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

    /// A warning for each dot or sum whose result is rounded to a narrower
    /// dtype than it accumulates in, then cast back up (through primitives
    /// keeping that dtype: a scale, a reshape): the rounding loses
    /// precision the program then computes with; outputting the wider
    /// dtype would not. Each: the node's output, the cast's, the message.
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
            // Each value reached, with the primitives from the node to it.
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

/// `label` as a `&'static str`, as profiled names are: each distinct label
/// is leaked once, however many graphs have it.
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

/// `lines` as [`Lines`], one shared copy per list.
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

/// `ranges` as a [`Scope`], one shared copy per nesting.
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

/// The graph as text, in the style of a jaxpr:
///
/// ```text
/// { lambda %0:f32[2,3]. let
///     %1:f32[2,3] = exp %0
///   in (%1) }
/// ```
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
