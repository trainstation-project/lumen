//! The graph: its values' types, inputs, nodes in topological order and outputs; building it node by node, reading it
//! back, and its display.

use std::fmt;

use super::{
    FUSION_SEPARATOR, Lines, Node, Primitive, Scope, TensorType, Var, intern, intern_lines,
};

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
    /// Whether nodes appended next compute gradients ([`Node::backward`]).
    pub(crate) backward: bool,
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

    /// Sets whether nodes appended from now on compute gradients ([`Node::backward`]).
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn set_backward(&mut self, backward: bool) {
        self.backward = backward;
    }

    /// Makes nodes appended from now on inherit `node`'s scope, source lines and whether it computes
    /// gradients.
    pub fn set_origin(&mut self, node: &Node) {
        self.scope = node.scope;
        self.sources = node.sources;
        self.backward = node.backward;
    }

    /// Makes nodes appended from now on inherit the first node's scope and whether it computes gradients,
    /// and the union of all their source lines.
    pub fn set_origins(&mut self, nodes: &[&Node]) {
        self.scope = nodes.first().map_or(&[], |n| n.scope);
        self.backward = nodes.first().is_some_and(|n| n.backward);
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
            backward: self.backward,
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
}

impl Graph {
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
