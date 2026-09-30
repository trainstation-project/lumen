//! Graphs of [`Primitive`]s, the program representation of compiled
//! execution (JAX: a jaxpr). `lumen.compile` traces a Python function once
//! per input signature into a [`Graph`]; the graph is then run as a whole.
//! [`plan`] compiles a graph into a static [`Plan`] for execution;
//! [`reference`], a CPU interpreter, defines what each primitive computes.

#[cfg(lumen_mps_linked)]
mod mps;
pub mod plan;
pub mod primitive;
#[cfg(feature = "python")]
pub(crate) mod python;
pub mod reference;
#[cfg(test)]
mod tests;

use std::fmt;

pub use plan::Plan;
pub use primitive::Primitive;

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
#[derive(Debug, Clone)]
pub struct Node {
    pub primitive: Primitive,
    pub inputs: Vec<Var>,
    pub output: Var,
}

/// A function from typed inputs to outputs, in SSA form: each node
/// defines one new value from earlier ones. Built by [`input`](Self::input)
/// and [`apply`](Self::apply), which checks every node's types.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    types: Vec<TensorType>,
    inputs: Vec<Var>,
    nodes: Vec<Node>,
    outputs: Vec<Var>,
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
        });
        Ok(output)
    }

    pub fn set_outputs(&mut self, outputs: &[Var]) -> Result<(), String> {
        self.check_vars(outputs)?;
        self.outputs = outputs.to_vec();
        Ok(())
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
