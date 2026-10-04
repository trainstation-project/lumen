//! Graphs of [`Primitive`]s, the program representation of compiled
//! execution (JAX: a jaxpr). `lumen.compile` traces a Python function once
//! per input signature into a [`Graph`]; the graph is then run as a whole.
//! [`plan`] compiles a graph into a static [`Plan`] for execution;
//! [`reference`](crate::ops::reference), a CPU interpreter, defines what
//! each primitive computes.

pub mod plan;
pub mod primitive;
#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
pub(crate) mod tests;

use std::fmt;

pub use plan::{Plan, PlanOptions};
pub use primitive::{FUSION_SEPARATOR, Primitive};

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

    /// A warning for each dot or sum whose result is rounded to a narrower
    /// dtype than it accumulates in, then cast back up (through primitives
    /// keeping that dtype: a scale, a reshape): the rounding loses
    /// precision the program then computes with; outputting the wider
    /// dtype would not.
    pub fn precision_warnings(&self) -> Vec<String> {
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
                    path.push(reader.primitive.name());
                    match reader.primitive {
                        ConvertElementType { .. } if wide.size_of() > narrow.size_of() => {
                            let (name, fix) = match node.primitive {
                                DotGeneral { .. } => (
                                    "dot_general",
                                    format!("pass output_dtype={accum_dtype} (F.matmul(x, y, accum_dtype, output_dtype))"),
                                ),
                                _ => ("reduce_sum", format!("cast its input to {accum_dtype} first")),
                            };
                            let operands: Vec<String> =
                                node.inputs.iter().map(|&i| self.types[i].to_string()).collect();
                            warnings.push(format!(
                                "{name}({}) -> {} accumulates in {accum_dtype} but outputs {narrow}, then \
                                 {name} {sep} {} casts it back to {wide}: the rounding loses precision; {fix} \
                                 for a more accurate graph",
                                operands.join(", "),
                                self.types[node.output],
                                path.join(FUSION_SEPARATOR),
                                sep = FUSION_SEPARATOR.trim(),
                            ));
                        }
                        DotGeneral { .. } | ReduceSum { .. } | ReduceMax { .. } | Fusion { .. } => {}
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
