//! Metal emitter: Metal statements computing a fused body's values at a given index.

use std::collections::HashMap;
use std::fmt::Write;

use super::Emit;
use crate::codegen::mps::*;
use crate::codegen::random::RandomCodegen;
use crate::graph::{Graph, Node, Primitive, UnaryOp, Var};
use crate::tensor::contiguous_strides;

/// Builds Metal locals for body values, memoizing each `(value, index)` pair.
pub(crate) struct Emitter<'a> {
    pub(crate) body: &'a Graph,
    /// Node index producing each non-input value.
    pub(crate) producer: HashMap<Var, usize>,
    /// Emitted statements, in order.
    pub(crate) lines: String,
    /// Local already holding each value at each index expression.
    pub(crate) values: HashMap<(Var, String), String>,
    /// Local already holding each index expression.
    pub(crate) indices: HashMap<String, String>,
    /// Counter for fresh local names.
    pub(crate) locals: usize,
    /// Inputs passed as constant scalars rather than buffers.
    pub(crate) by_value: &'a [bool],
    /// Values that are constant across the inner index (per row), computed once into `hoisted`.
    pub(crate) invariant: Vec<bool>,
    /// Locals holding invariant values, or values supplied by the enclosing kernel.
    pub(crate) row_locals: HashMap<Var, String>,
    /// Statements computing invariant values, emitted before the inner loop.
    pub(crate) hoisted: String,
    /// Whether any tensor exceeds `u32` elements, so indices need `ulong`.
    pub(crate) wide: bool,
}

impl<'a> Emitter<'a> {
    /// Returns a new unique local name starting with `prefix`.
    pub(crate) fn fresh(&mut self, prefix: &str) -> String {
        self.locals += 1;
        format!("{prefix}{}", self.locals - 1)
    }

    /// Binds an index expression to a local (reused if already bound); plain `j` and literals
    /// are returned as is.
    pub(crate) fn index(&mut self, expr: String) -> String {
        if expr == "j" || expr.parse::<u64>().is_ok() {
            return expr;
        }

        if let Some(name) = self.indices.get(&expr) {
            return name.clone();
        }

        let name = self.fresh("i");
        let ty = if self.wide { "ulong" } else { "uint" };
        writeln!(self.lines, "        {ty} {name} = {expr};").unwrap();
        self.indices.insert(expr, name.clone());
        name
    }

    /// Index into a layout node's operand for output index `idx`.
    pub(crate) fn operand_index(&mut self, node: &Node, idx: String) -> String {
        use Primitive::*;
        let body = self.body;
        let (x, ty) = (body.type_of(node.inputs[0]), body.type_of(node.output));
        let x_strides = contiguous_strides(&x.shape);

        match &node.primitive {
            Reshape { .. } => idx,
            BroadcastInDim {
                broadcast_dimensions,
                ..
            } => {
                let mut strides = vec![0; ty.shape.len()];

                for (k, &d) in broadcast_dimensions.iter().enumerate() {
                    if x.shape[k] != 1 {
                        strides[d] = x_strides[k];
                    }
                }

                self.index(gather_index(&idx, &ty.shape, &strides))
            }
            Transpose { permutation } => {
                let strides: Vec<usize> = permutation.iter().map(|&d| x_strides[d]).collect();
                self.index(gather_index(&idx, &ty.shape, &strides))
            }
            Slice { start_indices, .. } => {
                let start: usize = start_indices
                    .iter()
                    .zip(&x_strides)
                    .map(|(s, st)| s * st)
                    .sum();

                let i = gather_index(&idx, &ty.shape, &x_strides);
                self.index(format!("{start} + {i}"))
            }
            p => unreachable!("{p} is not a layout primitive"),
        }
    }

    /// Emits the statements computing `v` at flat index `idx` and returns the local holding it.
    pub(crate) fn element(&mut self, v: Var, idx: String) -> String {
        let key = (v, idx.clone());

        if let Some(name) = self.values.get(&key) {
            return name.clone();
        }

        let body = self.body;
        let ty = body.type_of(v);
        let t = metal_type(ty.dtype);

        let expr = match self.producer.get(&v) {
            None => {
                let k = body.inputs().iter().position(|&i| i == v).unwrap();

                match self.by_value.get(k) == Some(&true) {
                    true => format!("in{k}"),
                    false => format!("{}{idx})", load(k)),
                }
            }
            Some(&i) => {
                let node = &body.nodes()[i];
                use Primitive::*;

                match &node.primitive {
                    Add | Sub | Mul | Div | Max | Eq | Lt => {
                        let (x, y) = (
                            self.value(node.inputs[0], idx.clone()),
                            self.value(node.inputs[1], idx),
                        );

                        let op = functor(&node.primitive);
                        format!("{op}::apply({x}, {y})")
                    }
                    Neg if ty.dtype.is_float() => format!("-{}", self.value(node.inputs[0], idx)),
                    Neg => format!("{t}(-{})", self.value(node.inputs[0], idx)),
                    Exp | Log | Sqrt | Tanh | Logistic => {
                        let x = self.value(node.inputs[0], idx);
                        let op = functor(&node.primitive);
                        format!("{op}::apply({x})")
                    }
                    Cast { .. } => {
                        format!("convert_value<{t}>({})", self.value(node.inputs[0], idx))
                    }
                    Unary(op) => {
                        let x = self.value(node.inputs[0], idx);

                        match op {
                            UnaryOp::ReducePrecision {
                                exponent_bits,
                                mantissa_bits,
                            } => format!(
                                "reduce_precision_op({x}, {exponent_bits}u, {mantissa_bits}u)"
                            ),
                            op => format!("{}_op({x})", op.name()),
                        }
                    }
                    Binary(op) => {
                        let x = self.value(node.inputs[0], idx.clone());
                        let y = self.value(node.inputs[1], idx);
                        format!("{}_op({x}, {y})", op.name())
                    }
                    Select => {
                        let p = self.value(node.inputs[0], idx.clone());
                        let x = self.value(node.inputs[1], idx.clone());
                        let y = self.value(node.inputs[2], idx);
                        format!("{p} ? {x} : {y}")
                    }
                    Reshape { .. } | BroadcastInDim { .. } | Transpose { .. } | Slice { .. } => {
                        if self.invariant[node.inputs[0]] {
                            return self.value(node.inputs[0], idx);
                        }

                        let i = self.operand_index(node, idx);
                        return self.value(node.inputs[0], i);
                    }
                    Gather { axis } => {
                        // Split `idx` into outer, gathered and inner parts; the index operand broadcasts when it has
                        // fewer rows than the output.
                        let x = body.type_of(node.inputs[0]);
                        let m = body.type_of(node.inputs[1]).numel();
                        let n = x.shape[*axis];
                        let inner: usize = x.shape[axis + 1..].iter().product();
                        let mut t = idx.clone();

                        if inner != 1 {
                            t = format!("{t} / {inner}u");
                        }

                        if m != ty.numel() / inner.max(1) {
                            t = format!("({t}) % {m}u");
                        }

                        let t = self.index(t);
                        let k = self.value(node.inputs[1], t);

                        let (int, u) = if self.wide {
                            ("ulong", "ul")
                        } else {
                            ("uint", "u")
                        };

                        let k = self.index(format!("{int}({k})"));
                        let mut terms = Vec::new();

                        if ty.numel() > m * inner {
                            terms.push(format!("({idx} / {}{u}) * {}{u}", m * inner, n * inner));
                        }

                        terms.push(match inner {
                            1 => k,
                            _ => format!("{k} * {inner}{u} + {idx} % {inner}{u}"),
                        });

                        let i = self.index(terms.join(" + "));
                        return self.value(node.inputs[0], i);
                    }
                    Concatenate { dimension, .. } => {
                        let d = *dimension;
                        let span = contiguous_strides(&ty.shape)[d];
                        // Coordinate along the concatenated dim, then one branch per operand with the index rebased
                        // into that operand.
                        let mut c = idx.clone();

                        if span != 1 {
                            c = format!("{c} / {span}u");
                        }

                        let outer = ty.shape[..d].iter().product::<usize>();

                        if outer != 1 {
                            c = format!("({c}) % {}u", ty.shape[d]);
                        }

                        let c = self.index(c);
                        let name = self.fresh("v");
                        writeln!(self.lines, "        {t} {name};").unwrap();
                        self.values.insert(key, name.clone());
                        let mut start = 0;

                        for (k, &x) in node.inputs.iter().enumerate() {
                            let n = body.type_of(x).shape[d];

                            let branch = match k {
                                0 => format!("if ({c} < {}u)", start + n),
                                _ if k + 1 == node.inputs.len() => "else".into(),
                                _ => format!("else if ({c} < {}u)", start + n),
                            };

                            writeln!(self.lines, "        {branch} {{").unwrap();
                            // Each branch has its own scope, so locals cached inside it can't be reused after it.
                            let saved = (self.values.clone(), self.indices.clone());
                            let mut terms = Vec::new();

                            if outer != 1 {
                                terms.push(format!(
                                    "({idx} / {}u) * {}u",
                                    span * ty.shape[d],
                                    span * n
                                ));
                            }

                            terms.push(match span {
                                1 => format!("{c} - {start}u"),
                                _ => format!("({c} - {start}u) * {span}u"),
                            });

                            if span != 1 {
                                terms.push(format!("{idx} % {span}u"));
                            }

                            let i = self.index(terms.join(" + "));
                            let v = self.value(x, i);
                            writeln!(self.lines, "{name} = {v};").unwrap();
                            writeln!(self.lines, "}}").unwrap();
                            (self.values, self.indices) = saved;
                            start += n;
                        }

                        return name;
                    }
                    Full { fill_value, .. } => constant(ty.dtype, *fill_value),
                    RandomBits { offset, .. } => {
                        let seed = self.value(node.inputs[0], "0".into());
                        let start = self.value(node.inputs[1], "0".into());
                        Mps::bits(&seed, &start, *offset, &idx)
                    }
                    Iota { dimension, .. } => {
                        let mut strides = vec![0; ty.shape.len()];
                        strides[*dimension] = 1;
                        let i = self.index(gather_index(&idx, &ty.shape, &strides));
                        format!("from_int<{t}>({i})")
                    }
                    p => unreachable!("{p} is not fusible"),
                }
            }
        };

        let name = self.fresh("v");
        writeln!(self.lines, "        {t} {name} = {expr};").unwrap();
        self.values.insert(key, name.clone());
        name
    }
}

/// Metal emitter: Metal statements a fused body's values are computed in, invariant ones hoisted.
impl<'a> Emit<'a> for Emitter<'a> {
    /// Starts an emitter over `body` with nothing emitted.
    fn new(body: &'a Graph, by_value: &'a [bool]) -> Self {
        Emitter {
            body,
            by_value,
            producer: body
                .nodes()
                .iter()
                .enumerate()
                .map(|(i, n)| (n.output, i))
                .collect(),
            lines: String::new(),
            values: HashMap::new(),
            indices: HashMap::new(),
            locals: 0,
            wide: body.types.iter().any(|t| t.numel() > u32::MAX as usize),
            invariant: vec![false; body.types.len()],
            row_locals: HashMap::new(),
            hoisted: String::new(),
        }
    }

    /// Local holding `v` at `idx`; invariant values are computed once at `row` (or `0` for
    /// scalars) into the hoisted prelude.
    fn value(&mut self, v: Var, idx: String) -> String {
        if !self.invariant[v] {
            return self.element(v, idx);
        }

        if let Some(name) = self.row_locals.get(&v) {
            return name.clone();
        }

        // Emit the invariant value into a fresh buffer with empty caches, then restore the loop's.
        let lines = std::mem::take(&mut self.lines);

        let caches = (
            std::mem::take(&mut self.values),
            std::mem::take(&mut self.indices),
        );

        let idx = match self.body.type_of(v).numel() {
            1 => "0",
            _ => "row",
        };

        let name = self.element(v, idx.into());
        let hoisted = std::mem::replace(&mut self.lines, lines);
        self.hoisted.push_str(&hoisted);
        (self.values, self.indices) = caches;
        self.row_locals.insert(v, name.clone());
        name
    }
}
