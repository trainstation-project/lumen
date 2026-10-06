//! A graph lowered to a Core ML ML Program (`coremltools`'
//! `mlmodel/format/MIL.proto`, opset CoreML7), its specification
//! (`Model.proto`) written as protobuf ([`super::proto`]): each value
//! computed in its dtype (float16, float32, int32, bool).

use super::proto::Message;
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::{DType, Scalar};

/// A shape as Core ML's arrays take it: a scalar as [1].
pub(super) fn io_shape(shape: &[usize]) -> Vec<usize> {
    match shape.is_empty() {
        true => vec![1],
        false => shape.to_vec(),
    }
}

/// A dtype's MIL `DataType` and its name in a `cast`.
pub(super) fn mil_dtype(dtype: DType) -> Result<(i64, &'static str), String> {
    Ok(match dtype {
        DType::F16 => (10, "fp16"),
        DType::F32 => (11, "fp32"),
        DType::I32 => (23, "int32"),
        DType::Bool => (1, "bool"),
        d => {
            return Err(format!(
                "Core ML programs compute float16, float32, int32 and bool values, not {d}"
            ));
        }
    })
}

/// A dtype's Core ML array type (`ArrayFeatureType.ArrayDataType`): the
/// program's inputs and outputs are float16, float32 or int32.
pub(super) fn array_dtype(dtype: DType) -> Result<i64, String> {
    match dtype {
        DType::F16 => Ok(65552),
        DType::F32 => Ok(65568),
        DType::I32 => Ok(131104),
        d => Err(format!(
            "Core ML programs take and return float16, float32 and int32 arrays, not {d}"
        )),
    }
}

/// A MIL `ValueType`: a tensor of `dtype` and `shape`.
fn tensor_type(dtype: i64, shape: &[usize]) -> Message {
    let mut t = Message::new().int(1, dtype).int(2, shape.len() as i64);
    for &d in shape {
        t = t.message(
            3,
            Message::new().message(1, Message::new().int(1, d as i64)),
        );
    }
    Message::new().message(1, t)
}

/// A MIL `Value`: a tensor of `dtype` and `shape`, its elements `tensor`
/// (a `TensorValue`).
fn value(dtype: i64, shape: &[usize], tensor: Message) -> Message {
    Message::new()
        .message(2, tensor_type(dtype, shape))
        .message(3, Message::new().message(1, tensor))
}

/// A MIL string value.
fn string_value(s: &str) -> Message {
    let strings = Message::new().string(1, s);
    value(2, &[], Message::new().message(4, strings))
}

/// The elements `data` (host bytes) of a tensor of `dtype`, as a MIL
/// `TensorValue`.
fn tensor_value(dtype: DType, data: &[u8]) -> Message {
    let words = |n: usize| data.chunks_exact(n);
    match dtype {
        DType::F16 => Message::new().message(7, Message::new().bytes(1, data)),
        DType::F32 => {
            let floats = words(4).map(|w| f32::from_le_bytes(w.try_into().expect("4 bytes")));
            Message::new().message(1, Message::new().packed_floats(1, floats))
        }
        DType::I32 => {
            let ints =
                words(4).map(|w| i64::from(i32::from_le_bytes(w.try_into().expect("4 bytes"))));
            Message::new().message(2, Message::new().packed_ints(1, ints))
        }
        _ => Message::new().message(
            3,
            Message::new().packed_ints(1, data.iter().map(|&b| i64::from(b != 0))),
        ),
    }
}

/// `value` of `dtype`, as its bytes.
fn scalar_bytes(dtype: DType, value: Scalar) -> Vec<u8> {
    let f = match value {
        Scalar::Bool(b) => f64::from(u8::from(b)),
        Scalar::Int(i) => i as f64,
        Scalar::Float(f) => f,
    };
    match dtype {
        DType::F16 => half::f16::from_f64(f).to_le_bytes().to_vec(),
        DType::F32 => (f as f32).to_le_bytes().to_vec(),
        DType::I32 => match value {
            Scalar::Int(i) => (i as i32).to_le_bytes().to_vec(),
            _ => (f as i32).to_le_bytes().to_vec(),
        },
        _ => vec![u8::from(f != 0.0)],
    }
}

/// The graph being lowered: its MIL operations, and each value's MIL name
/// and shape: its type's, or (a broadcast, a constant: compact) one of
/// size-1 dimensions where it is the same along them, which elementwise
/// operations broadcast (numpy's rule, as MIL's) and others tile first.
/// Each operation's output, with the node it lowers (`None`: an input's or
/// an output's).
pub(super) type Owners = Vec<(String, Option<usize>)>;

struct Lower<'a> {
    graph: &'a Graph,
    ops: Vec<Message>,
    vals: Vec<Option<(String, Vec<usize>)>>,
    names: usize,
    /// The node being lowered, and each operation's.
    node: Option<usize>,
    owners: Owners,
}

impl Lower<'_> {
    fn fresh(&mut self) -> String {
        self.names += 1;
        format!("t{}", self.names)
    }

    /// An operation `kind` of `inputs` (each a parameter and its
    /// arguments' names), its output `dtype` and `shape`: its output's
    /// name.
    fn op(
        &mut self,
        kind: &str,
        inputs: &[(&str, Vec<String>)],
        dtype: DType,
        shape: &[usize],
    ) -> Result<String, String> {
        let name = self.fresh();
        self.named(name, kind, inputs, dtype, shape)
    }

    /// [`op`](Self::op), its output named `name`.
    fn named(
        &mut self,
        name: String,
        kind: &str,
        inputs: &[(&str, Vec<String>)],
        dtype: DType,
        shape: &[usize],
    ) -> Result<String, String> {
        let mut op = Message::new().string(1, kind);
        for (parameter, arguments) in inputs {
            let mut argument = Message::new();
            for a in arguments {
                argument = argument.message(1, Message::new().string(1, a));
            }
            op = op.entry(2, parameter, argument);
        }
        let output = Message::new()
            .string(1, &name)
            .message(2, tensor_type(mil_dtype(dtype)?.0, shape));
        op = op.message(3, output).entry(5, "name", string_value(&name));
        self.owners.push((name.clone(), self.node));
        self.ops.push(op);
        Ok(name)
    }

    /// A constant of `dtype` and `shape`, its elements' bytes `data`.
    fn constant(&mut self, dtype: DType, shape: &[usize], data: &[u8]) -> Result<String, String> {
        let name = self.fresh();
        let code = mil_dtype(dtype)?.0;
        let op = Message::new()
            .string(1, "const")
            .message(
                3,
                Message::new()
                    .string(1, &name)
                    .message(2, tensor_type(code, shape)),
            )
            .entry(5, "val", value(code, shape, tensor_value(dtype, data)))
            .entry(5, "name", string_value(&name));
        self.ops.push(op);
        Ok(name)
    }

    /// An int32 vector constant (a parameter: axes, a shape).
    fn ints(&mut self, values: &[usize]) -> Result<String, String> {
        let data: Vec<u8> = values
            .iter()
            .flat_map(|&v| (v as i32).to_le_bytes())
            .collect();
        self.constant(DType::I32, &[values.len()], &data)
    }

    fn int(&mut self, value: i64) -> Result<String, String> {
        self.constant(DType::I32, &[], &(value as i32).to_le_bytes())
    }

    fn boolean(&mut self, value: bool) -> Result<String, String> {
        self.constant(DType::Bool, &[], &[u8::from(value)])
    }

    /// Value `v` at its type's shape: tiled along its compact dimensions.
    fn full(&mut self, v: Var) -> Result<String, String> {
        let ty = self.graph.type_of(v).clone();
        let (name, shape) = self.vals[v].clone().ok_or("a value before its producer")?;
        if shape == ty.shape {
            return Ok(name);
        }
        let reps: Vec<usize> = ty.shape.iter().zip(&shape).map(|(&n, &s)| n / s).collect();
        let reps = self.ints(&reps)?;
        let tiled = self.op(
            "tile",
            &[("x", vec![name]), ("reps", vec![reps])],
            ty.dtype,
            &ty.shape,
        )?;
        self.vals[v] = Some((tiled.clone(), ty.shape.clone()));
        Ok(tiled)
    }

    /// Reshape `name` (`dtype`) to `shape`.
    fn reshape(&mut self, name: String, dtype: DType, shape: &[usize]) -> Result<String, String> {
        let s = self.ints(shape)?;
        self.op(
            "reshape",
            &[("x", vec![name]), ("shape", vec![s])],
            dtype,
            shape,
        )
    }

    /// Transpose `name` (`dtype`, of `shape`) by `perm`, if it moves any
    /// dimension.
    fn transpose(
        &mut self,
        name: String,
        dtype: DType,
        shape: &[usize],
        perm: &[usize],
    ) -> Result<(String, Vec<usize>), String> {
        let out: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        if perm.iter().enumerate().all(|(i, &p)| i == p) {
            return Ok((name, out));
        }
        let p = self.ints(perm)?;
        let name = self.op(
            "transpose",
            &[("x", vec![name]), ("perm", vec![p])],
            dtype,
            &out,
        )?;
        Ok((name, out))
    }

    fn lower_node(&mut self, node: &crate::graph::Node) -> Result<(String, Vec<usize>), String> {
        use Primitive::*;
        let graph = self.graph;
        let ty = graph.type_of(node.output).clone();
        let input = |k: usize| node.inputs[k];
        let elementwise = |this: &mut Self,
                           kind: &str,
                           params: &[&str]|
         -> Result<(String, Vec<usize>), String> {
            let mut args = Vec::new();
            let mut shape = vec![1; ty.shape.len()];
            for (k, &p) in params.iter().enumerate() {
                let (name, s) = this.vals[input(k)]
                    .clone()
                    .ok_or("a value before its producer")?;
                shape = shape.iter().zip(&s).map(|(&a, &b)| a.max(b)).collect();
                args.push((p, vec![name]));
            }
            let name = this.op(kind, &args, ty.dtype, &shape)?;
            Ok((name, shape))
        };
        let float = ty.dtype.is_float();
        Ok(match &node.primitive {
            Add => elementwise(self, "add", &["x", "y"])?,
            Sub => elementwise(self, "sub", &["x", "y"])?,
            Mul => elementwise(self, "mul", &["x", "y"])?,
            Div if float => elementwise(self, "real_div", &["x", "y"])?,
            Max => elementwise(self, "maximum", &["x", "y"])?,
            Eq => elementwise(self, "equal", &["x", "y"])?,
            Lt => elementwise(self, "less", &["x", "y"])?,
            Exp => elementwise(self, "exp", &["x"])?,
            Log => elementwise(self, "log", &["x"])?,
            Sqrt => elementwise(self, "sqrt", &["x"])?,
            Tanh => elementwise(self, "tanh", &["x"])?,
            Logistic => elementwise(self, "sigmoid", &["x"])?,
            Select => elementwise(self, "select", &["cond", "a", "b"])?,
            Neg => {
                // x * -1: exact, -0 kept.
                let (x, shape) = self.vals[input(0)]
                    .clone()
                    .ok_or("a value before its producer")?;
                let minus =
                    self.constant(ty.dtype, &[], &scalar_bytes(ty.dtype, Scalar::Int(-1)))?;
                let name = self.op(
                    "mul",
                    &[("x", vec![x]), ("y", vec![minus])],
                    ty.dtype,
                    &shape,
                )?;
                (name, shape)
            }
            Cast { new_dtype } => {
                let (x, shape) = self.vals[input(0)]
                    .clone()
                    .ok_or("a value before its producer")?;
                let d = self.constant_string(mil_dtype(*new_dtype)?.1)?;
                let name = self.op(
                    "cast",
                    &[("x", vec![x]), ("dtype", vec![d])],
                    ty.dtype,
                    &shape,
                )?;
                (name, shape)
            }
            Full {
                shape,
                fill_value,
                dtype,
            } => {
                let ones = vec![1; shape.len()];
                let name = self.constant(*dtype, &ones, &scalar_bytes(*dtype, *fill_value))?;
                (name, ones)
            }
            Iota {
                dtype,
                shape,
                dimension,
            } => {
                let mut compact = vec![1; shape.len()];
                compact[*dimension] = shape[*dimension];
                let data: Vec<u8> = (0..shape[*dimension])
                    .flat_map(|i| scalar_bytes(*dtype, Scalar::Int(i as i64)))
                    .collect();
                let name = self.constant(*dtype, &compact, &data)?;
                (name, compact)
            }
            BroadcastInDim {
                shape,
                broadcast_dimensions,
            } => {
                let x = input(0);
                let (name, xs) = self.vals[x].clone().ok_or("a value before its producer")?;
                let dtype = ty.dtype;
                // The operand's dimensions in the result's order.
                let mut perm: Vec<usize> = (0..broadcast_dimensions.len()).collect();
                perm.sort_by_key(|&i| broadcast_dimensions[i]);
                let (name, xs) = self.transpose(name, dtype, &xs, &perm)?;
                let mut compact = vec![1; shape.len()];
                for (k, &i) in perm.iter().enumerate() {
                    compact[broadcast_dimensions[i]] = xs[k];
                }
                let name = match compact == xs {
                    true => name,
                    false => self.reshape(name, dtype, &compact)?,
                };
                (name, compact)
            }
            Transpose { permutation } => {
                let (name, xs) = self.vals[input(0)]
                    .clone()
                    .ok_or("a value before its producer")?;
                self.transpose(name, ty.dtype, &xs, permutation)?
            }
            Reshape { new_sizes } => {
                let (name, xs) = self.vals[input(0)]
                    .clone()
                    .ok_or("a value before its producer")?;
                // The same everywhere: still so.
                if xs.iter().all(|&d| d == 1) {
                    let ones = vec![1; new_sizes.len()];
                    (self.reshape(name, ty.dtype, &ones)?, ones)
                } else {
                    let x = self.full(input(0))?;
                    (self.reshape(x, ty.dtype, new_sizes)?, new_sizes.clone())
                }
            }
            ReduceSum { axes, .. } | ReduceMax { axes } if !axes.is_empty() => {
                let x = input(0);
                let xt = graph.type_of(x).clone();
                let mut name = self.full(x)?;
                // Summed in its accumulation dtype: each element widened.
                let kind = match &node.primitive {
                    ReduceSum { accum_dtype, .. } => {
                        if *accum_dtype != xt.dtype {
                            let d = self.constant_string(mil_dtype(*accum_dtype)?.1)?;
                            name = self.op(
                                "cast",
                                &[("x", vec![name]), ("dtype", vec![d])],
                                *accum_dtype,
                                &xt.shape,
                            )?;
                        }
                        "reduce_sum"
                    }
                    _ => "reduce_max",
                };
                let a = self.ints(axes)?;
                let keep = self.boolean(false)?;
                let args = [
                    ("x", vec![name]),
                    ("axes", vec![a]),
                    ("keep_dims", vec![keep]),
                ];
                (self.op(kind, &args, ty.dtype, &ty.shape)?, ty.shape.clone())
            }
            Slice {
                start_indices,
                limit_indices,
            } => {
                let x = self.full(input(0))?;
                let (b, e) = (self.ints(start_indices)?, self.ints(limit_indices)?);
                let s = self.ints(&vec![1; start_indices.len()])?;
                let args = [
                    ("x", vec![x]),
                    ("begin", vec![b]),
                    ("end", vec![e]),
                    ("stride", vec![s]),
                ];
                (
                    self.op("slice_by_index", &args, ty.dtype, &ty.shape)?,
                    ty.shape.clone(),
                )
            }
            Concatenate { dimension } => {
                let mut values = Vec::new();
                for &v in &node.inputs {
                    values.push(self.full(v)?);
                }
                let (axis, interleave) = (self.int(*dimension as i64)?, self.boolean(false)?);
                let args = [
                    ("values", values),
                    ("axis", vec![axis]),
                    ("interleave", vec![interleave]),
                ];
                (
                    self.op("concat", &args, ty.dtype, &ty.shape)?,
                    ty.shape.clone(),
                )
            }
            Gather { axis } => {
                let (x, i) = (input(0), input(1));
                let (n, it) = (graph.type_of(x).shape[*axis], graph.type_of(i).clone());
                let x = self.full(x)?;
                let i = self.full(i)?;
                // Each index clamped into the axis, as the primitive does.
                let lo = self.constant(it.dtype, &[], &scalar_bytes(it.dtype, Scalar::Int(0)))?;
                let hi = self.constant(
                    it.dtype,
                    &[],
                    &scalar_bytes(it.dtype, Scalar::Int(n as i64 - 1)),
                )?;
                let i = self.op(
                    "maximum",
                    &[("x", vec![i]), ("y", vec![lo])],
                    it.dtype,
                    &it.shape,
                )?;
                let i = self.op(
                    "minimum",
                    &[("x", vec![i]), ("y", vec![hi])],
                    it.dtype,
                    &it.shape,
                )?;
                let (a, b, check) = (self.int(*axis as i64)?, self.int(0)?, self.boolean(false)?);
                let args = [
                    ("x", vec![x]),
                    ("indices", vec![i]),
                    ("axis", vec![a]),
                    ("batch_dims", vec![b]),
                    ("validate_indices", vec![check]),
                ];
                (
                    self.op("gather", &args, ty.dtype, &ty.shape)?,
                    ty.shape.clone(),
                )
            }
            DotGeneral { .. } => self.dot(node)?,
            p => {
                return Err(format!(
                    "Core ML programs do not run {} (with these operands)",
                    p.name()
                ));
            }
        })
    }

    fn constant_string(&mut self, s: &str) -> Result<String, String> {
        let name = self.fresh();
        let op = Message::new()
            .string(1, "const")
            .message(
                3,
                Message::new()
                    .string(1, &name)
                    .message(2, tensor_type(2, &[])),
            )
            .entry(5, "val", string_value(s))
            .entry(5, "name", string_value(&name));
        self.ops.push(op);
        Ok(name)
    }

    /// A dot_general as a (batched) matmul: each operand's dimensions
    /// ordered batch, free, contracting (the left's) or batch, contracting,
    /// free (the right's), folded into [B, M, K] and [B, K, N] (no B if
    /// there is no batch), the product unfolded.
    fn dot(&mut self, node: &crate::graph::Node) -> Result<(String, Vec<usize>), String> {
        let Primitive::DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            rhs_batch,
            output_dtype,
            ..
        } = &node.primitive
        else {
            unreachable!("a dot_general")
        };
        let graph = self.graph;
        let ty = graph.type_of(node.output).clone();
        let (l, r) = (
            graph.type_of(node.inputs[0]).clone(),
            graph.type_of(node.inputs[1]).clone(),
        );
        // float16 operands, a float16 result: Core ML's float16 matmul; a
        // float32 result: of the operands widened (exactly), in float32.
        let dtype = match (l.dtype, *output_dtype) {
            (DType::F16, DType::F16) => DType::F16,
            (DType::F16 | DType::F32, DType::F32) => DType::F32,
            (d, o) => {
                return Err(format!(
                    "Core ML programs run dots of float16 or float32 operands, not {d} to {o}"
                ));
            }
        };
        let free = |rank: usize, batch: &[usize], contracting: &[usize]| -> Vec<usize> {
            (0..rank)
                .filter(|d| !batch.contains(d) && !contracting.contains(d))
                .collect()
        };
        let (lf, rf) = (
            free(l.shape.len(), lhs_batch, lhs_contracting),
            free(r.shape.len(), rhs_batch, rhs_contracting),
        );
        let size =
            |t: &TensorType, dims: &[usize]| dims.iter().map(|&d| t.shape[d]).product::<usize>();
        let (b, m, k, n) = (
            size(&l, lhs_batch),
            size(&l, &lf),
            size(&l, lhs_contracting),
            size(&r, &rf),
        );
        let batched = !lhs_batch.is_empty();
        let folded = |rows: usize, cols: usize| match batched {
            true => vec![b, rows, cols],
            false => vec![rows, cols],
        };
        let operand = |this: &mut Self,
                       v: Var,
                       t: &TensorType,
                       order: Vec<usize>,
                       shape: Vec<usize>|
         -> Result<String, String> {
            let mut name = this.full(v)?;
            if t.dtype != dtype {
                let d = this.constant_string(mil_dtype(dtype)?.1)?;
                name = this.op(
                    "cast",
                    &[("x", vec![name]), ("dtype", vec![d])],
                    dtype,
                    &t.shape,
                )?;
            }
            let (name, s) = this.transpose(name, dtype, &t.shape, &order)?;
            match s == shape {
                true => Ok(name),
                false => this.reshape(name, dtype, &shape),
            }
        };
        let lhs_order = [lhs_batch.as_slice(), &lf, lhs_contracting].concat();
        let rhs_order = [rhs_batch.as_slice(), rhs_contracting, &rf].concat();
        let x = operand(self, node.inputs[0], &l, lhs_order, folded(m, k))?;
        let y = operand(self, node.inputs[1], &r, rhs_order, folded(k, n))?;
        let (tx, ty_) = (self.boolean(false)?, self.boolean(false)?);
        let args = [
            ("x", vec![x]),
            ("y", vec![y]),
            ("transpose_x", vec![tx]),
            ("transpose_y", vec![ty_]),
        ];
        let product = self.op("matmul", &args, dtype, &folded(m, n))?;
        let name = match folded(m, n) == ty.shape {
            true => product,
            false => self.reshape(product, dtype, &ty.shape)?,
        };
        Ok((name, ty.shape.clone()))
    }
}

/// `graph` as a Core ML model, its inputs with `constants` (their bytes,
/// contiguous) constants of it, the others the model's (`in0`, ...): its
/// specification (`Model.proto`), an ML Program, and each of its
/// operations' outputs with the node of `graph` it lowers.
pub(super) fn lower(
    graph: &Graph,
    constants: &[Option<&[u8]>],
) -> Result<(Vec<u8>, Owners), String> {
    let mut lower = Lower {
        graph,
        ops: Vec::new(),
        vals: vec![None; graph.types.len()],
        names: 0,
        node: None,
        owners: Vec::new(),
    };
    let mut function = Message::new();
    let mut description = Message::new();
    let feature = |name: &str, ty: &TensorType| -> Result<Message, String> {
        let shape = io_shape(&ty.shape);
        let array = Message::new()
            .packed_ints(1, shape.iter().map(|&d| d as i64))
            .int(2, array_dtype(ty.dtype)?);
        Ok(Message::new()
            .string(1, name)
            .message(3, Message::new().message(5, array)))
    };
    let mut runtime = 0;
    for (&v, constant) in graph.inputs().iter().zip(constants) {
        let ty = graph.type_of(v).clone();
        let code = mil_dtype(ty.dtype)?.0;
        let name = match constant {
            Some(data) => lower.constant(ty.dtype, &ty.shape, data)?,
            None => {
                let name = format!("in{runtime}");
                runtime += 1;
                description = description.message(1, feature(&name, &ty)?);
                function = function.message(
                    1,
                    Message::new()
                        .string(1, &name)
                        .message(2, tensor_type(code, &io_shape(&ty.shape))),
                );
                match ty.shape.is_empty() {
                    true => lower.reshape(name, ty.dtype, &[])?,
                    false => name,
                }
            }
        };
        lower.vals[v] = Some((name, ty.shape.clone()));
    }
    for (i, node) in graph.nodes().iter().enumerate() {
        lower.node = Some(i);
        let lowered = lower.lower_node(node)?;
        lower.vals[node.output] = Some(lowered);
    }
    lower.node = None;
    let mut outputs = Vec::new();
    for (k, &v) in graph.outputs().iter().enumerate() {
        let ty = graph.type_of(v).clone();
        description = description.message(10, feature(&format!("out{k}"), &ty)?);
        let x = lower.full(v)?;
        // Named as the model's output (a scalar as [1]).
        let name = format!("out{k}");
        match ty.shape.is_empty() {
            true => {
                let s = lower.ints(&[1])?;
                let args = [("x", vec![x]), ("shape", vec![s])];
                lower.named(name.clone(), "reshape", &args, ty.dtype, &[1])?
            }
            false => lower.named(
                name.clone(),
                "identity",
                &[("x", vec![x])],
                ty.dtype,
                &ty.shape,
            )?,
        };
        outputs.push(name);
    }
    let mut block = Message::new();
    for out in &outputs {
        block = block.string(2, out);
    }
    for op in lower.ops {
        block = block.message(3, op);
    }
    let function = function.string(2, "CoreML7").entry(3, "CoreML7", block);
    let program = Message::new().int(1, 1).entry(2, "main", function);
    let spec = Message::new()
        .int(1, 8)
        .message(2, description)
        .message(502, program)
        .into_bytes();
    Ok((spec, lower.owners))
}
