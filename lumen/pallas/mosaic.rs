//! A [`Kernel`] lowered to Mosaic: MLIR in the `tpu` dialect (with `arith`,
//! `math`, `vector`, `memref`, `scf`), written in the form libtpu
//! deserializes: every op generic and named `stable_mosaic.<dialect>.<op>`,
//! the module carrying `stable_mosaic.version` (JAX's
//! `jaxlib/mosaic/serde.cc`). The rules are Pallas's
//! (`jax/_src/pallas/mosaic/lowering.py`).

use std::fmt::Write;

use super::{Block, Kernel, Node, Op, Space, Type, Var};
use crate::graph::{Primitive, TensorType};
use crate::{DType, Scalar};

/// The Mosaic serialization version written: libtpu reads it and older ones.
const VERSION: u32 = 17;

const FAST: &str = "fastmath = #arith.fastmath<none>";
const OVERFLOW: &str = "overflowFlags = #arith.overflow<none>";

fn element(dtype: DType) -> &'static str {
    use DType::*;
    match dtype {
        Bool => "i1",
        U8 | I8 => "i8",
        U16 | I16 => "i16",
        U32 | I32 => "i32",
        U64 | I64 => "i64",
        F16 => "f16",
        BF16 => "bf16",
        F32 => "f32",
        F64 => "f64",
    }
}

fn unsigned(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::Bool
    )
}

fn dims(shape: &[usize]) -> String {
    shape.iter().map(|d| format!("{d}x")).collect()
}

/// A value's type: a scalar, or a vector.
fn value_type(t: &TensorType) -> String {
    match t.shape.is_empty() {
        true => element(t.dtype).to_string(),
        false => format!("vector<{}{}>", dims(&t.shape), element(t.dtype)),
    }
}

/// A buffer's element: its dtype's, but bool's int32 (as Mosaic stores it).
fn stored(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "i32",
        other => element(other),
    }
}

fn type_name(ty: &Type) -> String {
    match ty {
        Type::Value(t) => value_type(t),
        Type::Ref(t, space) => {
            let space = match space {
                Space::Any => "any",
                Space::Vmem => "vmem",
                Space::Smem => "smem",
            };
            format!(
                "memref<{}{}, #tpu.memory_space<{space}>>",
                dims(&t.shape),
                stored(t.dtype)
            )
        }
        Type::View(t, tiled) => {
            let strides: Vec<String> = tiled.strides.iter().map(usize::to_string).collect();
            format!(
                "memref<{}{}, #tpu.tiled<{},[{}]>, #tpu.memory_space<any>>",
                dims(&t.shape),
                stored(t.dtype),
                tiled.tiles,
                strides.join(",")
            )
        }
        Type::Semaphore(shape) => format!(
            "memref<{}!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>",
            dims(shape)
        ),
    }
}

/// `v` as an attribute of `dtype`'s element: an integer wrapped to its
/// width, a float exact (decimal to 17 digits, hex bits if not finite).
fn literal(v: &Scalar, dtype: DType) -> String {
    let bits = dtype.size_of() * 8;
    match (dtype.is_float(), v) {
        (_, Scalar::Bool(b)) if dtype == DType::Bool => b.to_string(),
        (true, _) => {
            let x = v.to_f64();
            match (x.is_finite(), dtype) {
                (true, _) => format!("{x:.17e}"),
                (false, DType::F64) => format!("0x{:016X}", x.to_bits()),
                (false, DType::F32) => format!("0x{:08X}", (x as f32).to_bits()),
                (false, DType::BF16) => {
                    format!("0x{:04X}", crate::tensor::dtype::to_bf16(x).to_bits())
                }
                (false, _) => format!("0x{:04X}", crate::tensor::dtype::to_f16(x).to_bits()),
            }
        }
        (false, _) if dtype == DType::Bool => (v.to_f64() != 0.0).to_string(),
        (false, _) => {
            let x = match v {
                Scalar::Int(i) => *i,
                other => other.to_f64() as i64,
            };
            // Two's complement in `bits`, signed: MLIR's integers are.
            let shift = 64 - bits;
            ((x << shift) >> shift).to_string()
        }
    }
}

/// Operands of names and types, borrowed.
fn refs(ins: &[(String, String)]) -> Vec<(&str, String)> {
    ins.iter().map(|(n, t)| (n.as_str(), t.clone())).collect()
}

/// Writes a kernel's ops.
struct Emitter<'a> {
    kernel: &'a Kernel,
    names: Vec<String>,
    temps: usize,
    out: String,
    depth: usize,
}

impl Emitter<'_> {
    fn name(&self, v: Var) -> &str {
        &self.names[v]
    }

    fn ty(&self, v: Var) -> &Type {
        self.kernel.type_of(v)
    }

    fn value(&self, v: Var) -> &TensorType {
        match self.ty(v) {
            Type::Value(t) => t,
            _ => unreachable!("checked when built"),
        }
    }

    /// `name` of `operands` (names, types) with `attrs`, of results of
    /// `results` (types): their names.
    fn op(
        &mut self,
        name: &str,
        operands: &[(&str, String)],
        attrs: &str,
        results: &[String],
    ) -> Vec<String> {
        let pad = "  ".repeat(self.depth);
        let n = self.temps;
        self.temps += 1;
        let (assign, names) = match results.len() {
            0 => (String::new(), vec![]),
            1 => (format!("%t{n} = "), vec![format!("%t{n}")]),
            k => (
                format!("%t{n}:{k} = "),
                (0..k).map(|i| format!("%t{n}#{i}")).collect(),
            ),
        };
        let args: Vec<&str> = operands.iter().map(|(n, _)| *n).collect();
        let types: Vec<&str> = operands.iter().map(|(_, t)| t.as_str()).collect();
        let attrs = match attrs {
            "" => String::new(),
            a => format!(" {{{a}}}"),
        };
        let result = match results.len() {
            1 => results[0].clone(),
            _ => format!("({})", results.join(", ")),
        };
        let _ = writeln!(
            self.out,
            "{pad}{assign}\"stable_mosaic.{name}\"({}){attrs} : ({}) -> {result}",
            args.join(", "),
            types.join(", ")
        );
        names
    }

    /// A one-result op on values.
    fn unary(
        &mut self,
        name: &str,
        operands: &[&(String, TensorType)],
        attrs: &str,
        out: &TensorType,
    ) -> String {
        let operands: Vec<(&str, String)> = operands
            .iter()
            .map(|(n, t)| (n.as_str(), value_type(t)))
            .collect();
        self.op(name, &operands, attrs, &[value_type(out)])
            .remove(0)
    }

    fn constant(&mut self, v: &Scalar, t: &TensorType) -> String {
        let lit = literal(v, t.dtype);
        let attrs = match t.shape.is_empty() {
            true if t.dtype == DType::Bool => format!("value = {lit}"),
            true => format!("value = {lit} : {}", element(t.dtype)),
            false => format!("value = dense<{lit}> : {}", value_type(t)),
        };
        self.op("arith.constant", &[], &attrs, &[value_type(t)])
            .remove(0)
    }

    fn index(&mut self, i: usize) -> String {
        self.op(
            "arith.constant",
            &[],
            &format!("value = {i} : index"),
            &["index".into()],
        )
        .remove(0)
    }

    fn i32(&mut self, i: usize) -> String {
        self.op(
            "arith.constant",
            &[],
            &format!("value = {i} : i32"),
            &["i32".into()],
        )
        .remove(0)
    }

    fn block(&mut self, block: &Block) -> Result<(), String> {
        for node in &block.nodes {
            self.node(node)?;
        }
        Ok(())
    }

    fn node(&mut self, node: &Node) -> Result<(), String> {
        let ins: Vec<(String, String)> = node
            .inputs
            .iter()
            .map(|&v| (self.name(v).to_string(), type_name(self.ty(v))))
            .collect();
        let out_types: Vec<String> = node
            .outputs
            .iter()
            .map(|&v| type_name(self.ty(v)))
            .collect();
        let names: Vec<String> = match &node.op {
            Op::Prim(p) => {
                let args: Vec<(String, TensorType)> = node
                    .inputs
                    .iter()
                    .map(|&v| (self.name(v).to_string(), self.value(v).clone()))
                    .collect();
                let out = self.value(node.outputs[0]).clone();
                vec![self.primitive(p, &args, &out)?]
            }
            Op::Bitcast(_) => {
                let name = match self.value(node.inputs[0]).shape.is_empty() {
                    true => "arith.bitcast",
                    false => "tpu.bitcast",
                };
                self.op(name, &refs(&ins), "", &out_types)
            }
            Op::ShiftLeft => self.op("arith.shli", &refs(&ins), OVERFLOW, &out_types),
            Op::ShiftRight => self.op("arith.shrui", &refs(&ins), "", &out_types),
            Op::And => self.op("arith.andi", &refs(&ins), "", &out_types),
            Op::Or => self.op("arith.ori", &refs(&ins), "", &out_types),
            Op::MultipleOf(m) => {
                let attrs = format!("multiple = {m} : i32");
                self.op("tpu.assume_multiple", &refs(&ins), &attrs, &out_types)
            }
            Op::Slice { starts, sizes, squeezed } => {
                let src = node.inputs[0];
                let mut dynamic = node.inputs[1..].iter();
                let mut operands = vec![ins[0].clone()];
                for start in starts {
                    let start = match start {
                        Some(s) => self.i32(*s),
                        None => self.name(*dynamic.next().expect("checked when built")).to_string(),
                    };
                    operands.push((start, "i32".into()));
                }
                let whole = match self.ty(src) {
                    Type::Ref(t, space) => Type::Ref(TensorType::new(t.dtype, sizes), *space),
                    Type::View(t, tiled) => Type::View(TensorType::new(t.dtype, sizes), tiled.clone()),
                    _ => Type::Semaphore(sizes.clone()),
                };
                let attrs = format!("operandSegmentSizes = array<i32: 1, {}, 0>", starts.len());
                let sliced = self.op("tpu.memref_slice", &refs(&operands), &attrs, &[type_name(&whole)]);
                match squeezed.iter().any(|&s| s) {
                    true => self.op("tpu.memref_squeeze", &[(&sliced[0], type_name(&whole))], "", &out_types),
                    false => sliced,
                }
            }
            Op::View => {
                // At its row's first element, in elements of its dtype as
                // stored (a word multiple: a row is 512 bytes).
                let Type::View(t, _) = self.ty(node.outputs[0]) else {
                    unreachable!("a view's type")
                };
                let per_row = 512 / t.dtype.size_of().max(if t.dtype == DType::Bool { 4 } else { 1 });
                let k = self.i32(per_row);
                let at = self.op("arith.muli", &[(&ins[1].0, "i32".into()), (&k, "i32".into())], OVERFLOW, &["i32".into()]);
                let attrs = format!("multiple = {per_row} : i32");
                let at = self.op("tpu.assume_multiple", &[(&at[0], "i32".into())], &attrs, &["i32".into()]);
                let operands = [(ins[0].0.as_str(), ins[0].1.clone()), (at[0].as_str(), "i32".to_string())];
                self.op(
                    "tpu.reinterpret_cast",
                    &operands,
                    "operandSegmentSizes = array<i32: 1, 1, 0, 0>",
                    &out_types,
                )
            }
            Op::Reshape(_) => self.op("tpu.memref_reshape", &refs(&ins), "", &out_types),
            Op::RefBitcast(_) => self.op("tpu.memref_bitcast", &refs(&ins), "", &out_types),
            Op::Load => match node.inputs.len() {
                1 => {
                    let t = match self.ty(node.inputs[0]) {
                        Type::Ref(t, _) => t.clone(),
                        _ => unreachable!("checked when built"),
                    };
                    let (rank, dtype) = (t.shape.len(), t.dtype);
                    let zero = self.index(0);
                    let mut operands = refs(&ins);
                    operands.extend((0..rank).map(|_| (zero.as_str(), "index".to_string())));
                    match dtype {
                        // Stored as int32: nonzero.
                        DType::Bool => {
                            let words = TensorType::new(DType::I32, &t.shape);
                            let w = self.op("vector.load", &operands, "", &[value_type(&words)]).remove(0);
                            let zero = (self.constant(&Scalar::Int(0), &words), words.clone());
                            let w = (w, words);
                            let out = self.value(node.outputs[0]).clone();
                            vec![self.unary("arith.cmpi", &[&w, &zero], "predicate = 1 : i64", &out)]
                        }
                        _ => self.op("vector.load", &operands, "", &out_types),
                    }
                }
                _ => {
                    let i = self.op("arith.index_cast", &[(&ins[1].0, "i32".into())], "", &["index".into()]);
                    self.op("memref.load", &[(&ins[0].0, ins[0].1.clone()), (&i[0], "index".into())], "", &out_types)
                }
            },
            Op::Store => {
                let Type::Ref(t, _) = self.ty(node.inputs[0]) else {
                    unreachable!("checked when built")
                };
                let (rank, shape, dtype) = (t.shape.len(), t.shape.clone(), t.dtype);
                // A bool, stored as int32.
                let value = match dtype {
                    DType::Bool => {
                        let words = TensorType::new(DType::I32, &shape);
                        let b = (ins[1].0.clone(), TensorType::new(DType::Bool, &shape));
                        (self.unary("arith.extui", &[&b], "", &words), value_type(&words))
                    }
                    _ => ins[1].clone(),
                };
                let zero = self.index(0);
                let mut operands = vec![(value.0.as_str(), value.1.clone()), (ins[0].0.as_str(), ins[0].1.clone())];
                operands.extend((0..rank).map(|_| (zero.as_str(), "index".to_string())));
                let attrs = format!(
                    "add = false, operandSegmentSizes = array<i32: 1, 1, {rank}, 0>, strides = array<i32>"
                );
                self.op("tpu.vector_store", &operands, &attrs, &[])
            }
            Op::DmaStart => self.op(
                "tpu.enqueue_dma",
                &refs(&ins),
                "operandSegmentSizes = array<i32: 1, 0, 1, 1, 0, 0, 0>, priority = 0 : i32, strict_ordering = false",
                &[],
            ),
            Op::DmaWait => {
                let operands = [ins[2].clone(), ins[0].clone(), ins[1].clone()];
                self.op(
                    "tpu.wait_dma2",
                    &refs(&operands),
                    "operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, strict_ordering = false",
                    &[],
                )
            }
            Op::If(block) => {
                let pad = "  ".repeat(self.depth);
                let _ = writeln!(self.out, "{pad}\"stable_mosaic.scf.if\"({}) ({{", ins[0].0);
                self.depth += 1;
                self.block(block)?;
                self.op("scf.yield", &[], "", &[]);
                self.depth -= 1;
                // No else: its region empty.
                let _ = writeln!(self.out, "{pad}}}, {{\n{pad}}}) : (i1) -> ()");
                vec![]
            }
            Op::For(block) => {
                let step = self.i32(1);
                let pad = "  ".repeat(self.depth);
                let n = self.temps;
                self.temps += 1;
                let k = node.outputs.len();
                let assign = match k {
                    0 => String::new(),
                    1 => format!("%t{n} = "),
                    k => format!("%t{n}:{k} = "),
                };
                let mut operands: Vec<&str> = vec![&ins[0].0, &ins[1].0, &step];
                operands.extend(ins[2..].iter().map(|(n, _)| n.as_str()));
                let _ = writeln!(self.out, "{pad}{assign}\"stable_mosaic.scf.for\"({}) ({{", operands.join(", "));
                for &a in &block.args {
                    self.names[a] = format!("%v{a}");
                }
                let args: Vec<String> = block
                    .args
                    .iter()
                    .map(|&a| format!("{}: {}", self.names[a], type_name(self.ty(a))))
                    .collect();
                let _ = writeln!(self.out, "{pad}^bb0({}):", args.join(", "));
                self.depth += 1;
                self.block(block)?;
                let results: Vec<(String, String)> = block
                    .results
                    .iter()
                    .map(|&v| (self.name(v).to_string(), type_name(self.ty(v))))
                    .collect();
                self.op("scf.yield", &refs(&results), "", &[]);
                self.depth -= 1;
                let mut types = vec!["i32".to_string(); 3];
                types.extend(ins[2..].iter().map(|(_, t)| t.clone()));
                let result = match k {
                    1 => out_types[0].clone(),
                    _ => format!("({})", out_types.join(", ")),
                };
                let _ = writeln!(self.out, "{pad}}}) : ({}) -> {result}", types.join(", "));
                match k {
                    0 => vec![],
                    1 => vec![format!("%t{n}")],
                    k => (0..k).map(|i| format!("%t{n}#{i}")).collect(),
                }
            }
        };
        for (&v, name) in node.outputs.iter().zip(names) {
            self.names[v] = name;
        }
        Ok(())
    }

    /// `p` on `args` (names and types), of result type `out`: its name.
    fn primitive(
        &mut self,
        p: &Primitive,
        args: &[(String, TensorType)],
        out: &TensorType,
    ) -> Result<String, String> {
        use Primitive::*;
        if let Full { fill_value, .. } = p {
            return Ok(self.constant(fill_value, out));
        }
        let a = &args[0];
        let float = a.1.dtype.is_float();
        let uns = unsigned(a.1.dtype);
        Ok(match p {
            Add | Sub | Mul | Div | Max => {
                let (name, attrs) = match (p, float, uns) {
                    (Add, true, _) => ("addf", FAST),
                    (Sub, true, _) => ("subf", FAST),
                    (Mul, true, _) => ("mulf", FAST),
                    (Div, true, _) => ("divf", FAST),
                    (Max, true, _) => ("maximumf", FAST),
                    (Add, false, _) => ("addi", OVERFLOW),
                    (Sub, false, _) => ("subi", OVERFLOW),
                    (Mul, false, _) => ("muli", OVERFLOW),
                    (Div, false, false) => ("divsi", ""),
                    (Div, false, true) => ("divui", ""),
                    (Max, false, false) => ("maxsi", ""),
                    _ => ("maxui", ""),
                };
                self.unary(&format!("arith.{name}"), &[a, &args[1]], attrs, out)
            }
            Eq | Lt => {
                let (name, predicate) = match (p, float, uns) {
                    (Eq, true, _) => ("cmpf", 1),
                    (Lt, true, _) => ("cmpf", 4),
                    (Eq, false, _) => ("cmpi", 0),
                    (Lt, false, false) => ("cmpi", 2),
                    _ => ("cmpi", 6),
                };
                let attrs = match float {
                    true => format!("{FAST}, predicate = {predicate} : i64"),
                    false => format!("predicate = {predicate} : i64"),
                };
                self.unary(&format!("arith.{name}"), &[a, &args[1]], &attrs, out)
            }
            Neg if float => self.unary("arith.negf", &[a], FAST, out),
            Neg => {
                let zero = (self.constant(&Scalar::Int(0), &a.1), a.1.clone());
                self.unary("arith.subi", &[&zero, a], OVERFLOW, out)
            }
            Exp | Log | Sqrt | Tanh => {
                let name = match p {
                    Exp => "math.exp",
                    Log => "math.log",
                    Sqrt => "math.sqrt",
                    _ => "math.tanh",
                };
                self.unary(name, &[a], FAST, out)
            }
            Logistic => {
                // 1 / (1 + exp(-x)).
                let neg = (self.unary("arith.negf", &[a], FAST, out), out.clone());
                let exp = (self.unary("math.exp", &[&neg], FAST, out), out.clone());
                let one = (self.constant(&Scalar::Float(1.0), out), out.clone());
                let sum = (
                    self.unary("arith.addf", &[&one, &exp], FAST, out),
                    out.clone(),
                );
                self.unary("arith.divf", &[&one, &sum], FAST, out)
            }
            Cast { new_dtype } => self.cast(a, *new_dtype)?,
            Select => self.unary("arith.select", &[a, &args[1], &args[2]], "", out),
            BroadcastInDim {
                shape,
                broadcast_dimensions,
            } => {
                if a.1.shape == *shape {
                    return Ok(a.0.clone());
                }
                if a.1.shape.is_empty() {
                    return Ok(self.unary("vector.broadcast", &[a], "", out));
                }
                // The operand's dimensions where they go, 1 elsewhere.
                let mut expanded = vec![1; shape.len()];
                for (&d, &n) in broadcast_dimensions.iter().zip(&a.1.shape) {
                    expanded[d] = n;
                }
                let expanded = TensorType::new(a.1.dtype, &expanded);
                let cast = (
                    self.unary("vector.shape_cast", &[a], "", &expanded),
                    expanded.clone(),
                );
                match expanded.shape == *shape {
                    true => cast.0,
                    false => self.unary("vector.broadcast", &[&cast], "", out),
                }
            }
            Reshape { new_sizes } => match (a.1.shape.is_empty(), new_sizes.is_empty()) {
                _ if a.1.shape == *new_sizes => a.0.clone(),
                (true, false) => self.unary("vector.broadcast", &[a], "", out),
                (false, false) => self.unary("vector.shape_cast", &[a], "", out),
                _ => return Err("TPU kernels do not reshape a vector to a scalar yet".into()),
            },
            ReduceSum { axes, .. } | ReduceMax { axes } if !out.shape.is_empty() => {
                let accum = match p {
                    ReduceSum { accum_dtype, .. } => *accum_dtype,
                    _ => a.1.dtype,
                };
                let src = match accum == a.1.dtype {
                    true => a.clone(),
                    false => {
                        let t = TensorType::new(accum, &a.1.shape);
                        (self.cast(a, accum)?, t)
                    }
                };
                let (kind, identity) = match (p, accum.is_float(), unsigned(accum)) {
                    (ReduceSum { .. }, true, _) => ("add", Scalar::Float(0.0)),
                    (ReduceSum { .. }, false, _) => ("add", Scalar::Int(0)),
                    (_, true, _) => ("maximumf", Scalar::Float(f64::NEG_INFINITY)),
                    (_, false, false) => ("maxsi", Scalar::Int(i64::MIN)),
                    _ => ("maxui", Scalar::Int(0)),
                };
                let acc = (self.constant(&identity, out), out.clone());
                let dims: Vec<String> = axes.iter().map(usize::to_string).collect();
                let attrs = format!(
                    "kind = #vector.kind<{kind}>, reduction_dims = array<i64: {}>",
                    dims.join(", ")
                );
                self.unary("vector.multi_reduction", &[&src, &acc], &attrs, out)
            }
            DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                accum_dtype,
                output_dtype,
            } => {
                let b = &args[1];
                if !lhs_batch.is_empty()
                    || !rhs_batch.is_empty()
                    || a.1.shape.len() != 2
                    || b.1.shape.len() != 2
                    || lhs_contracting.len() != 1
                {
                    return Err(
                        "TPU kernels multiply matrices: two dimensions, one contracting, no batch"
                            .into(),
                    );
                }
                let (lc, rc) = (lhs_contracting[0], rhs_contracting[0]);
                let (lf, rf) = (1 - lc, 1 - rc);
                let acc_type = TensorType::new(*accum_dtype, &out.shape);
                let zero = match accum_dtype.is_float() {
                    true => Scalar::Float(0.0),
                    false => Scalar::Int(0),
                };
                let acc = (self.constant(&zero, &acc_type), acc_type.clone());
                let precision = match a.1.dtype {
                    DType::F32 => "precision = #tpu.contract_precision<fp32>, ",
                    _ => "",
                };
                let attrs = format!(
                    "dimension_numbers = #tpu.dot_dimension_numbers<[{lc}], [{rc}], [{lf}], [{rf}], \
                     [0, {lf}, 1, {rf}], [], []>, {precision}transpose_lhs = false, \
                     transpose_lhs_hint = false, transpose_rhs = false"
                );
                let product = (
                    self.unary("tpu.matmul", &[a, b, &acc], &attrs, &acc_type),
                    acc_type,
                );
                match output_dtype == accum_dtype {
                    true => product.0,
                    false => self.cast(&product, *output_dtype)?,
                }
            }
            Slice {
                start_indices,
                limit_indices,
                ..
            } => {
                let list = |v: &[usize]| {
                    v.iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let sizes: Vec<usize> = start_indices
                    .iter()
                    .zip(limit_indices)
                    .map(|(s, l)| l - s)
                    .collect();
                let attrs = format!(
                    "offsets = [{}], sizes = [{}], strides = [{}]",
                    list(start_indices),
                    list(&sizes),
                    list(&vec![1; sizes.len()])
                );
                self.unary("vector.extract_strided_slice", &[a], &attrs, out)
            }
            other => return Err(format!("{} does not run in TPU kernels yet", other.name())),
        })
    }

    /// `a` converted to `dtype` (`convert_element_type`).
    fn cast(&mut self, a: &(String, TensorType), dtype: DType) -> Result<String, String> {
        let from = a.1.dtype;
        let out = TensorType::new(dtype, &a.1.shape);
        let (bits_in, bits_out) = (from.size_of(), dtype.size_of());
        let name = match (from.is_float(), dtype.is_float()) {
            _ if from == dtype => return Ok(a.0.clone()),
            _ if dtype == DType::Bool => {
                let zero = (self.constant(&Scalar::Int(0), &a.1), a.1.clone());
                return Ok(match from.is_float() {
                    true => self.unary(
                        "arith.cmpf",
                        &[a, &zero],
                        &format!("{FAST}, predicate = 13 : i64"),
                        &out,
                    ),
                    false => self.unary("arith.cmpi", &[a, &zero], "predicate = 1 : i64", &out),
                });
            }
            (true, true) if bits_out > bits_in => "arith.extf",
            (true, true) => "arith.truncf",
            (false, true) if unsigned(from) => "arith.uitofp",
            (false, true) => "arith.sitofp",
            (true, false) if unsigned(dtype) => "arith.fptoui",
            (true, false) => "arith.fptosi",
            // Integers of one width: the same bits.
            (false, false) if bits_out == bits_in && from != DType::Bool => return Ok(a.0.clone()),
            (false, false) if bits_out > bits_in || from == DType::Bool => match unsigned(from) {
                true => "arith.extui",
                false => "arith.extsi",
            },
            (false, false) => "arith.trunci",
        };
        Ok(self.unary(name, &[a], "", &out))
    }
}

/// `kernel` as a Mosaic module, its function named `name`.
pub fn lower(kernel: &Kernel, name: &str) -> Result<String, String> {
    if !kernel.is_closed() {
        return Err("a kernel's `If` or `For` was not ended".into());
    }
    let mut e = Emitter {
        kernel,
        names: vec![String::new(); kernel.types.len()],
        temps: 0,
        out: String::new(),
        depth: 2,
    };
    let params: Vec<Var> = kernel
        .params()
        .iter()
        .chain(kernel.scratch())
        .copied()
        .collect();
    for (i, &p) in params.iter().enumerate() {
        e.names[p] = format!("%arg{i}");
    }
    e.block(kernel.body())?;
    e.op("func.return", &[], "", &[]);
    let args: Vec<String> = params
        .iter()
        .map(|&p| format!("{}: {}", e.names[p], type_name(kernel.type_of(p))))
        .collect();
    let types: Vec<String> = params
        .iter()
        .map(|&p| type_name(kernel.type_of(p)))
        .collect();
    Ok(format!(
        "\"builtin.module\"() ({{\n  \"stable_mosaic.func.func\"() ({{\n  ^bb0({}):\n{}  }}) \
         {{dimension_semantics = [], function_type = ({}) -> (), scalar_prefetch = 1 : i64, \
         scratch_operands = {} : i64, sym_name = \"{name}\", tpu.core_type = #tpu.core_type<tc>}} : \
         () -> ()\n}}) {{stable_mosaic.version = {VERSION} : i64}} : () -> ()\n",
        args.join(", "),
        e.out,
        types.join(", "),
        kernel.scratch().len(),
    ))
}
