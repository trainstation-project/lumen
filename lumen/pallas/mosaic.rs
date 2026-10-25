use std::fmt::Write;

use super::{Block, Kernel, Node, Op, Space, Type, Var};
use crate::graph::{Primitive, TensorType};
use crate::tensor::dtype::{f8e4m3b11fnuz, f8e4m3fn, f8e5m2};
use crate::{DType, Scalar};

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
        I2 | U2 => "i2",
        I4 | U4 => "i4",
        F8E4M3FN => "f8E4M3FN",
        F8E5M2 => "f8E5M2",
        F8E4M3B11FNUZ => "f8E4M3B11FNUZ",
    }
}

fn stored_bits(dtype: DType) -> usize {
    match dtype {
        DType::Bool => 32,
        d => d.bits(),
    }
}

fn unsigned(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::U2 | DType::U4 | DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::Bool
    )
}

fn dims(shape: &[usize]) -> String {
    shape.iter().map(|d| format!("{d}x")).collect()
}

fn value_type(t: &TensorType) -> String {
    match t.shape.is_empty() {
        true => element(t.dtype).to_string(),
        false => format!("vector<{}{}>", dims(&t.shape), element(t.dtype)),
    }
}

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
                Space::Hbm => "hbm",
                Space::Vmem => "vmem",
                Space::VmemShared => "vmem_shared",
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

fn literal(v: &Scalar, dtype: DType) -> String {
    let bits = dtype.bits();
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
                (false, DType::F8E4M3FN) => format!("0x{:02X}", f8e4m3fn::from_f64(x).0),
                (false, DType::F8E5M2) => format!("0x{:02X}", f8e5m2::from_f64(x).0),
                (false, DType::F8E4M3B11FNUZ) => format!("0x{:02X}", f8e4m3b11fnuz::from_f64(x).0),
                (false, _) => format!("0x{:04X}", crate::tensor::dtype::to_f16(x).to_bits()),
            }
        }
        (false, _) if dtype == DType::Bool => (v.to_f64() != 0.0).to_string(),
        (false, _) => {
            let x = match v {
                Scalar::Int(i) => *i,
                other => other.to_f64() as i64,
            };
            let shift = 64 - bits;
            ((x << shift) >> shift).to_string()
        }
    }
}

fn refs(ins: &[(String, String)]) -> Vec<(&str, String)> {
    ins.iter().map(|(n, t)| (n.as_str(), t.clone())).collect()
}

struct Emitter<'a> {
    kernel: &'a Kernel,
    names: Vec<String>,
    views: std::collections::HashMap<Var, (String, String)>,
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
            Op::Roll(axis) => {
                let attrs = format!("dimension = {axis} : si32");
                self.op("tpu.dynamic_rotate", &refs(&ins), &attrs, &out_types)
            }
            Op::MultipleOf(m) => {
                let attrs = format!("multiple = {m} : i32");
                self.op("tpu.assume_multiple", &refs(&ins), &attrs, &out_types)
            }
            Op::Slice { starts, .. } if matches!(self.ty(node.inputs[0]), Type::View(..)) => {
                let tiled = match self.ty(node.inputs[0]) {
                    Type::View(_, tiled) => tiled.clone(),
                    _ => unreachable!("a view"),
                };
                let (memory, base) = self.views[&node.inputs[0]].clone();
                let rank = starts.len();
                let first = rank - tiled.tile.len().min(rank);
                let tile_elems: usize = tiled.tile.iter().product();
                let mut dynamic = node.inputs[1..].iter();
                let mut offset = base;
                for (d, start) in starts.iter().enumerate() {
                    let per = if d >= first { tiled.tile[d - first] } else { 1 };
                    let step = tiled.strides[d] * tile_elems;
                    let term = match start {
                        Some(0) => continue,
                        Some(s) => self.i32(s / per * step),
                        None => {
                            let s = self.name(*dynamic.next().expect("checked when built")).to_string();
                            let s = match per {
                                1 => s,
                                per => {
                                    let p = self.i32(per);
                                    self.op("arith.divsi", &[(&s, "i32".into()), (&p, "i32".into())], "", &["i32".into()])
                                        .remove(0)
                                }
                            };
                            let k = self.i32(step);
                            self.op("arith.muli", &[(&s, "i32".into()), (&k, "i32".into())], OVERFLOW, &["i32".into()])
                                .remove(0)
                        }
                    };
                    offset = self
                        .op("arith.addi", &[(&offset, "i32".into()), (&term, "i32".into())], OVERFLOW, &["i32".into()])
                        .remove(0);
                }
                let Type::View(t, _) = self.ty(node.outputs[0]) else {
                    unreachable!("a view's window")
                };
                let words = 256 / stored_bits(t.dtype);
                let attrs = format!("multiple = {words} : i32");
                let at = self.op("tpu.assume_multiple", &[(&offset, "i32".into())], &attrs, &["i32".into()]);
                let memory_type = type_name(self.ty(*self.kernel.params().last().expect("the memory")));
                self.views.insert(node.outputs[0], (memory.clone(), at[0].clone()));
                self.op(
                    "tpu.reinterpret_cast",
                    &[(&memory, memory_type), (&at[0], "i32".into())],
                    "operandSegmentSizes = array<i32: 1, 1, 0, 0>",
                    &out_types,
                )
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
                let Type::View(t, _) = self.ty(node.outputs[0]) else {
                    unreachable!("a view's type")
                };
                let per_row = 4096 / stored_bits(t.dtype);
                let k = self.i32(per_row);
                let at = self.op("arith.muli", &[(&ins[1].0, "i32".into()), (&k, "i32".into())], OVERFLOW, &["i32".into()]);
                let attrs = format!("multiple = {per_row} : i32");
                let at = self.op("tpu.assume_multiple", &[(&at[0], "i32".into())], &attrs, &["i32".into()]);
                self.views.insert(node.outputs[0], (ins[0].0.clone(), at[0].clone()));
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
            Op::Load => match self.ty(node.inputs[0]) {
                Type::Ref(_, Space::Vmem) => {
                    let t = match self.ty(node.inputs[0]) {
                        Type::Ref(t, _) => t.clone(),
                        _ => unreachable!("checked when built"),
                    };
                    let (rank, dtype) = (t.shape.len(), t.dtype);
                    let zero = self.index(0);
                    let mut operands = refs(&ins);
                    operands.extend((0..rank).map(|_| (zero.as_str(), "index".to_string())));
                    match dtype {
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
                    let dtype = self.value(node.outputs[0]).dtype;
                    let mut operands = vec![(ins[0].0.clone(), ins[0].1.clone())];
                    for (i, _) in &ins[1..] {
                        let i = self.op("arith.index_cast", &[(i, "i32".into())], "", &["index".into()]);
                        operands.push((i[0].clone(), "index".into()));
                    }
                    match dtype {
                        DType::Bool => {
                            let word = TensorType::new(DType::I32, &[]);
                            let w = self.op("memref.load", &refs(&operands), "", &["i32".into()]).remove(0);
                            let zero = (self.constant(&Scalar::Int(0), &word), word.clone());
                            let out = self.value(node.outputs[0]).clone();
                            vec![self.unary("arith.cmpi", &[&(w, word), &zero], "predicate = 1 : i64", &out)]
                        }
                        _ => self.op("memref.load", &refs(&operands), "", &out_types),
                    }
                }
            },
            Op::Store if matches!(self.ty(node.inputs[0]), Type::Ref(_, Space::Smem)) => {
                let value = match self.value(node.inputs[1]).dtype {
                    DType::Bool => {
                        let b = (ins[1].0.clone(), TensorType::new(DType::Bool, &[]));
                        let word = TensorType::new(DType::I32, &[]);
                        (self.unary("arith.extui", &[&b], "", &word), "i32".to_string())
                    }
                    _ => ins[1].clone(),
                };
                let mut operands = vec![value, ins[0].clone()];
                for (i, _) in &ins[2..] {
                    let i = self.op("arith.index_cast", &[(i, "i32".into())], "", &["index".into()]);
                    operands.push((i[0].clone(), "index".into()));
                }
                self.op("memref.store", &refs(&operands), "nontemporal = false", &[])
            }
            Op::Store => {
                let Type::Ref(t, _) = self.ty(node.inputs[0]) else {
                    unreachable!("checked when built")
                };
                let (rank, shape, dtype) = (t.shape.len(), t.shape.clone(), t.dtype);
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
            Op::StridedLoad(strides) | Op::StridedStore(strides) => {
                let list: Vec<String> = strides.iter().map(usize::to_string).collect();
                let attrs = format!("strides = array<i32: {}>", list.join(", "));
                let (mut operands, starts) = match node.outputs.is_empty() {
                    true => (vec![ins[1].clone(), ins[0].clone()], &ins[2..]),
                    false => (vec![ins[0].clone()], &ins[1..]),
                };
                for (i, _) in starts {
                    let i = self.op("arith.index_cast", &[(i, "i32".into())], "", &["index".into()]);
                    operands.push((i[0].clone(), "index".into()));
                }
                let unit = strides.iter().all(|&s| s == 1);
                let rank = strides.len();
                let (name, attrs) = match (node.outputs.is_empty(), unit) {
                    (true, true) => (
                        "tpu.vector_store",
                        format!("add = false, operandSegmentSizes = array<i32: 1, 1, {rank}, 0>, strides = array<i32>"),
                    ),
                    (true, false) => ("tpu.strided_store", attrs),
                    (false, true) => ("vector.load", String::new()),
                    (false, false) => ("tpu.strided_load", attrs),
                };
                self.op(name, &refs(&operands), &attrs, &out_types)
            }
            Op::Delay => self.op("tpu.delay", &refs(&ins), "", &[]),
            Op::DmaStart => self.op(
                "tpu.enqueue_dma",
                &refs(&ins),
                "operandSegmentSizes = array<i32: 1, 0, 1, 1, 0, 0, 0>, priority = 0 : i32, strict_ordering = false",
                &[],
            ),
            Op::IndirectDmaStart => self.op("tpu.enqueue_indirect_dma", &refs(&ins), "add = false", &[]),
            Op::IndirectDmaWait => {
                let operands = [ins[3].clone(), ins[0].clone(), ins[1].clone()];
                self.op("tpu.wait_indirect_dma", &refs(&operands), "", &[])
            }
            Op::DmaWait => {
                let operands = [ins[2].clone(), ins[0].clone(), ins[1].clone()];
                self.op(
                    "tpu.wait_dma2",
                    &refs(&operands),
                    "operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, strict_ordering = false",
                    &[],
                )
            }
            Op::If { then, otherwise } => {
                let (assign, names) = self.results(node.outputs.len());
                let pad = "  ".repeat(self.depth);
                let _ = writeln!(self.out, "{pad}{assign}\"stable_mosaic.scf.if\"({}) ({{", ins[0].0);
                self.region(then, "scf.yield", None)?;
                let _ = writeln!(self.out, "{pad}}}, {{");
                if let Some(otherwise) = otherwise {
                    self.region(otherwise, "scf.yield", None)?;
                }
                let _ = writeln!(self.out, "{pad}}}) : (i1) -> {}", self.signature(&out_types));
                names
            }
            Op::While { before, after } => {
                let (assign, names) = self.results(node.outputs.len());
                let pad = "  ".repeat(self.depth);
                let operands: Vec<&str> = ins.iter().map(|(n, _)| n.as_str()).collect();
                let _ = writeln!(self.out, "{pad}{assign}\"stable_mosaic.scf.while\"({}) ({{", operands.join(", "));
                self.region(before, "scf.condition", Some(&before.args))?;
                let _ = writeln!(self.out, "{pad}}}, {{");
                self.region(after, "scf.yield", Some(&after.args))?;
                let types: Vec<String> = ins.iter().map(|(_, t)| t.clone()).collect();
                let _ = writeln!(self.out, "{pad}}}) : ({}) -> {}", types.join(", "), self.signature(&out_types));
                names
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
        if let Iota { dimension, .. } = p {
            let (shape, dimension) = match out.shape.len() {
                1 => (vec![1, out.shape[0]], 1),
                _ => (out.shape.clone(), *dimension),
            };
            let ty = TensorType::new(out.dtype, &shape);
            let attrs = format!("dimensions = array<i32: {dimension}>");
            let iota = (
                self.op("tpu.iota", &[], &attrs, &[value_type(&ty)])
                    .remove(0),
                ty,
            );
            return Ok(match out.shape.len() {
                1 => self.unary("vector.shape_cast", &[&iota], "", out),
                _ => iota.0,
            });
        }
        let Some(a) = args.first() else {
            return Err(format!("{} does not run in TPU kernels yet", p.name()));
        };
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
            Eq | Lt if a.1.dtype == DType::Bool => {
                let b = &args[1];
                let t = a.1.clone();
                let ones = self.splat(Scalar::Bool(true), &t);
                match p {
                    Eq => {
                        let x = self.apply("arith.xori", &[a, b], "", &t);
                        self.unary("arith.xori", &[&x, &ones], "", out)
                    }
                    _ => {
                        let not_a = self.apply("arith.xori", &[a, &ones], "", &t);
                        self.unary("arith.andi", &[&not_a, b], "", out)
                    }
                }
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
            Select if out.dtype == DType::Bool && !out.shape.is_empty() => {
                let ones = self.splat(Scalar::Bool(true), out);
                let not_a = self.apply("arith.xori", &[a, &ones], "", out);
                let t = self.apply("arith.andi", &[a, &args[1]], "", out);
                let f = self.apply("arith.andi", &[&not_a, &args[2]], "", out);
                self.unary("arith.ori", &[&t, &f], "", out)
            }
            Select => self.unary("arith.select", &[a, &args[1], &args[2]], "", out),
            TakeAlongAxis { axis } => {
                let rank = a.1.shape.len();
                if args[1].1.shape != a.1.shape || axis + 2 < rank {
                    return Err(
                        "take_along_axis runs in TPU kernels of indices of the operand's shape, along one of its last two dimensions"
                            .into(),
                    );
                }
                let bits = a.1.dtype.bits();
                let index = match bits {
                    8 => DType::I8,
                    16 => DType::I16,
                    _ => DType::I32,
                };
                let wide = TensorType::new(index, &a.1.shape);
                let indices = match args[1].1.dtype.bits() {
                    b if b == bits => args[1].clone(),
                    b if b < bits => {
                        let ext = if unsigned(args[1].1.dtype) {
                            "arith.extui"
                        } else {
                            "arith.extsi"
                        };
                        (self.unary(ext, &[&args[1]], "", &wide), wide)
                    }
                    _ => {
                        return Err(format!(
                            "take_along_axis of {} in TPU kernels: indices of its width or narrower, got {}",
                            a.1.dtype, args[1].1.dtype
                        ));
                    }
                };
                let attrs = format!("dimensions = array<i32: {axis}>");
                self.unary("tpu.dynamic_gather", &[a, &indices], &attrs, out)
            }
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
                _ => self.extract(a, out),
            },
            ReduceMax { .. } if a.1.dtype == DType::Bool => {
                let words = TensorType::new(DType::I32, &a.1.shape);
                let w = (self.unary("arith.extui", &[a], "", &words), words);
                let reduced = TensorType::new(DType::I32, &out.shape);
                let r = (self.primitive(p, &[w], &reduced)?, reduced.clone());
                let zero = (self.constant(&Scalar::Int(0), &reduced), reduced);
                self.unary("arith.cmpi", &[&r, &zero], "predicate = 1 : i64", out)
            }
            ReduceSum { axes, .. } | ReduceMax { axes } if out.shape.is_empty() => {
                let mut shape = vec![1];
                shape.extend(&a.1.shape);
                let lifted = TensorType::new(a.1.dtype, &shape);
                let lifted = (self.unary("vector.shape_cast", &[a], "", &lifted), lifted);
                let axes: Vec<usize> = axes.iter().map(|d| d + 1).collect();
                let reduced = match p {
                    ReduceSum { accum_dtype, .. } => ReduceSum {
                        axes,
                        accum_dtype: *accum_dtype,
                    },
                    _ => ReduceMax { axes },
                };
                let one = TensorType::new(out.dtype, &[1]);
                let v = (
                    self.primitive(&reduced, std::slice::from_ref(&lifted), &one)?,
                    one,
                );
                self.extract(&v, out)
            }
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
                    (_, false, false) => ("maxsi", Scalar::Int(-(1 << (accum.bits() - 1)))),
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
                if lhs_contracting.len() > 1 {
                    return Err(
                        "a dot of more than one contracting dimension is not lowered".into(),
                    );
                }
                let free = |rank: usize, contracting: &[usize], batch: &[usize]| -> Vec<usize> {
                    (0..rank)
                        .filter(|d| !contracting.contains(d) && !batch.contains(d))
                        .collect()
                };
                let (lf, rf) = (
                    free(a.1.shape.len(), lhs_contracting, lhs_batch),
                    free(b.1.shape.len(), rhs_contracting, rhs_batch),
                );
                let list = |v: &[usize]| {
                    v.iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let order: Vec<String> = lhs_batch
                    .iter()
                    .map(|d| format!("0, {d}"))
                    .chain(lf.iter().map(|d| format!("0, {d}")))
                    .chain(rf.iter().map(|d| format!("1, {d}")))
                    .collect();
                let numbers = format!(
                    "#tpu.dot_dimension_numbers<[{}], [{}], [{}], [{}], [{}], [{}], [{}]>",
                    list(lhs_contracting),
                    list(rhs_contracting),
                    list(&lf),
                    list(&rf),
                    order.join(", "),
                    list(lhs_batch),
                    list(rhs_batch)
                );
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
                    "dimension_numbers = {numbers}, {precision}transpose_lhs = false, \
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
            Unary(op) => self.unary_op(*op, a, out)?,
            Binary(op) => self.binary_op(*op, a, &args[1], out)?,
            Concatenate { dimension, .. } => {
                let parts: Vec<&(String, TensorType)> = args.iter().collect();
                let attrs = format!("dimension = {dimension} : i32");
                self.unary("tpu.concatenate", &parts, &attrs, out)
            }
            Transpose { permutation } => {
                let attrs = format!(
                    "permutation = array<i64: {}>",
                    permutation
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                self.unary("tpu.transpose", &[a], &attrs, out)
            }
            other => return Err(format!("{} does not run in TPU kernels yet", other.name())),
        })
    }

    fn results(&mut self, k: usize) -> (String, Vec<String>) {
        let n = self.temps;
        self.temps += 1;
        match k {
            0 => (String::new(), vec![]),
            1 => (format!("%t{n} = "), vec![format!("%t{n}")]),
            k => (
                format!("%t{n}:{k} = "),
                (0..k).map(|i| format!("%t{n}#{i}")).collect(),
            ),
        }
    }

    fn signature(&self, types: &[String]) -> String {
        match types.len() {
            1 => types[0].clone(),
            _ => format!("({})", types.join(", ")),
        }
    }

    fn region(
        &mut self,
        block: &Block,
        terminator: &str,
        args: Option<&[Var]>,
    ) -> Result<(), String> {
        if let Some(args) = args {
            for &a in args {
                self.names[a] = format!("%v{a}");
            }
            let pad = "  ".repeat(self.depth);
            let list: Vec<String> = args
                .iter()
                .map(|&a| format!("{}: {}", self.names[a], type_name(self.ty(a))))
                .collect();
            let _ = writeln!(self.out, "{pad}^bb0({}):", list.join(", "));
        }
        self.depth += 1;
        self.block(block)?;
        let results: Vec<(String, String)> = block
            .results
            .iter()
            .map(|&v| (self.name(v).to_string(), type_name(self.ty(v))))
            .collect();
        self.op(terminator, &refs(&results), "", &[]);
        self.depth -= 1;
        Ok(())
    }

    fn extract(&mut self, a: &(String, TensorType), out: &TensorType) -> String {
        let zeros = vec!["0"; a.1.shape.len()].join(", ");
        let attrs = format!("static_position = array<i64: {zeros}>");
        self.unary("vector.extract", &[a], &attrs, out)
    }

    fn splat(&mut self, v: Scalar, ty: &TensorType) -> (String, TensorType) {
        (self.constant(&v, ty), ty.clone())
    }

    fn apply(
        &mut self,
        name: &str,
        args: &[&(String, TensorType)],
        attrs: &str,
        out: &TensorType,
    ) -> (String, TensorType) {
        (self.unary(name, args, attrs, out), out.clone())
    }

    fn compare(
        &mut self,
        a: &(String, TensorType),
        b: &(String, TensorType),
        predicate: u32,
    ) -> (String, TensorType) {
        let out = TensorType::new(DType::Bool, &a.1.shape);
        match a.1.dtype.is_float() {
            true => self.apply(
                "arith.cmpf",
                &[a, b],
                &format!("{FAST}, predicate = {predicate} : i64"),
                &out,
            ),
            false => self.apply(
                "arith.cmpi",
                &[a, b],
                &format!("predicate = {predicate} : i64"),
                &out,
            ),
        }
    }

    fn is_nan(&mut self, x: &(String, TensorType)) -> (String, TensorType) {
        let eq = self.compare(x, x, 1);
        let ones = self.splat(Scalar::Bool(true), &eq.1);
        self.apply("arith.xori", &[&eq, &ones], "", &eq.1.clone())
    }

    fn select(
        &mut self,
        c: &(String, TensorType),
        t: &(String, TensorType),
        f: &(String, TensorType),
    ) -> (String, TensorType) {
        let out = t.1.clone();
        self.apply("arith.select", &[c, t, f], "", &out)
    }

    fn unary_op(
        &mut self,
        op: crate::graph::UnaryOp,
        a: &(String, TensorType),
        out: &TensorType,
    ) -> Result<String, String> {
        use crate::graph::UnaryOp as U;
        let t = a.1.clone();
        let float = t.dtype.is_float();
        let math = |op: U| match op {
            U::Sin => "math.sin",
            U::Cos => "math.cos",
            U::Tan => "math.tan",
            U::Asin => "math.asin",
            U::Acos => "math.acos",
            U::Atan => "math.atan",
            U::Sinh => "math.sinh",
            U::Cosh => "math.cosh",
            U::Asinh => "math.asinh",
            U::Acosh => "math.acosh",
            U::Atanh => "math.atanh",
            U::Floor => "math.floor",
            U::Ceil => "math.ceil",
            U::RoundEven => "math.roundeven",
            U::Log1p => "math.log1p",
            U::Expm1 => "math.expm1",
            U::Exp2 => "math.exp2",
            U::Log2 => "math.log2",
            U::Rsqrt => "math.rsqrt",
            U::Cbrt => "math.cbrt",
            _ => "",
        };
        Ok(match op {
            U::Abs if float => self.unary("math.absf", &[a], FAST, out),
            U::Abs => self.unary("math.absi", &[a], "", out),
            U::PopulationCount => self.unary("math.ctpop", &[a], "", out),
            U::Clz => self.unary("math.ctlz", &[a], "", out),
            U::Not => {
                let ones = match t.dtype {
                    DType::Bool => Scalar::Bool(true),
                    _ => Scalar::Int(-1),
                };
                let ones = self.splat(ones, &t);
                self.unary("arith.xori", &[a, &ones], "", out)
            }
            U::Sign => {
                let (zero, one, minus) = match float {
                    true => (Scalar::Float(0.0), Scalar::Float(1.0), Scalar::Float(-1.0)),
                    false => (Scalar::Int(0), Scalar::Int(1), Scalar::Int(-1)),
                };
                let (zero, one, minus) = (
                    self.splat(zero, &t),
                    self.splat(one, &t),
                    self.splat(minus, &t),
                );
                let (gt, lt) = match (float, unsigned(t.dtype)) {
                    (true, _) => (2, 4),
                    (false, false) => (4, 2),
                    (false, true) => (8, 6),
                };
                let above = self.compare(a, &zero, gt);
                let below = self.compare(a, &zero, lt);
                let rest = self.select(&below, &minus, a);
                self.select(&above, &one, &rest).0
            }
            U::IsFinite => {
                let abs = self.apply("math.absf", &[a], FAST, &t);
                let inf = self.splat(Scalar::Float(f64::INFINITY), &t);
                self.compare(&abs, &inf, 4).0
            }
            U::ErfInv => self.erf_inv(a)?,
            U::ReducePrecision {
                exponent_bits,
                mantissa_bits,
            } => self.reduce_precision(a, exponent_bits, mantissa_bits)?,
            op => self.unary(math(op), &[a], FAST, out),
        })
    }

    fn binary_op(
        &mut self,
        op: crate::graph::BinaryOp,
        a: &(String, TensorType),
        b: &(String, TensorType),
        out: &TensorType,
    ) -> Result<String, String> {
        use crate::graph::BinaryOp as B;
        let (float, uns) = (a.1.dtype.is_float(), unsigned(a.1.dtype));
        if matches!(op, B::Pow) && b.1.dtype.is_int() {
            return self.integer_pow(a, b, out);
        }
        let (name, attrs) = match op {
            B::Pow => ("math.powf", FAST),
            B::Rem if float => ("arith.remf", FAST),
            B::Rem if uns => ("arith.remui", ""),
            B::Rem => ("arith.remsi", ""),
            B::Min if float => ("arith.minimumf", FAST),
            B::Min if uns => ("arith.minui", ""),
            B::Min => ("arith.minsi", ""),
            B::And => ("arith.andi", ""),
            B::Or => ("arith.ori", ""),
            B::Xor => ("arith.xori", ""),
            B::ShiftLeft => ("arith.shli", OVERFLOW),
            B::ShiftRightLogical => ("arith.shrui", ""),
            B::ShiftRightArithmetic => ("arith.shrsi", ""),
            B::Nextafter => return self.nextafter(a, b),
        };
        Ok(self.unary(name, &[a, b], attrs, out))
    }

    fn integer_pow(
        &mut self,
        a: &(String, TensorType),
        b: &(String, TensorType),
        out: &TensorType,
    ) -> Result<String, String> {
        let f32 = TensorType::new(DType::F32, &a.1.shape);
        let i32 = TensorType::new(DType::I32, &a.1.shape);
        let x = (self.cast(a, DType::F32)?, f32.clone());
        let e = (self.cast(b, DType::F32)?, f32.clone());
        let abs = self.apply("math.absf", &[&x], FAST, &f32);
        let magnitude = self.apply("math.powf", &[&abs, &e], FAST, &f32);
        let bits = self.apply("tpu.bitcast", &[&x], "", &i32);
        let zero = self.splat(Scalar::Int(0), &i32);
        let negative = self.compare(&bits, &zero, 2);
        let one = self.splat(Scalar::Int(1), &b.1);
        let low = self.apply("arith.andi", &[b, &one], "", &b.1);
        let none = self.splat(Scalar::Int(0), &b.1);
        let odd = self.compare(&low, &none, 1);
        let flip = self.apply("arith.andi", &[&negative, &odd], "", &odd.1.clone());
        let negated = self.apply("arith.negf", &[&magnitude], FAST, &f32);
        let r = self.select(&flip, &negated, &magnitude);
        self.cast(&r, out.dtype)
    }

    fn erf_inv(&mut self, x: &(String, TensorType)) -> Result<String, String> {
        let t = x.1.clone();
        if t.dtype != DType::F32 {
            return Err("erf_inv runs in TPU kernels on float32".into());
        }
        let c = |e: &mut Self, v: f64| e.splat(Scalar::Float(v), &t);
        let neg = self.apply("arith.negf", &[x], FAST, &t);
        let square = self.apply("arith.mulf", &[x, &neg], FAST, &t);
        let log = self.apply("math.log1p", &[&square], FAST, &t);
        let w = self.apply("arith.negf", &[&log], FAST, &t);
        let poly = |e: &mut Self, w: &(String, TensorType), cs: &[f64]| {
            let mut p = c(e, cs[0]);
            for &k in &cs[1..] {
                let pw = e.apply("arith.mulf", &[&p, w], FAST, &t);
                let k = c(e, k);
                p = e.apply("arith.addf", &[&k, &pw], FAST, &t);
            }
            p
        };
        let two_half = c(self, 2.5);
        let w1 = self.apply("arith.subf", &[&w, &two_half], FAST, &t);
        let p1 = poly(
            self,
            &w1,
            &[
                2.810_226_36e-08,
                3.432_739_39e-07,
                -3.523_387_7e-06,
                -4.391_506_54e-06,
                0.000_218_580_87,
                -0.001_253_725_03,
                -0.004_177_681_64,
                0.246_640_727,
                1.501_409_41,
            ],
        );
        let root = self.apply("math.sqrt", &[&w], FAST, &t);
        let three = c(self, 3.0);
        let w2 = self.apply("arith.subf", &[&root, &three], FAST, &t);
        let p2 = poly(
            self,
            &w2,
            &[
                -0.000_200_214_257,
                0.000_100_950_558,
                0.001_349_343_22,
                -0.003_673_428_44,
                0.005_739_507_73,
                -0.007_622_461_3,
                0.009_438_870_47,
                1.001_674_06,
                2.832_976_82,
            ],
        );
        let five = c(self, 5.0);
        let small = self.compare(&w, &five, 4);
        let p = self.select(&small, &p1, &p2);
        let px = self.apply("arith.mulf", &[&p, x], FAST, &t);
        let abs = self.apply("math.absf", &[x], FAST, &t);
        let one = c(self, 1.0);
        let edge = self.compare(&abs, &one, 1);
        let inf = c(self, f64::INFINITY);
        let infinite = self.apply("arith.mulf", &[x, &inf], FAST, &t);
        Ok(self.select(&edge, &infinite, &px).0)
    }

    fn reduce_precision(
        &mut self,
        x: &(String, TensorType),
        exponent_bits: u32,
        mantissa_bits: u32,
    ) -> Result<String, String> {
        let t = x.1.clone();
        let f32t = TensorType::new(DType::F32, &t.shape);
        let i32t = TensorType::new(DType::I32, &t.shape);
        let wide = match t.dtype {
            DType::F32 => x.clone(),
            DType::BF16 | DType::F16 => self.apply("arith.extf", &[x], "", &f32t),
            other => return Err(format!("reduce_precision of {other} in TPU kernels")),
        };
        let int = |e: &mut Self, v: i64| e.splat(Scalar::Int(v), &i32t);
        let mut bits = self.apply("tpu.bitcast", &[&wide], "", &i32t);
        if mantissa_bits < 23 {
            let shift = 23 - mantissa_bits;
            let last = int(self, 1 << shift);
            let base = int(self, (1 << (shift - 1)) - 1);
            let s = int(self, i64::from(shift));
            let lastbit = self.apply("arith.andi", &[&bits, &last], "", &i32t);
            let lastbit = self.apply("arith.shrui", &[&lastbit, &s], "", &i32t);
            let bias = self.apply("arith.addi", &[&lastbit, &base], OVERFLOW, &i32t);
            let sum = self.apply("arith.addi", &[&bits, &bias], OVERFLOW, &i32t);
            let mask = int(self, !((1i64 << shift) - 1));
            bits = self.apply("arith.andi", &[&sum, &mask], "", &i32t);
        }
        if exponent_bits < 8 {
            let reduced = (1i64 << (exponent_bits - 1)) - 1;
            let exp_mask = int(self, 0x7F80_0000);
            let exponent = self.apply("arith.andi", &[&bits, &exp_mask], "", &i32t);
            let max = int(self, (127 + reduced) << 23);
            let min = int(self, (127 - reduced) << 23);
            let over = self.compare(&exponent, &max, 8);
            let under = self.compare(&exponent, &min, 7);
            let sign_mask = int(self, -0x8000_0000);
            let sign = self.apply("arith.andi", &[&bits, &sign_mask], "", &i32t);
            let inf = self.apply("arith.ori", &[&sign, &exp_mask], "", &i32t);
            let kept = self.select(&under, &sign, &bits);
            bits = self.select(&over, &inf, &kept);
        }
        let back = self.apply("tpu.bitcast", &[&bits], "", &f32t);
        let nan = self.is_nan(&wide);
        let result = self.select(&nan, &wide, &back);
        Ok(match t.dtype {
            DType::F32 => result.0,
            _ => self.unary("arith.truncf", &[&result], "", &t),
        })
    }

    fn nextafter(
        &mut self,
        x: &(String, TensorType),
        y: &(String, TensorType),
    ) -> Result<String, String> {
        let t = x.1.clone();
        let bits_dtype = match t.dtype {
            DType::F32 => DType::I32,
            DType::BF16 | DType::F16 => DType::I16,
            other => return Err(format!("nextafter of {other} in TPU kernels")),
        };
        let it = TensorType::new(bits_dtype, &t.shape);
        let width = 8 * t.dtype.size_of() as i64;
        let int = |e: &mut Self, v: i64| e.splat(Scalar::Int(v), &it);
        let xb = self.apply("tpu.bitcast", &[x], "", &it);
        let one = int(self, 1);
        let up = self.apply("arith.addi", &[&xb, &one], OVERFLOW, &it);
        let down = self.apply("arith.subi", &[&xb, &one], OVERFLOW, &it);
        let zero = self.splat(Scalar::Float(0.0), &t);
        let towards_up = self.compare(x, y, 4);
        let positive = self.compare(x, &zero, 2);
        let away = self.apply(
            "arith.xori",
            &[&towards_up, &positive],
            "",
            &positive.1.clone(),
        );
        let next = self.select(&away, &down, &up);
        let sign_bit = int(self, -(1i64 << (width - 1)) | 1);
        let tiny_neg = sign_bit;
        let below = self.compare(y, &zero, 4);
        let tiny = self.select(&below, &tiny_neg, &one);
        let is_zero = self.compare(x, &zero, 1);
        let next = self.select(&is_zero, &tiny, &next);
        let next = self.apply("tpu.bitcast", &[&next], "", &t);
        let same = self.compare(x, y, 1);
        let next = self.select(&same, y, &next);
        let nan_x = self.is_nan(x);
        let nan_y = self.is_nan(y);
        let nan = self.apply("arith.ori", &[&nan_x, &nan_y], "", &nan_x.1);
        let quiet = self.splat(Scalar::Float(f64::NAN), &t);
        Ok(self.select(&nan, &quiet, &next).0)
    }

    fn cast(&mut self, a: &(String, TensorType), dtype: DType) -> Result<String, String> {
        let from = a.1.dtype;
        let out = TensorType::new(dtype, &a.1.shape);
        let (bits_in, bits_out) = (from.bits(), dtype.bits());
        if from == DType::Bool && dtype.is_float() {
            let wide = TensorType::new(DType::I32, &a.1.shape);
            let w = (self.unary("arith.extui", &[a], "", &wide), wide);
            return self.cast(&w, dtype);
        }
        if dtype == DType::Bool && from != DType::Bool && bits_in < 32 {
            let wide = match from.is_float() {
                true => DType::F32,
                false if unsigned(from) => DType::U32,
                false => DType::I32,
            };
            let w = (self.cast(a, wide)?, TensorType::new(wide, &a.1.shape));
            return self.cast(&w, DType::Bool);
        }
        let name = match (from.is_float(), dtype.is_float()) {
            _ if from == dtype => return Ok(a.0.clone()),
            _ if dtype == DType::Bool => {
                let zero = (self.constant(&Scalar::Int(0), &a.1), a.1.clone());
                return Ok(match from.is_float() {
                    true => {
                        let eq = self.apply(
                            "arith.cmpf",
                            &[a, &zero],
                            &format!("{FAST}, predicate = 1 : i64"),
                            &out,
                        );
                        let ones = self.splat(Scalar::Bool(true), &out);
                        self.unary("arith.xori", &[&eq, &ones], "", &out)
                    }
                    false => self.unary("arith.cmpi", &[a, &zero], "predicate = 1 : i64", &out),
                });
            }
            (true, true) if bits_out == bits_in => {
                let wide = TensorType::new(DType::F32, &a.1.shape);
                let w = (self.unary("arith.extf", &[a], "", &wide), wide);
                return self.cast(&w, dtype);
            }
            (true, true) if bits_out > bits_in => "arith.extf",
            (true, true) => "arith.truncf",
            (false, true) if unsigned(from) => "arith.uitofp",
            (false, true) => "arith.sitofp",
            (true, false) if unsigned(dtype) => "arith.fptoui",
            (true, false) => "arith.fptosi",
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

pub fn lower(kernel: &Kernel, name: &str) -> Result<String, String> {
    if !kernel.is_closed() {
        return Err("a kernel's `If` or `For` was not ended".into());
    }
    let mut e = Emitter {
        kernel,
        names: vec![String::new(); kernel.types.len()],
        views: Default::default(),
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
    let attributes = match kernel.mesh() {
        None => format!(
            "dimension_semantics = [], function_type = ({}) -> (), scalar_prefetch = 1 : i64, \
             scratch_operands = {} : i64, sym_name = \"{name}\", tpu.core_type = #tpu.core_type<tc>",
            types.join(", "),
            kernel.scratch().len(),
        ),
        Some([cores, subcores]) => {
            let windows: Vec<String> = kernel.params()[2..]
                .iter()
                .map(|&p| {
                    let Type::Ref(t, _) = kernel.type_of(p) else {
                        unreachable!("a SparseCore kernel's operands are refs")
                    };
                    let zeros = vec!["0"; t.shape.len()].join(", ");
                    format!("{{transform_indices = affine_map<(d0, d1) -> ({zeros})>}}")
                })
                .collect();
            format!(
                "dimension_semantics = [#tpu.dimension_semantics<core_parallel>, \
                 #tpu.dimension_semantics<subcore_parallel>], function_type = ({}) -> (), \
                 iteration_bounds = array<i64: {cores}, {subcores}>, scalar_prefetch = 0 : i64, \
                 scratch_operands = {} : i64, sym_name = \"{name}\", \
                 tpu.core_type = #tpu.core_type<sc_vector_subcore>, window_params = [{}]",
                types.join(", "),
                kernel.scratch().len(),
                windows.join(", "),
            )
        }
    };
    Ok(format!(
        "\"builtin.module\"() ({{\n  \"stable_mosaic.func.func\"() ({{\n  ^bb0({}):\n{}  }}) \
         {{{attributes}}} : () -> ()\n}}) {{stable_mosaic.version = {VERSION} : i64}} : () -> ()\n",
        args.join(", "),
        e.out,
    ))
}
