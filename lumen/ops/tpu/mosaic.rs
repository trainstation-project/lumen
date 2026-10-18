//! A builder of Mosaic kernels: MLIR in the `tpu` dialect (with `arith`,
//! `vector`, `memref`, `scf`), the language of a TPU core's own kernels
//! (Pallas lowers to it). It writes the form libtpu deserializes: every op
//! generic and named `stable_mosaic.<dialect>.<op>`, the module carrying
//! `stable_mosaic.version` (`jaxlib/mosaic/serde.cc`).
//!
//! A kernel's arguments are, in order: its grid's indices (`i32`s), its
//! scalars (`i32[n]` in SMEM), optionally host bytes copied to the TPU
//! (`u32[rows, 1, 128]`), the device's memory (`u32[rows, 1, 128]`, which
//! it writes: aliased to its result), the result, its VMEM scratch and a
//! DMA semaphore.

use std::fmt::{self, Write};

/// The Mosaic serialization version written: libtpu reads it and older ones.
const VERSION: u32 = 17;

/// What memory a buffer is in: HBM (`any`: where XLA put it), VMEM or SMEM.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Space {
    Any,
    Vmem,
    Smem,
}

/// A value's type: a scalar, a vector (registers), a buffer or a DMA
/// semaphore. Element types are MLIR's (`i32`, `f32`, `bf16`, `i1`, `index`).
#[derive(Clone, PartialEq)]
pub(crate) enum Type {
    Scalar(&'static str),
    Vector(Vec<usize>, &'static str),
    Memref(Vec<usize>, &'static str, Space),
    Semaphore,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims = |shape: &[usize]| shape.iter().map(|d| format!("{d}x")).collect::<String>();
        match self {
            Type::Scalar(t) => write!(f, "{t}"),
            Type::Vector(shape, t) => write!(f, "vector<{}{t}>", dims(shape)),
            Type::Memref(shape, t, space) => {
                let space = match space {
                    Space::Any => "any",
                    Space::Vmem => "vmem",
                    Space::Smem => "smem",
                };
                write!(f, "memref<{}{t}, #tpu.memory_space<{space}>>", dims(shape))
            }
            Type::Semaphore => {
                write!(
                    f,
                    "memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>"
                )
            }
        }
    }
}

/// An SSA value and its type.
#[derive(Clone)]
pub(crate) struct Value {
    name: String,
    pub(crate) ty: Type,
}

impl Value {
    /// The shape of a vector or buffer.
    pub(crate) fn shape(&self) -> &[usize] {
        match &self.ty {
            Type::Vector(shape, _) | Type::Memref(shape, _, _) => shape,
            _ => &[],
        }
    }

    fn element(&self) -> &'static str {
        match &self.ty {
            Type::Scalar(t) | Type::Vector(_, t) | Type::Memref(_, t, _) => t,
            Type::Semaphore => unreachable!("a semaphore has no elements"),
        }
    }
}

/// A kernel being written: its arguments, then its body op by op.
pub(crate) struct Kernel {
    args: Vec<Value>,
    grid: Vec<usize>,
    scratch: usize,
    windows: usize,
    body: String,
    depth: usize,
    next: usize,
}

impl Kernel {
    /// A kernel run over `grid`, taking `scalars` i32s, `host` rows of host
    /// bytes (if any), the device's `rows` of memory, and VMEM `scratch`
    /// buffers.
    pub(crate) fn new(
        grid: &[usize],
        scalars: usize,
        host: Option<usize>,
        rows: usize,
        scratch: &[Type],
    ) -> Self {
        let rows_of = |n| Type::Memref(vec![n, 1, 128], "i32", Space::Any);
        let mut types: Vec<Type> = grid.iter().map(|_| Type::Scalar("i32")).collect();
        types.push(Type::Memref(vec![scalars], "i32", Space::Smem));
        types.extend(host.map(rows_of));
        types.extend([rows_of(rows), rows_of(rows)]);
        types.extend(scratch.iter().cloned());
        types.push(Type::Semaphore);
        let args = types
            .into_iter()
            .enumerate()
            .map(|(i, ty)| Value {
                name: format!("%arg{i}"),
                ty,
            })
            .collect();
        Kernel {
            args,
            grid: grid.to_vec(),
            scratch: scratch.len(),
            windows: 2 + usize::from(host.is_some()),
            body: String::new(),
            depth: 2,
            next: 0,
        }
    }

    /// The index of grid dimension `d` this step runs.
    pub(crate) fn program_id(&self, d: usize) -> Value {
        self.args[d].clone()
    }

    fn arg(&self, from_end: usize) -> Value {
        self.args[self.args.len() - 1 - from_end].clone()
    }

    /// The host bytes copied to the TPU for this run.
    pub(crate) fn host(&self) -> Value {
        self.args[self.grid.len() + 1].clone()
    }

    /// The device's memory, as it was before the kernel.
    pub(crate) fn memory(&self) -> Value {
        self.arg(self.scratch + 2)
    }

    /// The device's memory, as the kernel writes it.
    pub(crate) fn result(&self) -> Value {
        self.arg(self.scratch + 1)
    }

    /// VMEM scratch buffer `i`.
    pub(crate) fn scratch(&self, i: usize) -> Value {
        self.arg(self.scratch - i)
    }

    /// `name` applied to `operands` with `attrs`, of result type `result`
    /// (none: no result).
    fn op(&mut self, name: &str, operands: &[&Value], attrs: &str, result: Option<Type>) -> Value {
        let value = Value {
            name: match result {
                Some(_) => format!("%{}", self.next),
                None => String::new(),
            },
            ty: result.clone().unwrap_or(Type::Scalar("")),
        };
        let pad = "  ".repeat(self.depth);
        let assign = match result {
            Some(_) => {
                self.next += 1;
                format!("{} = ", value.name)
            }
            None => String::new(),
        };
        let names = operands.iter().map(|v| v.name.as_str()).collect::<Vec<_>>();
        let types = operands
            .iter()
            .map(|v| v.ty.to_string())
            .collect::<Vec<_>>();
        let attrs = match attrs {
            "" => String::new(),
            attrs => format!(" {{{attrs}}}"),
        };
        let result = result.map_or("()".to_string(), |t| t.to_string());
        let _ = writeln!(
            self.body,
            "{pad}{assign}\"stable_mosaic.{name}\"({}){attrs} : ({}) -> {result}",
            names.join(", "),
            types.join(", "),
        );
        value
    }

    /// An i32 constant.
    pub(crate) fn i32(&mut self, v: i64) -> Value {
        let attrs = format!("value = {v} : i32");
        self.op("arith.constant", &[], &attrs, Some(Type::Scalar("i32")))
    }

    fn index(&mut self, v: usize) -> Value {
        let attrs = format!("value = {v} : index");
        self.op("arith.constant", &[], &attrs, Some(Type::Scalar("index")))
    }

    /// A vector of `shape` of i32 `v`s.
    pub(crate) fn splat_i32(&mut self, v: i32, shape: &[usize]) -> Value {
        let ty = Type::Vector(shape.to_vec(), "i32");
        let attrs = format!("value = dense<{v}> : {ty}");
        self.op("arith.constant", &[], &attrs, Some(ty))
    }

    /// A vector of `shape` of f32 zeros.
    pub(crate) fn zeros(&mut self, shape: &[usize]) -> Value {
        let ty = Type::Vector(shape.to_vec(), "f32");
        let attrs = format!("value = dense<0.000000e+00> : {ty}");
        self.op("arith.constant", &[], &attrs, Some(ty))
    }

    /// `a op b` (two values of one type) for an `arith` op `name`.
    fn binary(&mut self, name: &str, a: &Value, b: &Value) -> Value {
        let attrs = match name {
            "muli" | "addi" | "subi" | "shli" => "overflowFlags = #arith.overflow<none>",
            "addf" => "fastmath = #arith.fastmath<none>",
            _ => "",
        };
        self.op(&format!("arith.{name}"), &[a, b], attrs, Some(a.ty.clone()))
    }

    pub(crate) fn add(&mut self, a: &Value, b: &Value) -> Value {
        match a.element() {
            "f32" => self.binary("addf", a, b),
            _ => self.binary("addi", a, b),
        }
    }

    pub(crate) fn mul(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("muli", a, b)
    }

    pub(crate) fn and(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("andi", a, b)
    }

    pub(crate) fn or(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("ori", a, b)
    }

    pub(crate) fn shift_left(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("shli", a, b)
    }

    /// `a >> b`, shifting in zeros.
    pub(crate) fn shift_right(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("shrui", a, b)
    }

    /// `a + b * c` of i32 scalars, `b` and `c` constants folded.
    pub(crate) fn affine(&mut self, a: &Value, b: &Value, c: i64) -> Value {
        let c = self.i32(c);
        let bc = self.mul(b, &c);
        self.add(a, &bc)
    }

    /// Whether i32s `a` and `b` are equal.
    pub(crate) fn equal(&mut self, a: &Value, b: &Value) -> Value {
        self.op(
            "arith.cmpi",
            &[a, b],
            "predicate = 0 : i64",
            Some(Type::Scalar("i1")),
        )
    }

    /// Scalar `i` of the kernel's.
    pub(crate) fn scalar(&mut self, i: usize) -> Value {
        let at = self.index(i);
        let scalars = self.args[self.grid.len()].clone();
        self.op(
            "memref.load",
            &[&scalars, &at],
            "",
            Some(Type::Scalar("i32")),
        )
    }

    /// `v`, which the kernel promises is a multiple of `m`.
    pub(crate) fn multiple_of(&mut self, v: &Value, m: usize) -> Value {
        let attrs = format!("multiple = {m} : i32");
        self.op("tpu.assume_multiple", &[v], &attrs, Some(v.ty.clone()))
    }

    /// The `shape` of `memref` from index `at`.
    pub(crate) fn slice(&mut self, memref: &Value, at: &[&Value], shape: &[usize]) -> Value {
        let Type::Memref(_, t, space) = memref.ty else {
            unreachable!("slicing a memref")
        };
        let attrs = format!("operandSegmentSizes = array<i32: 1, {}, 0>", at.len());
        let operands: Vec<&Value> = std::iter::once(memref).chain(at.iter().copied()).collect();
        let ty = Type::Memref(shape.to_vec(), t, space);
        self.op("tpu.memref_slice", &operands, &attrs, Some(ty))
    }

    /// `memref`'s elements, row-major, as `shape`.
    pub(crate) fn reshape(&mut self, memref: &Value, shape: &[usize]) -> Value {
        let Type::Memref(_, t, space) = memref.ty else {
            unreachable!("reshaping a memref")
        };
        let ty = Type::Memref(shape.to_vec(), t, space);
        self.op("tpu.memref_reshape", &[memref], "", Some(ty))
    }

    /// Copies `src` to `dst` (of one shape), waiting for it.
    pub(crate) fn copy(&mut self, src: &Value, dst: &Value) {
        let sem = self.arg(0);
        self.op(
            "tpu.enqueue_dma",
            &[src, dst, &sem],
            "operandSegmentSizes = array<i32: 1, 0, 1, 1, 0, 0, 0>, priority = 0 : i32, \
             strict_ordering = false",
            None,
        );
        self.op(
            "tpu.wait_dma2",
            &[&sem, src, dst],
            "operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, strict_ordering = false",
            None,
        );
    }

    /// A VMEM buffer's elements, as a vector.
    pub(crate) fn load(&mut self, memref: &Value) -> Value {
        let zero = self.index(0);
        let mut operands = vec![memref];
        operands.extend(memref.shape().iter().map(|_| &zero));
        let ty = Type::Vector(memref.shape().to_vec(), memref.element());
        self.op("vector.load", &operands, "", Some(ty))
    }

    /// Stores vector `v` in VMEM buffer `memref`, of its shape.
    pub(crate) fn store(&mut self, v: &Value, memref: &Value) {
        let zero = self.index(0);
        let mut operands = vec![v, memref];
        operands.extend(memref.shape().iter().map(|_| &zero));
        let attrs = format!(
            "add = false, operandSegmentSizes = array<i32: 1, 1, {}, 0>, strides = array<i32>",
            memref.shape().len()
        );
        self.op("tpu.vector_store", &operands, &attrs, None);
    }

    /// `v`'s bits as elements of `t` (of its width).
    pub(crate) fn bitcast(&mut self, v: &Value, t: &'static str) -> Value {
        let ty = Type::Vector(v.shape().to_vec(), t);
        self.op("tpu.bitcast", &[v], "", Some(ty))
    }

    /// `v` rounded to float type `t`, narrower.
    pub(crate) fn truncate(&mut self, v: &Value, t: &'static str) -> Value {
        let ty = Type::Vector(v.shape().to_vec(), t);
        self.op("arith.truncf", &[v], "", Some(ty))
    }

    /// `v` widened (exactly) to float type `t`.
    pub(crate) fn extend(&mut self, v: &Value, t: &'static str) -> Value {
        let ty = Type::Vector(v.shape().to_vec(), t);
        self.op("arith.extf", &[v], "", Some(ty))
    }

    /// `acc + lhs · rhs` on the matrix unit: `lhs` `[m, k]`, `rhs` `[k, n]`
    /// (`[n, k]` if `transposed`), `acc` f32 `[m, n]`. f32 operands multiply
    /// in full f32 precision.
    pub(crate) fn matmul(
        &mut self,
        lhs: &Value,
        rhs: &Value,
        acc: &Value,
        transposed: bool,
    ) -> Value {
        let (contracting, free) = if transposed { (1, 0) } else { (0, 1) };
        let precision = match lhs.element() {
            "f32" => "precision = #tpu.contract_precision<fp32>, ",
            _ => "",
        };
        let attrs = format!(
            "dimension_numbers = #tpu.dot_dimension_numbers<[1], [{contracting}], [0], [{free}], \
             [0, 0, 1, {free}], [], []>, {precision}transpose_lhs = false, \
             transpose_lhs_hint = false, transpose_rhs = false"
        );
        self.op("tpu.matmul", &[lhs, rhs, acc], &attrs, Some(acc.ty.clone()))
    }

    /// Runs what `body` writes only if `cond`.
    pub(crate) fn when(&mut self, cond: &Value, body: impl FnOnce(&mut Self)) {
        let pad = "  ".repeat(self.depth);
        let _ = writeln!(
            self.body,
            "{pad}\"stable_mosaic.scf.if\"({}) ({{",
            cond.name
        );
        self.depth += 1;
        body(self);
        self.op("scf.yield", &[], "", None);
        self.depth -= 1;
        let _ = writeln!(self.body, "{pad}}}) : (i1) -> ()");
    }

    /// The module: the kernel, named `name`.
    pub(crate) fn finish(mut self, name: &str) -> String {
        self.op("func.return", &[], "", None);
        let args = self
            .args
            .iter()
            .map(|a| format!("{}: {}", a.name, a.ty))
            .collect::<Vec<_>>();
        let types = self
            .args
            .iter()
            .map(|a| a.ty.to_string())
            .collect::<Vec<_>>();
        let semantics = vec!["#tpu.dimension_semantics<arbitrary>"; self.grid.len()];
        let bounds = self.grid.iter().map(usize::to_string).collect::<Vec<_>>();
        format!(
            "\"builtin.module\"() ({{\n  \"stable_mosaic.func.func\"() ({{\n  ^bb0({}):\n{}  }}) \
             {{dimension_semantics = [{}], function_type = ({}) -> (), iteration_bounds = \
             array<i64: {}>, scalar_prefetch = 1 : i64, scratch_operands = {} : i64, sym_name = \
             \"{name}\", tpu.core_type = #tpu.core_type<tc>, window_params = [{}]}} : () -> ()\n}}) \
             {{stable_mosaic.version = {VERSION} : i64}} : () -> ()\n",
            args.join(", "),
            self.body,
            semantics.join(", "),
            types.join(", "),
            bounds.join(", "),
            // The semaphore is scratch too.
            self.scratch + 1,
            vec!["{}"; self.windows].join(", "),
        )
    }
}
