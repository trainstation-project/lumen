//! lumen's Pallas: TPU kernels written in Python with Pallas's API
//! (`lumen/pallas/__init__.py`), traced by lumen's tracer into a [`Kernel`],
//! and lowered here to Mosaic ([`mosaic`]), the MLIR dialect of a TPU core's
//! kernels, which libtpu compiles. No JAX: this is Pallas's front end, its
//! lowering rules and pipeline ported to lumen (JAX's are Apache 2.0).
//!
//! A kernel is a block of nodes on values and refs. Values are scalars or
//! vectors, computed by graph [`Primitive`]s (their types inferred as the
//! graph's are) or by a TPU's bit ops; refs are buffers in a memory space
//! (HBM, VMEM, SMEM), sliced, reshaped, loaded, stored and copied by DMA;
//! `If` and `For` hold blocks of their own. Every kernel works on a
//! device's whole memory (`u32[rows, 1, 128]`, the TPU allocator's buffer):
//! its parameters are `i32` scalars (SMEM), optionally host bytes copied to
//! the TPU, the memory as read, and the memory as written (the same
//! buffer), then its scratch (VMEM buffers, DMA semaphores).

use crate::DType;
use crate::graph::{Primitive, TensorType};

pub mod mosaic;
#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
mod tests;

pub type Var = usize;

/// Where a buffer is: HBM (`Any`: wherever XLA put it), a core's vector
/// memory, or its scalar memory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Space {
    Any,
    Vmem,
    Smem,
}

/// A kernel value's type.
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    /// A scalar (shape `[]`) or a vector (in registers).
    Value(TensorType),
    /// A buffer of a memory space.
    Ref(TensorType, Space),
    /// DMA semaphores, an array of `shape` (`[]`: one).
    Semaphore(Vec<usize>),
    /// A tensor in the device's memory, in a TPU's tiles
    /// (`device::tpu::layout`): an HBM ref of that layout.
    View(TensorType, Tiled),
}

/// A view's tiled layout: its tiles (Mosaic's, `(8,128)(2,1)`), a tile's
/// shape (its last dimensions'), and each dimension's stride in tiles.
#[derive(Clone, Debug, PartialEq)]
pub struct Tiled {
    pub tiles: String,
    pub tile: Vec<usize>,
    pub strides: Vec<usize>,
}

/// What a node does.
#[derive(Clone, Debug)]
pub enum Op {
    /// A graph primitive on values: its operands, its result.
    Prim(Primitive),
    /// A value's bits as another dtype of its width (`pltpu.bitcast`).
    Bitcast(DType),
    /// A tensor of a type in the device's memory at an i32 row of it (its
    /// operands the memory and the row), in its TPU layout
    /// (`tpu.reinterpret_cast`).
    View,
    /// Integer bit ops, of two values of one type (`lax.shift_left`,
    /// `shift_right_logical`, `bitwise_and`, `bitwise_or`).
    ShiftLeft,
    ShiftRight,
    And,
    Or,
    /// An i32 scalar, which the kernel promises is a multiple of this
    /// (`pl.multiple_of`).
    MultipleOf(usize),
    /// A ref's window: from `starts` (`None`: the next dynamic operand, an
    /// i32 scalar) of `sizes`, without the dimensions `squeezed` (of size
    /// 1). A semaphore array's too.
    Slice {
        starts: Vec<Option<usize>>,
        sizes: Vec<usize>,
        squeezed: Vec<bool>,
    },
    /// A ref's elements, row-major, as another shape.
    Reshape(Vec<usize>),
    /// A ref's bits as another dtype of its width.
    RefBitcast(DType),
    /// A VMEM ref's whole value; an SMEM ref's element at an i32 index.
    Load,
    /// Stores a value in a VMEM ref (of its shape): `(ref, value)`.
    Store,
    /// Starts copying `src` to `dst` (of one shape), signalling `sem`:
    /// `(src, dst, sem)`.
    DmaStart,
    /// Waits for the copy of `(src, dst, sem)`.
    DmaWait,
    /// Runs its block if its i1 operand.
    If(Block),
    /// Runs its block for `i` in `lo..hi` (i32 operands), carrying values
    /// (the operands after them): the block's arguments are `i` and the
    /// carried values, its results the next ones, the node's the last.
    For(Block),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<Var>,
    pub outputs: Vec<Var>,
}

/// Nodes in order, with arguments (a loop's) and results (a loop's next
/// carried values).
#[derive(Clone, Debug, Default)]
pub struct Block {
    pub args: Vec<Var>,
    pub nodes: Vec<Node>,
    pub results: Vec<Var>,
}

/// A block being written: the kernel's body, an `If`'s or a `For`'s.
#[derive(Debug)]
enum Open {
    If { cond: Var, block: Block },
    For { inputs: Vec<Var>, block: Block },
}

/// A kernel: its parameters, scratch and body.
#[derive(Debug)]
pub struct Kernel {
    types: Vec<Type>,
    params: Vec<Var>,
    scratch: Vec<Var>,
    body: Block,
    open: Vec<Open>,
}

/// The memory, as a kernel sees it: `rows` rows of 128 32-bit words.
fn words(rows: usize) -> Type {
    Type::Ref(TensorType::new(DType::U32, &[rows, 1, 128]), Space::Any)
}

impl Kernel {
    /// A kernel taking `scalars` i32s, `host` rows of host bytes (if any)
    /// and the device's `rows` rows of memory.
    pub fn new(scalars: usize, host: Option<usize>, rows: usize) -> Self {
        let mut k = Kernel {
            types: Vec::new(),
            params: Vec::new(),
            scratch: Vec::new(),
            body: Block::default(),
            open: Vec::new(),
        };
        let mut params = vec![Type::Ref(
            TensorType::new(DType::I32, &[scalars]),
            Space::Smem,
        )];
        params.extend(host.map(words));
        params.extend([words(rows), words(rows)]);
        k.params = params.into_iter().map(|t| k.var(t)).collect();
        k
    }

    fn var(&mut self, ty: Type) -> Var {
        self.types.push(ty);
        self.types.len() - 1
    }

    /// The parameters: the scalars, the host rows (if any), the memory as
    /// read and as written.
    pub fn params(&self) -> &[Var] {
        &self.params
    }

    pub fn scratch(&self) -> &[Var] {
        &self.scratch
    }

    pub fn body(&self) -> &Block {
        &self.body
    }

    pub fn type_of(&self, var: Var) -> &Type {
        &self.types[var]
    }

    /// A new scratch buffer or semaphore array.
    pub fn alloc(&mut self, ty: Type) -> Result<Var, String> {
        match ty {
            Type::Ref(_, Space::Vmem) | Type::Semaphore(_) => {}
            _ => return Err("scratch is VMEM or DMA semaphores".into()),
        }
        let v = self.var(ty);
        self.scratch.push(v);
        Ok(v)
    }

    fn check(&self, vars: &[Var]) -> Result<(), String> {
        match vars.iter().find(|&&v| v >= self.types.len()) {
            Some(v) => Err(format!("%{v} is not a value of this kernel")),
            None => Ok(()),
        }
    }

    fn block(&mut self) -> &mut Block {
        match self.open.last_mut() {
            Some(Open::If { block, .. } | Open::For { block, .. }) => block,
            None => &mut self.body,
        }
    }

    /// Appends `op` on `inputs`, of results of `types`.
    fn push(&mut self, op: Op, inputs: &[Var], types: Vec<Type>) -> Vec<Var> {
        let outputs: Vec<Var> = types.into_iter().map(|t| self.var(t)).collect();
        self.block().nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            outputs: outputs.clone(),
        });
        outputs
    }

    fn value(&self, v: Var) -> Result<&TensorType, String> {
        match &self.types[v] {
            Type::Value(t) => Ok(t),
            other => Err(format!("%{v} is not a value but a {other:?}")),
        }
    }

    fn reference(&self, v: Var) -> Result<(&TensorType, Space), String> {
        match &self.types[v] {
            Type::Ref(t, space) => Ok((t, *space)),
            Type::View(t, _) => Ok((t, Space::Any)),
            other => Err(format!("%{v} is not a ref but a {other:?}")),
        }
    }

    /// The value `primitive(inputs...)`.
    pub fn apply(&mut self, primitive: Primitive, inputs: &[Var]) -> Result<Var, String> {
        self.check(inputs)?;
        let args = inputs
            .iter()
            .map(|&v| self.value(v))
            .collect::<Result<Vec<_>, _>>()?;
        let ty = primitive.infer(&args)?;
        Ok(self.push(Op::Prim(primitive), inputs, vec![Type::Value(ty)])[0])
    }

    /// The value `op(inputs...)` of a TPU's own ops.
    pub fn bits(&mut self, op: Op, inputs: &[Var]) -> Result<Var, String> {
        self.check(inputs)?;
        let a = self.value(inputs[0])?.clone();
        let ty = match &op {
            Op::Bitcast(dtype) if dtype.size_of() == a.dtype.size_of() => {
                TensorType::new(*dtype, &a.shape)
            }
            Op::Bitcast(dtype) => {
                return Err(format!("bitcast: {} to {dtype} changes the width", a.dtype));
            }
            Op::ShiftLeft | Op::ShiftRight | Op::And | Op::Or => {
                let b = self.value(*inputs.get(1).ok_or("a bit op takes two values")?)?;
                if *b != a || !(a.dtype.is_int() || a.dtype == DType::Bool) {
                    return Err(format!(
                        "bit ops take two integer values of one type, got {a} and {b}"
                    ));
                }
                a
            }
            Op::MultipleOf(_) if a == TensorType::new(DType::I32, &[]) => a,
            Op::MultipleOf(_) => return Err(format!("multiple_of takes an int32 scalar, got {a}")),
            other => return Err(format!("{other:?} is not a value op")),
        };
        Ok(self.push(op, inputs, vec![Type::Value(ty)])[0])
    }

    /// `src`'s window (see [`Op::Slice`]); `dynamic` are the starts not in
    /// `starts`, in order.
    pub fn slice(
        &mut self,
        src: Var,
        starts: Vec<Option<usize>>,
        dynamic: &[Var],
        sizes: Vec<usize>,
        squeezed: Vec<bool>,
    ) -> Result<Var, String> {
        self.check(&[src])?;
        self.check(dynamic)?;
        let shape = match &self.types[src] {
            Type::Ref(t, _) | Type::View(t, _) => t.shape.clone(),
            Type::Semaphore(shape) => shape.clone(),
            other => return Err(format!("only refs are sliced, not a {other:?}")),
        };
        let rank = shape.len();
        if starts.len() != rank || sizes.len() != rank || squeezed.len() != rank {
            return Err(format!(
                "a slice of a rank-{rank} ref needs {rank} starts and sizes"
            ));
        }
        if starts.iter().filter(|s| s.is_none()).count() != dynamic.len() {
            return Err("a slice needs a dynamic start for each start it does not give".into());
        }
        for (d, ((start, &size), &n)) in starts.iter().zip(&sizes).zip(&shape).enumerate() {
            if size > n || start.is_some_and(|s| s + size > n) {
                return Err(format!(
                    "a slice of {size} at {start:?} of dimension {d} ({n})"
                ));
            }
            if squeezed[d] && size != 1 {
                return Err(format!("dimension {d} is squeezed but of size {size}"));
            }
        }
        // A view's window is a view of whole tiles of it (in its tiled
        // dimensions): its own, from the memory (`tpu.reinterpret_cast`).
        if let Type::View(_, tiled) = &self.types[src] {
            let first = rank - tiled.tile.len().min(rank);
            for d in first..rank {
                let t = tiled.tile[d - first];
                if starts[d].is_some_and(|s| !s.is_multiple_of(t)) || squeezed[d] {
                    return Err(format!(
                        "a window of a tiled view starts on a tile ({t}) of dimension {d}, unsqueezed"
                    ));
                }
            }
        }
        for &v in dynamic {
            if *self.value(v)? != TensorType::new(DType::I32, &[]) {
                return Err("dynamic starts are int32 scalars".into());
            }
        }
        let kept: Vec<usize> = sizes
            .iter()
            .zip(&squeezed)
            .filter_map(|(&n, &s)| (!s).then_some(n))
            .collect();
        let ty = match &self.types[src] {
            Type::Ref(t, space) => Type::Ref(TensorType::new(t.dtype, &kept), *space),
            // A window of a view: its tiles, the strides of its dimensions
            // kept.
            Type::View(t, tiled) => {
                let strides = tiled
                    .strides
                    .iter()
                    .zip(&squeezed)
                    .filter_map(|(&s, &q)| (!q).then_some(s))
                    .collect();
                let tiled = Tiled {
                    tiles: tiled.tiles.clone(),
                    tile: tiled.tile.clone(),
                    strides,
                };
                Type::View(TensorType::new(t.dtype, &kept), tiled)
            }
            _ => Type::Semaphore(kept),
        };
        let inputs: Vec<Var> = std::iter::once(src)
            .chain(dynamic.iter().copied())
            .collect();
        let op = Op::Slice {
            starts,
            sizes,
            squeezed,
        };
        Ok(self.push(op, &inputs, vec![ty])[0])
    }

    /// A `ty` tensor in `memory` (the kernel's, as read or written) at i32
    /// row `row`, in its TPU layout.
    pub fn view(&mut self, memory: Var, row: Var, ty: TensorType) -> Result<Var, String> {
        self.check(&[memory, row])?;
        if !self.params.contains(&memory) || self.types[memory] != words(self.rows()) {
            return Err("a view is of the kernel's memory".into());
        }
        if *self.value(row)? != TensorType::new(DType::I32, &[]) {
            return Err("a view's row is an int32 scalar".into());
        }
        let layout = crate::device::tpu::layout::Layout::new(ty.dtype, &ty.shape)?;
        let (tiles, strides) = layout.tiles();
        let tile = layout.tile_shape();
        let ty = Type::View(
            ty,
            Tiled {
                tiles,
                tile,
                strides,
            },
        );
        Ok(self.push(Op::View, &[memory, row], vec![ty])[0])
    }

    /// The memory's rows.
    fn rows(&self) -> usize {
        match &self.types[*self.params.last().expect("the memory")] {
            Type::Ref(t, _) => t.shape[0],
            _ => unreachable!("the memory is a ref"),
        }
    }

    /// `src`'s elements as `shape`.
    pub fn reshape(&mut self, src: Var, shape: Vec<usize>) -> Result<Var, String> {
        self.check(&[src])?;
        if let Type::View(..) = self.types[src] {
            return Err("a tiled view is not reshaped: its tiles are its shape's".into());
        }
        let (t, space) = self.reference(src)?;
        if t.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
            return Err(format!("a ref of {t} cannot be reshaped to {shape:?}"));
        }
        let ty = Type::Ref(TensorType::new(t.dtype, &shape), space);
        Ok(self.push(Op::Reshape(shape), &[src], vec![ty])[0])
    }

    /// `src`'s bits as `dtype` (of its width).
    pub fn ref_bitcast(&mut self, src: Var, dtype: DType) -> Result<Var, String> {
        self.check(&[src])?;
        if let Type::View(..) = self.types[src] {
            return Err("a tiled view is not bitcast: view the tensor as its dtype".into());
        }
        let (t, space) = self.reference(src)?;
        if dtype.size_of() != t.dtype.size_of() {
            return Err(format!("a ref of {} cannot be read as {dtype}", t.dtype));
        }
        let ty = Type::Ref(TensorType::new(dtype, &t.shape), space);
        Ok(self.push(Op::RefBitcast(dtype), &[src], vec![ty])[0])
    }

    /// A VMEM ref's value, or an SMEM ref's element at i32 `index`.
    pub fn load(&mut self, src: Var, index: Option<Var>) -> Result<Var, String> {
        self.check(&[src])?;
        let (t, space) = self.reference(src)?;
        let t = t.clone();
        let (inputs, ty) = match (space, index) {
            (Space::Vmem, None) => (vec![src], t),
            (Space::Smem, Some(i)) if t.shape.len() == 1 => {
                self.check(&[i])?;
                (vec![src, i], TensorType::new(t.dtype, &[]))
            }
            _ => return Err("loads read a VMEM ref whole, or an SMEM vector's element".into()),
        };
        Ok(self.push(Op::Load, &inputs, vec![Type::Value(ty)])[0])
    }

    /// Stores `value` in VMEM ref `dst`.
    pub fn store(&mut self, dst: Var, value: Var) -> Result<(), String> {
        self.check(&[dst, value])?;
        let (t, space) = self.reference(dst)?;
        let v = self.value(value)?;
        if space != Space::Vmem || t != v {
            return Err(format!(
                "stores write a VMEM ref of the value's type, got {t} and {v}"
            ));
        }
        self.push(Op::Store, &[dst, value], vec![]);
        Ok(())
    }

    /// Starts (or with `wait`, waits for) copying `src` to `dst` with `sem`.
    pub fn dma(&mut self, src: Var, dst: Var, sem: Var, wait: bool) -> Result<(), String> {
        self.check(&[src, dst, sem])?;
        let (s, d) = (self.reference(src)?.0, self.reference(dst)?.0);
        if s != d {
            return Err(format!(
                "a copy is between refs of one type, got {s} and {d}"
            ));
        }
        if self.types[sem] != Type::Semaphore(vec![]) {
            return Err("a copy signals one DMA semaphore".into());
        }
        let op = if wait { Op::DmaWait } else { Op::DmaStart };
        self.push(op, &[src, dst, sem], vec![]);
        Ok(())
    }

    /// Starts an `If` on i1 scalar `cond`: what follows, until
    /// [`end_if`](Self::end_if), runs only if it holds.
    pub fn begin_if(&mut self, cond: Var) -> Result<(), String> {
        self.check(&[cond])?;
        if *self.value(cond)? != TensorType::new(DType::Bool, &[]) {
            return Err("a condition is a bool scalar".into());
        }
        self.open.push(Open::If {
            cond,
            block: Block::default(),
        });
        Ok(())
    }

    pub fn end_if(&mut self) -> Result<(), String> {
        match self.open.pop() {
            Some(Open::If { cond, block }) => {
                self.push(Op::If(block), &[cond], vec![]);
                Ok(())
            }
            other => {
                self.open.extend(other);
                Err("no `If` is open".into())
            }
        }
    }

    /// Starts a `For` over i32 scalars `lo..hi` carrying `carried`: its
    /// index and carried values, as its block sees them.
    pub fn begin_for(
        &mut self,
        lo: Var,
        hi: Var,
        carried: &[Var],
    ) -> Result<(Var, Vec<Var>), String> {
        self.check(&[lo, hi])?;
        self.check(carried)?;
        let i32 = TensorType::new(DType::I32, &[]);
        if *self.value(lo)? != i32 || *self.value(hi)? != i32 {
            return Err("a loop's bounds are int32 scalars".into());
        }
        let types = carried
            .iter()
            .map(|&v| self.value(v).cloned().map(Type::Value))
            .collect::<Result<Vec<_>, _>>()?;
        let index = self.var(Type::Value(i32));
        let args: Vec<Var> = types.into_iter().map(|t| self.var(t)).collect();
        let block = Block {
            args: std::iter::once(index).chain(args.iter().copied()).collect(),
            ..Block::default()
        };
        let inputs = [lo, hi]
            .into_iter()
            .chain(carried.iter().copied())
            .collect();
        self.open.push(Open::For { inputs, block });
        Ok((index, args))
    }

    /// Ends the open `For`, `results` its next carried values: the last.
    pub fn end_for(&mut self, results: &[Var]) -> Result<Vec<Var>, String> {
        self.check(results)?;
        let Some(Open::For { inputs, mut block }) = self.open.pop() else {
            return Err("no `For` is open".into());
        };
        let carried = &block.args[1..];
        let types: Vec<Type> = carried.iter().map(|&v| self.types[v].clone()).collect();
        let given: Vec<Type> = results.iter().map(|&v| self.types[v].clone()).collect();
        if given != types {
            return Err(format!("a loop carries {types:?}, its body gave {given:?}"));
        }
        block.results = results.to_vec();
        Ok(self.push(Op::For(block), &inputs, types))
    }

    /// Whether every `If` and `For` begun has ended.
    pub fn is_closed(&self) -> bool {
        self.open.is_empty()
    }
}
