use crate::DType;
use crate::graph::{Primitive, TensorType};

pub mod mosaic;
#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
mod tests;

pub type Var = usize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Space {
    Any,
    Vmem,
    Smem,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Value(TensorType),
    Ref(TensorType, Space),
    Semaphore(Vec<usize>),
    View(TensorType, Tiled),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tiled {
    pub tiles: String,
    pub tile: Vec<usize>,
    pub strides: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum Op {
    Prim(Primitive),
    Bitcast(DType),
    View,
    ShiftLeft,
    ShiftRight,
    And,
    Or,
    MultipleOf(usize),
    Slice {
        starts: Vec<Option<usize>>,
        sizes: Vec<usize>,
        squeezed: Vec<bool>,
    },
    Reshape(Vec<usize>),
    RefBitcast(DType),
    Load,
    Store,
    StridedLoad(Vec<usize>),
    StridedStore(Vec<usize>),
    DmaStart,
    DmaWait,
    Delay,
    If {
        then: Block,
        otherwise: Option<Block>,
    },
    While {
        before: Block,
        after: Block,
    },
    For(Block),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<Var>,
    pub outputs: Vec<Var>,
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub args: Vec<Var>,
    pub nodes: Vec<Node>,
    pub results: Vec<Var>,
}

#[derive(Debug)]
enum Open {
    If {
        cond: Var,
        block: Block,
        then: Option<Block>,
    },
    For {
        inputs: Vec<Var>,
        block: Block,
    },
    While {
        inputs: Vec<Var>,
        block: Block,
        before: Option<Block>,
    },
}

#[derive(Debug)]
pub struct Kernel {
    types: Vec<Type>,
    params: Vec<Var>,
    scratch: Vec<Var>,
    body: Block,
    open: Vec<Open>,
}

fn words(rows: usize) -> Type {
    Type::Ref(TensorType::new(DType::U32, &[rows, 1, 128]), Space::Any)
}

impl Kernel {
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

    pub fn alloc(&mut self, ty: Type) -> Result<Var, String> {
        match ty {
            Type::Ref(_, Space::Vmem | Space::Smem) | Type::Semaphore(_) => {}
            _ => return Err("scratch is VMEM, SMEM or DMA semaphores".into()),
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
            Some(Open::If { block, .. } | Open::For { block, .. } | Open::While { block, .. }) => {
                block
            }
            None => &mut self.body,
        }
    }

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

    pub fn apply(&mut self, primitive: Primitive, inputs: &[Var]) -> Result<Var, String> {
        self.check(inputs)?;
        let args = inputs
            .iter()
            .map(|&v| self.value(v))
            .collect::<Result<Vec<_>, _>>()?;
        let ty = primitive.infer(&args)?;
        Ok(self.push(Op::Prim(primitive), inputs, vec![Type::Value(ty)])[0])
    }

    pub fn bits(&mut self, op: Op, inputs: &[Var]) -> Result<Var, String> {
        self.check(inputs)?;
        let a = self.value(inputs[0])?.clone();
        let ty = match &op {
            Op::Bitcast(dtype) if dtype.bits() == a.dtype.bits() => {
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

    fn rows(&self) -> usize {
        match &self.types[*self.params.last().expect("the memory")] {
            Type::Ref(t, _) => t.shape[0],
            _ => unreachable!("the memory is a ref"),
        }
    }

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

    pub fn ref_bitcast(&mut self, src: Var, dtype: DType) -> Result<Var, String> {
        self.check(&[src])?;
        if let Type::View(..) = self.types[src] {
            return Err("a tiled view is not bitcast: view the tensor as its dtype".into());
        }
        let (t, space) = self.reference(src)?;
        if dtype.bits() != t.dtype.bits() {
            return Err(format!("a ref of {} cannot be read as {dtype}", t.dtype));
        }
        let ty = Type::Ref(TensorType::new(dtype, &t.shape), space);
        Ok(self.push(Op::RefBitcast(dtype), &[src], vec![ty])[0])
    }

    pub fn load(&mut self, src: Var, indices: &[Var]) -> Result<Var, String> {
        self.check(&[src])?;
        self.check(indices)?;
        let (t, space) = self.reference(src)?;
        let t = t.clone();
        let ty = match (space, indices.len()) {
            (Space::Vmem, 0) => t,
            (Space::Smem, n) if n == t.shape.len() => TensorType::new(t.dtype, &[]),
            _ => {
                return Err(
                    "loads read a VMEM ref whole, or an SMEM ref's element (an index a dimension)"
                        .into(),
                );
            }
        };
        let inputs: Vec<Var> = std::iter::once(src)
            .chain(indices.iter().copied())
            .collect();
        Ok(self.push(Op::Load, &inputs, vec![Type::Value(ty)])[0])
    }

    pub fn store(&mut self, dst: Var, value: Var, indices: &[Var]) -> Result<(), String> {
        self.check(&[dst, value])?;
        self.check(indices)?;
        let (t, space) = self.reference(dst)?;
        let v = self.value(value)?;
        let fits = match (space, indices.len()) {
            (Space::Vmem, 0) => t == v,
            (Space::Smem, n) => n == t.shape.len() && *v == TensorType::new(t.dtype, &[]),
            _ => false,
        };
        if !fits {
            return Err(format!(
                "stores write a VMEM ref whole, or an SMEM ref's element, of its dtype: got {t} and {v}"
            ));
        }
        let inputs: Vec<Var> = [dst, value]
            .into_iter()
            .chain(indices.iter().copied())
            .collect();
        self.push(Op::Store, &inputs, vec![]);
        Ok(())
    }

    pub fn load_strided(
        &mut self,
        src: Var,
        starts: &[Var],
        sizes: &[usize],
        strides: &[usize],
    ) -> Result<Var, String> {
        let dtype = self.strided(src, starts, sizes, strides)?;
        let ty = Type::Value(TensorType::new(dtype, sizes));
        let inputs: Vec<Var> = std::iter::once(src).chain(starts.iter().copied()).collect();
        Ok(self.push(Op::StridedLoad(strides.to_vec()), &inputs, vec![ty])[0])
    }

    pub fn store_strided(
        &mut self,
        dst: Var,
        value: Var,
        starts: &[Var],
        strides: &[usize],
    ) -> Result<(), String> {
        self.check(&[value])?;
        let v = self.value(value)?.clone();
        let dtype = self.strided(dst, starts, &v.shape, strides)?;
        if v.dtype != dtype {
            return Err(format!("a strided store of {dtype} got {v}"));
        }
        let inputs: Vec<Var> = [dst, value]
            .into_iter()
            .chain(starts.iter().copied())
            .collect();
        self.push(Op::StridedStore(strides.to_vec()), &inputs, vec![]);
        Ok(())
    }

    fn strided(
        &self,
        r: Var,
        starts: &[Var],
        sizes: &[usize],
        strides: &[usize],
    ) -> Result<DType, String> {
        self.check(&[r])?;
        self.check(starts)?;
        let (t, space) = self.reference(r)?;
        let rank = t.shape.len();
        let fits = space == Space::Vmem
            && t.dtype != DType::Bool
            && [starts.len(), sizes.len(), strides.len()] == [rank; 3]
            && t.shape
                .iter()
                .zip(sizes)
                .zip(strides)
                .all(|((n, k), s)| *k == 0 || (k - 1) * s < *n);
        if !fits {
            return Err(
                "strided loads and stores are of a VMEM ref (not bool): a start, size and stride (0: broadcast) a dimension, within it"
                    .into(),
            );
        }
        for &s in starts {
            if *self.value(s)? != TensorType::new(DType::I32, &[]) {
                return Err("a strided access's starts are int32 scalars".into());
            }
        }
        Ok(t.dtype)
    }

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

    pub fn delay(&mut self, nanos: Var) -> Result<(), String> {
        self.check(&[nanos])?;
        if *self.value(nanos)? != TensorType::new(DType::I32, &[]) {
            return Err("a delay is an int32 scalar of nanoseconds".into());
        }
        self.push(Op::Delay, &[nanos], vec![]);
        Ok(())
    }

    pub fn begin_if(&mut self, cond: Var) -> Result<(), String> {
        self.check(&[cond])?;
        if *self.value(cond)? != TensorType::new(DType::Bool, &[]) {
            return Err("a condition is a bool scalar".into());
        }
        self.open.push(Open::If {
            cond,
            block: Block::default(),
            then: None,
        });
        Ok(())
    }

    pub fn begin_else(&mut self, results: &[Var]) -> Result<(), String> {
        self.check(results)?;
        match self.open.last_mut() {
            Some(Open::If { block, then, .. }) if then.is_none() => {
                let mut done = std::mem::take(block);
                done.results = results.to_vec();
                *then = Some(done);
                Ok(())
            }
            _ => Err("no `If` without an else is open".into()),
        }
    }

    pub fn end_if(&mut self, results: &[Var]) -> Result<Vec<Var>, String> {
        self.check(results)?;
        let Some(Open::If { cond, block, then }) = self.open.pop() else {
            return Err("no `If` is open".into());
        };
        let (then, otherwise) = match then {
            Some(then) => {
                let mut otherwise = block;
                otherwise.results = results.to_vec();
                (then, Some(otherwise))
            }
            None if results.is_empty() => (block, None),
            None => return Err("an `If` without an else has no values".into()),
        };
        let types: Vec<Type> = then
            .results
            .iter()
            .map(|&v| self.types[v].clone())
            .collect();
        let others: Vec<Type> = otherwise
            .iter()
            .flat_map(|b| &b.results)
            .map(|&v| self.types[v].clone())
            .collect();
        if otherwise.is_some() && types != others {
            return Err(format!("an `If`'s branches give {types:?} and {others:?}"));
        }
        let op = Op::If { then, otherwise };
        Ok(self.push(op, &[cond], types))
    }

    pub fn begin_while(&mut self, carried: &[Var]) -> Result<Vec<Var>, String> {
        self.check(carried)?;
        let types = carried
            .iter()
            .map(|&v| self.value(v).cloned().map(Type::Value))
            .collect::<Result<Vec<_>, _>>()?;
        let args: Vec<Var> = types.into_iter().map(|t| self.var(t)).collect();
        let block = Block {
            args: args.clone(),
            ..Block::default()
        };
        self.open.push(Open::While {
            inputs: carried.to_vec(),
            block,
            before: None,
        });
        Ok(args)
    }

    pub fn begin_while_body(&mut self, cond: Var) -> Result<Vec<Var>, String> {
        self.check(&[cond])?;
        if *self.value(cond)? != TensorType::new(DType::Bool, &[]) {
            return Err("a loop's condition is a bool scalar".into());
        }
        let Some(Open::While { block, before, .. }) = self.open.last() else {
            return Err("no `While` is open".into());
        };
        if before.is_some() {
            return Err("the `While`'s body has begun".into());
        }
        let types: Vec<Type> = block.args.iter().map(|&v| self.types[v].clone()).collect();
        let args: Vec<Var> = types.into_iter().map(|t| self.var(t)).collect();
        let Some(Open::While { block, before, .. }) = self.open.last_mut() else {
            unreachable!("checked above")
        };
        let mut done = std::mem::take(block);
        done.results = std::iter::once(cond)
            .chain(done.args.iter().copied())
            .collect();
        *before = Some(done);
        block.args = args.clone();
        Ok(args)
    }

    pub fn end_while(&mut self, results: &[Var]) -> Result<Vec<Var>, String> {
        self.check(results)?;
        let Some(Open::While {
            inputs,
            mut block,
            before: Some(before),
        }) = self.open.pop()
        else {
            return Err("no `While` with a body is open".into());
        };
        let types: Vec<Type> = block.args.iter().map(|&v| self.types[v].clone()).collect();
        let given: Vec<Type> = results.iter().map(|&v| self.types[v].clone()).collect();
        if given != types {
            return Err(format!("a loop carries {types:?}, its body gave {given:?}"));
        }
        block.results = results.to_vec();
        let op = Op::While {
            before,
            after: block,
        };
        Ok(self.push(op, &inputs, types))
    }

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

    pub fn is_closed(&self) -> bool {
        self.open.is_empty()
    }
}
