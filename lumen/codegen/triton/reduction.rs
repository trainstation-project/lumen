//! Triton sources for fused reductions (sum/max, optionally split across programs) and cumulative sums.

use super::config::{
    COLUMN_LANES, MOST_SPLITS, ROW_LANES, SPLIT_BELOW, SPLIT_ROW_LANES, SPLIT_SPAN,
};
use super::*;
use crate::graph::TensorType;

/// A generated reduction kernel and its launch shape.
pub(crate) struct Reduction {
    pub name: String,
    pub source: String,
    /// Reduced elements each output processes per loop iteration.
    pub lanes: usize,
    /// Programs per output; above 1 the kernel writes partials and `NAME_final` combines them.
    pub splits: usize,
}

/// The body's `reduce_sum`/`reduce_max` node, if it has one.
pub(crate) fn reduction_root(body: &Graph) -> Option<&Node> {
    body.nodes().iter().find(|n| {
        matches!(
            n.primitive,
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
        )
    })
}

/// 2-D tile a reduction or scan program works on: outputs along one axis, reduced lanes along the other.
struct Tile {
    lanes: usize,
    /// Outputs index rows and lanes index columns (the reduced axis is innermost in memory).
    rows: bool,
}

impl Tile {
    /// Tile for `count` reduced elements; lanes are the next power of two capped per orientation.
    fn new(shape: &[usize], reduced: impl Fn(usize) -> bool, count: usize, split: bool) -> Self {
        // Lanes go along the minor axis when the innermost non-unit dim is reduced, for coalesced loads.
        let rows = shape.iter().rposition(|&n| n != 1).is_none_or(reduced);
        let most = match (rows, split) {
            (true, false) => ROW_LANES,
            (true, true) => SPLIT_ROW_LANES,
            (false, _) => COLUMN_LANES,
        };
        Tile {
            lanes: count.next_power_of_two().min(most),
            rows,
        }
    }

    /// Triton expression for the number of outputs per program.
    fn outputs(&self) -> String {
        format!("BLOCK_SIZE // {}", self.lanes)
    }

    /// Triton shape of the tile.
    fn shape(&self) -> String {
        match self.rows {
            true => format!("[{}, {}]", self.outputs(), self.lanes),
            false => format!("[{}, {}]", self.lanes, self.outputs()),
        }
    }

    /// Tile axis the lanes run along.
    fn axis(&self) -> usize {
        if self.rows { 1 } else { 0 }
    }

    /// Broadcasts an outputs vector and a lanes vector to the tile and combines them with `op`.
    fn join(&self, outer: &str, inner: &str, op: &str) -> String {
        match self.rows {
            true => format!("{outer}[:, None] {op} {inner}[None, :]"),
            false => format!("{inner}[:, None] {op} {outer}[None, :]"),
        }
    }

    /// Broadcasts an outputs vector to the tile.
    fn spread(&self, outer: &str) -> String {
        match self.rows {
            true => format!("{outer}[:, None]"),
            false => format!("{outer}[None, :]"),
        }
    }
}

/// Combine expression of `a` and `b` for a sum or a max; bool max is `or`, float max propagates NaN.
pub(super) fn combine(sum: bool, dtype: DType) -> &'static str {
    match (sum, dtype) {
        (true, _) => "a + b",
        (false, DType::Bool) => "a | b",
        (false, d) if d.is_float() => "tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)",
        (false, _) => "tl.maximum(a, b)",
    }
}

/// Identity of max for `dtype`: `-inf` for floats, the minimum for signed ints, 0 otherwise.
pub(super) fn lowest(dtype: DType) -> String {
    match dtype {
        d if d.is_float() => "float(\"-inf\")".into(),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            format!("-{}", 1u64 << (dtype.bits() - 1))
        }
        _ => "0".into(),
    }
}

/// Indents every line of `lines` by four spaces.
fn indented(lines: &str) -> String {
    lines.lines().map(|l| format!("    {l}\n")).collect()
}

/// Generates a kernel reducing `root`'s operand (computed by the prologue part of `body`) and applying
/// the rest of `body` to the result. Small output counts split the reduced axis across programs.
pub(crate) fn reduction(body: &Graph, by_value: &[bool], root: &Node) -> Reduction {
    let x = body.type_of(root.inputs[0]);
    let a = body.type_of(root.output).dtype;
    let acc = triton_type(a).expect("a Triton dtype");
    let (axes, sum) = match &root.primitive {
        Primitive::ReduceSum { axes, .. } => (axes, true),
        Primitive::ReduceMax { axes } => (axes, false),
        p => unreachable!("{p} is not a reduction"),
    };
    // Flat offsets split into a kept part (`o`) and a reduced part (`r`), summed to address the input.
    let strides = contiguous_strides(&x.shape);
    let kept: Vec<usize> = (0..x.shape.len()).filter(|d| !axes.contains(d)).collect();
    let mut reduced = axes.clone();
    reduced.sort_unstable();
    let shape = |ds: &[usize]| ds.iter().map(|&d| x.shape[d]).collect::<Vec<_>>();
    let stride = |ds: &[usize]| ds.iter().map(|&d| strides[d]).collect::<Vec<_>>();
    let count: usize = shape(&reduced).iter().product();
    let splits = match body.type_of(root.output).numel() < SPLIT_BELOW {
        true => (count / SPLIT_SPAN).clamp(1, MOST_SPLITS),
        false => 1,
    };
    let tile = Tile::new(&x.shape, |d| axes.contains(&d), count, splits > 1);
    let o_at = gather_index("o", &shape(&kept), &stride(&kept));
    let r_at = gather_index("r", &shape(&reduced), &stride(&reduced));

    let mut prologue = Emitter::new(body, by_value);
    let element = prologue.value(root.inputs[0], "j".into());
    let element = match x.dtype == a {
        true => element,
        false => format!("{element}.to({acc})"),
    };
    // The epilogue starts with the reduced value bound at output index `o`.
    let mut epilogue = Emitter::new(body, by_value);
    epilogue.locals = prologue.locals;
    epilogue
        .values
        .insert((root.output, "o".into()), "reduced".into());
    let result = epilogue.value(body.outputs()[0], "o".into());

    let (lanes, outputs, tile_shape, axis) =
        (tile.lanes, tile.outputs(), tile.shape(), tile.axis());
    let (j, mask) = (
        tile.join("jo", "jr", "+"),
        tile.join("(o < n)", &format!("(r < {count})"), "&"),
    );
    let (combine, init) = (combine(sum, a), lowest(a));
    let init = if sum { "0".into() } else { init };
    let (lines, tail) = (indented(&prologue.lines), &epilogue.lines);
    // Each split covers a whole number of lane-sized chunks.
    let per = count.div_ceil(splits).div_ceil(lanes) * lanes;
    // Split programs are numbered split-major: `s` picks the split, the remainder the output block.
    let (block, range, finish) = match splits {
        1 => (
            "tl.program_id(0)".to_string(),
            format!("0, {count}, {lanes}"),
            format!("    mask = o < n\n{tail}    STORE(out, o, {result}, mask)\n"),
        ),
        _ => (
            "tl.program_id(0) % tl.cdiv(n, {outputs})".to_string(),
            format!("s * {per}, tl.minimum((s + 1) * {per}, {count}), {lanes}"),
            format!("    tl.store(out + o * {splits} + s, reduced, mask=o < n)\n"),
        ),
    };
    let block = block.replace("{outputs}", &outputs);
    let kernel_params = params(by_value, None, usize::from(splits == 1));
    let mut source = formatdoc!(
        "
        @triton.jit
        def NAME_combine(a, b):
            return ({combine}).to(a.dtype)


        @triton.jit
        def NAME(
        {kernel_params}):
            s = tl.program_id(0) // tl.cdiv(n, {outputs})
            o = ({block}).to(tl.int64) * ({outputs}) + tl.arange(0, {outputs})
            jo = ({o_at}) + 0 * o
            empty = tl.full([], {init}, {acc})
            acc = tl.full({tile_shape}, {init}, {acc})
            for start in range({range}):
                r = start + tl.arange(0, {lanes})
                jr = ({r_at}) + 0 * r
                j = {j}
                mask = {mask}
        {lines}        acc = NAME_combine(acc, tl.where(mask, {element}, empty))
            reduced = tl.reduce(acc, {axis}, NAME_combine)
        {finish}"
    );
    // Second pass: each program combines the partials of its outputs, padded to a power of two.
    if splits > 1 {
        let parts = splits.next_power_of_two();
        let params = params(by_value, Some("partials"), 1);
        source += &formatdoc!(
            "


            @triton.jit
            def NAME_final(
            {params}):
                o = tl.program_id(0).to(tl.int64) * (BLOCK_SIZE // {parts}) + tl.arange(0, BLOCK_SIZE // {parts})
                p = tl.arange(0, {parts})
                empty = tl.full([], {init}, {acc})
                mask = (o < n)[:, None] & (p < {splits})[None, :]
                partial = tl.load(partials + o[:, None] * {splits} + p[None, :], mask=mask, other=empty)
                reduced = tl.reduce(partial, 1, NAME_combine)
                mask = o < n
            {tail}    STORE(out, o, {result}, mask)
            "
        );
    }
    let (name, source) = named("fusion", source);
    Reduction {
        name,
        source,
        lanes,
        splits,
    }
}

/// Generates an inclusive cumulative sum along `axis` accumulating in `a`, walking the axis backwards
/// when `reverse`. Returns `(name, source, lanes)`.
pub(crate) fn scan(
    x: &TensorType,
    axis: usize,
    reverse: bool,
    a: DType,
) -> (String, String, usize) {
    let acc = triton_type(a).expect("a Triton dtype");
    let length = x.shape[axis];
    let inner: usize = x.shape[axis + 1..].iter().product();
    let tile = Tile::new(&x.shape, |d| d == axis, length, false);
    let (lanes, outputs, axis) = (tile.lanes, tile.outputs(), tile.axis());
    // Offset of each output's line: outer index times the axis span plus the inner index.
    let line = format!("(o // {inner}) * {} + o % {inner}", length * inner);
    let step = match reverse {
        true => format!("({} - k) * {inner}", length.max(1) - 1),
        false => format!("k * {inner}"),
    };
    let (j, mask, carried) = (
        tile.join("jo", "jk", "+"),
        tile.join("(o < n)", &format!("(k < {length})"), "&"),
        tile.spread("carry"),
    );
    let element = match x.dtype == a {
        true => "LOAD0(in0, j, mask)".to_string(),
        false => format!("LOAD0(in0, j, mask).to({acc})"),
    };
    let params = params(&[false], None, 1);
    let source = formatdoc!(
        "
        @triton.jit
        def NAME_combine(a, b):
            return (a + b).to(a.dtype)


        @triton.jit
        def NAME(
        {params}):
            o = tl.program_id(0).to(tl.int64) * ({outputs}) + tl.arange(0, {outputs})
            jo = {line}
            carry = tl.zeros([{outputs}], {acc})
            for start in range(0, {length}, {lanes}):
                k = start + tl.arange(0, {lanes})
                jk = {step}
                j = {j}
                mask = {mask}
                x = {element}
                x = tl.where(mask, x, tl.zeros_like(x))
                STORE(out, j, {carried} + tl.associative_scan(x, {axis}, NAME_combine), mask)
                carry = carry + tl.reduce(x, {axis}, NAME_combine)
        "
    );
    let (name, source) = named("scan", source);
    (name, source, lanes)
}
