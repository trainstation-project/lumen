//! Triton sources for fused reductions (sum/max, optionally split across programs) and cumulative sums.

use super::config::{COLUMN_LANES, ROW_LANES, SPLIT_ROW_LANES};
use super::*;
use crate::graph::TensorType;

pub(crate) use crate::codegen::kinds::reduction_root;

/// 2-D tile a reduction or scan program works on: outputs along one axis, reduced lanes along the other.
pub(crate) struct Tile {
    pub(crate) lanes: usize,
    /// Outputs index rows and lanes index columns (the reduced axis is innermost in memory).
    pub(crate) rows: bool,
}

impl Tile {
    /// Tile for `count` reduced elements; lanes are the next power of two capped per orientation.
    pub(crate) fn new(
        shape: &[usize],
        reduced: impl Fn(usize) -> bool,
        count: usize,
        split: bool,
    ) -> Self {
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
    pub(crate) fn outputs(&self) -> String {
        format!("BLOCK_SIZE // {}", self.lanes)
    }

    /// Triton shape of the tile.
    pub(crate) fn shape(&self) -> String {
        match self.rows {
            true => format!("[{}, {}]", self.outputs(), self.lanes),
            false => format!("[{}, {}]", self.lanes, self.outputs()),
        }
    }

    /// Tile axis the lanes run along.
    pub(crate) fn axis(&self) -> usize {
        if self.rows { 1 } else { 0 }
    }

    /// Broadcasts an outputs vector and a lanes vector to the tile and combines them with `op`.
    pub(crate) fn join(&self, outer: &str, inner: &str, op: &str) -> String {
        match self.rows {
            true => format!("{outer}[:, None] {op} {inner}[None, :]"),
            false => format!("{inner}[:, None] {op} {outer}[None, :]"),
        }
    }

    /// Broadcasts an outputs vector to the tile.
    pub(crate) fn spread(&self, outer: &str) -> String {
        match self.rows {
            true => format!("{outer}[:, None]"),
            false => format!("{outer}[None, :]"),
        }
    }
}

/// Combine expression of `a` and `b` for a sum or a max; bool max is `or`, float max propagates NaN.
pub(crate) fn combine(sum: bool, dtype: DType) -> &'static str {
    match (sum, dtype) {
        (true, _) => "a + b",
        (false, DType::Bool) => "a | b",
        (false, d) if d.is_float() => "tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)",
        (false, _) => "tl.maximum(a, b)",
    }
}

/// Expression reducing `x` along `axis` for a sum or a max accumulating in `dtype`: `tl.sum` (in
/// `dtype`) or `tl.max`, or `tl.reduce` with `combine` where those differ from the reduction (a float
/// max propagates NaN, a bool max is `or`).
pub(crate) fn reduce_along(x: &str, axis: usize, sum: bool, dtype: DType, combine: &str) -> String {
    let t = triton_type(dtype).expect("a Triton dtype");
    match (sum, dtype) {
        (true, d) if d != DType::Bool => format!("tl.sum({x}, {axis}, dtype={t})"),
        (false, d) if !d.is_float() && d != DType::Bool => format!("tl.max({x}, {axis})"),
        _ => format!("tl.reduce({x}, {axis}, {combine})"),
    }
}

/// Identity of max for `dtype`: `-inf` for floats, the minimum for signed ints, 0 otherwise.
pub(crate) fn lowest(dtype: DType) -> String {
    match dtype {
        d if d.is_float() => "float(\"-inf\")".into(),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            format!("-{}", 1u64 << (dtype.bits() - 1))
        }
        _ => "0".into(),
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
    let total = reduce_along("x", axis, true, a, "NAME_combine");
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
                carry = carry + {total}
        "
    );
    let (name, source) = named("scan", source);
    (name, source, lanes)
}
