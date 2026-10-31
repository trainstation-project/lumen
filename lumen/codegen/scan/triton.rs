//! Triton code generation for `Cumsum`: a tile of lines scanned in chunks along the axis, each chunk's
//! total carried into the next.

use indoc::formatdoc;

use super::ScanCodegen;
use crate::DType;
use crate::codegen::reduction::triton::{Tile, reduce_along};
use crate::codegen::utils::triton::{Triton, named, params, triton_type};
use crate::graph::TensorType;

impl ScanCodegen for Triton {
    /// The kernel's name and source, and the lanes along the axis each program scans at a time.
    type Code = (String, String, usize);

    /// A tile of lines scanned `lanes` elements at a time, each chunk's total carried into the next.
    fn scan(
        x: &TensorType,
        axis: usize,
        reverse: bool,
        a: DType,
    ) -> Result<(String, String, usize), String> {
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
        Ok((name, source, lanes))
    }
}
