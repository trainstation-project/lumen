//! Triton reduction kernels: a fused reduction, optionally split across programs.

use std::fmt::Write;

use indoc::formatdoc;

use super::ReductionCodegen;
use crate::codegen::triton::config::{MOST_SPLITS, SPLIT_BELOW, SPLIT_SPAN};
use crate::codegen::triton::reduction::{Tile, combine, lowest, reduce_along};
use crate::codegen::triton::utils::indented;
use crate::codegen::triton::*;
use crate::graph::{Graph, Node, Primitive};
use crate::tensor::contiguous_strides;

/// A generated reduction kernel and its launch shape.
pub(crate) struct Reduction {
    pub name: String,
    pub source: String,
    /// Reduced elements each output processes per loop iteration.
    pub lanes: usize,
    /// Programs per output; above 1 the kernel writes partials and `NAME_final` combines them.
    pub splits: usize,
}

/// Triton reductions: one kernel a fusion, small output counts splitting the reduced axis across programs
/// and a `NAME_final` kernel combining their partials.
impl ReductionCodegen for Triton {
    /// The kernel with its launch shape.
    type Code = Reduction;

    /// Generates a kernel reducing `root`'s operand (computed by the prologue part of `body`) and applying
    /// the rest of `body` to the result. Small output counts split the reduced axis across programs.
    fn reduction(body: &Graph, by_value: &[bool], root: &Node) -> Result<Reduction, String> {
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
        // Values the prologue computes that other kernels read too (hosted, as MPS writes them) are
        // written where it reads each element, once (by whichever split covers it).
        let extras = body.outputs().len() - 1;
        for (k, &v) in body.outputs()[1..].iter().enumerate() {
            let value = prologue.value(v, "j".into());
            writeln!(
                prologue.lines,
                "    STORE{}(out{}, j, {value}, mask)",
                k + 1,
                k + 1
            )
            .unwrap();
        }
        let outs: Vec<String> = (1..=extras).map(|k| format!("out{k}")).collect();
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
        let (whole, partials) = (
            reduce_along("acc", axis, sum, a, "NAME_combine"),
            reduce_along("partial", 1, sum, a, "NAME_combine"),
        );
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
        // A split's first pass stores only the extras (`STORE` stays unused), the second the result.
        let extra = (extras > 0).then(|| outs.join(",\n    "));
        let kernel_params = params(by_value, extra.as_deref(), 1 + extras);
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
                jo = ({o_at}) + tl.zeros_like(o)
                empty = tl.full([], {init}, {acc})
                acc = tl.full({tile_shape}, {init}, {acc})
                for start in range({range}):
                    r = start + tl.arange(0, {lanes})
                    jr = ({r_at}) + tl.zeros_like(r)
                    j = {j}
                    mask = {mask}
            {lines}        acc = NAME_combine(acc, tl.where(mask, {element}, empty))
                reduced = {whole}
            {finish}"
        );
        // Second pass: each program combines the partials of its outputs, padded to a power of two.
        if splits > 1 {
            let parts = splits.next_power_of_two();
            let reads: Vec<&str> = outs
                .iter()
                .map(String::as_str)
                .chain(["partials"])
                .collect();
            let params = params(by_value, Some(&reads.join(",\n    ")), 1);
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
                    reduced = {partials}
                    mask = o < n
                {tail}    STORE(out, o, {result}, mask)
                "
            );
        }
        let (name, source) = named("fusion", source);
        Ok(Reduction {
            name,
            source,
            lanes,
            splits,
        })
    }
}
