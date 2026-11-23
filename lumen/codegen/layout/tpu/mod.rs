//! Encodes reshapes of TPU-tiled tensors that split or merge the last dimension, and concatenations copied into
//! their result.

use crate::codegen::devices::tpu::Tpu;
use crate::codegen::devices::tpu::{Kernel, Rows};
use crate::codegen::dot_general::tpu::block;
use crate::codegen::layout::LayoutCodegen;
use crate::device::tpu::layout::Layout;
use crate::graph::{Graph, Primitive, TensorType};
use crate::plan::StepBase;

/// TPU layout: the fixed reshape Pallas program, specialized by a spec of the split or merged last dim;
/// transposes fuse into the elementwise kernels.
impl LayoutCodegen for Tpu {
    /// The kernel's name and spec.
    type Code = (String, String);
    type Options = ();

    /// None: a transpose fuses into the elementwise kernels.
    fn transpose(_: &Graph, _: &[bool], _: &()) -> Result<(String, String), String> {
        Err("a transpose on TPU fuses into the elementwise kernels".into())
    }

    /// The program reshaping `from` into `to` by splitting its last dim `a * w` into `[a, w]` or merging it
    /// back: the flat side `[lead, m, a * w]` and the tiled side `[lead * m, a, w]` (each side's leading dims
    /// merged, so size-1 dims may come and go); other reshapes would move elements between tiles and are rejected.
    fn reshape(from: &TensorType, to: &TensorType) -> Result<(String, String), String> {
        let (f, t) = (&from.shape, &to.shape);

        let failed = || {
            Err(format!(
                "reshape on TPU: {from} to {to} moves elements between tiles other than by splitting or merging its last dimension"
            ))
        };

        // `(lead, m, a, w)` when `flat` is `[lead.., m, a * w]` and `tiled` is `[lead * m.., a, w]`.
        let forms = |flat: &[usize], tiled: &[usize]| {
            let (r, s) = (flat.len(), tiled.len());

            if r < 2 || s < 2 {
                return None;
            }

            let (a, w) = (tiled[s - 2], tiled[s - 1]);
            let (m, lead) = (flat[r - 2], flat[..r - 2].iter().product::<usize>());
            let rows = tiled[..s - 2].iter().product::<usize>();
            (flat[r - 1] == a * w && rows == lead * m).then_some((lead, m, a, w))
        };

        // Splitting `from`'s last dim, else merging into `to`'s (with `a` 1, rows moving into slabs of their own).
        let (split, (lead, m, a, w)) = match (forms(f, t), forms(t, f)) {
            (Some(x), _) => (true, x),
            (_, Some(x)) => (false, x),
            _ => return failed(),
        };

        let rows = Layout::new(from.dtype, &[1, 1])?.rows;

        let bm = block(m, rows, 256)
            .ok_or_else(|| format!("reshape on TPU: {m} rows have no block of whole tiles"))?;

        let spec = format!(
            "\"split\": {split}, \"lead\": {lead}, \"m\": {m}, \"a\": {a}, \"w\": {w}, \"bm\": {bm}, \"dtype\": \"{}\"",
            from.dtype.full_name()
        );

        Ok((
            format!("reshape_{}_{a}x{w}", if split { "split" } else { "merge" }),
            spec,
        ))
    }
}

/// The kernel of the reshape `step` of TPU-tiled tensors.
pub(crate) fn kernel(step: &StepBase) -> Result<Kernel, String> {
    let (kernel, spec) = Tpu::reshape(&step.inputs[0].1, &step.output.1)?;

    Ok(Kernel::new(
        "lumen.codegen.layout.tpu.kernel",
        kernel,
        spec,
        Rows::Step,
    ))
}

/// Most bytes of one block of a concatenation's result its vector kernel stages (inputs and result double
/// buffered take four times this).
const CONCAT_BYTES: usize = 2 << 20;

/// The kernel of the concatenation `step` (a `concatenate` fusion of one copying concatenation) of TPU-tiled
/// tensors: DMAs of each operand into its slot when every slot starts on a tile (a leading axis, or whole
/// tiles of rows or columns before the last operand), else blocks of the result built from its operands' blocks
/// (`lanes` along the last axis, `rows` along the one before).
pub(crate) fn concatenate(step: &StepBase) -> Result<Kernel, String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("a concatenate fusion")
    };

    let Some(node) = body
        .nodes()
        .iter()
        .find(|n| matches!(n.primitive, Primitive::Concatenate { .. }))
    else {
        unreachable!("a concatenate fusion of one concatenation")
    };

    let Primitive::Concatenate { dimension, .. } = node.primitive else {
        unreachable!("a concatenation")
    };

    // Each operand's step input (a value concatenated twice is one input).
    let operands: Vec<usize> = node
        .inputs
        .iter()
        .map(|&v| body.inputs().iter().position(|&u| u == v))
        .collect::<Option<_>>()
        .ok_or("a concatenation on TPU of other than its fusion's inputs")?;

    let out = &step.output.1;
    let rank = out.shape.len();

    if rank < 2 {
        return Err(format!(
            "a concatenation into {out} of fewer than 2 dimensions does not run on TPU yet"
        ));
    }

    let layout = Layout::new(out.dtype, &out.shape)?;
    let sizes: Vec<usize> = operands
        .iter()
        .map(|&k| step.inputs[k].1.shape[dimension])
        .collect();
    let (_, m, n) = layout.padded();
    let bytes = layout.stored.size_of();

    // Every slot but the last fills whole tiles, so each starts on one.
    let unit = match rank - dimension {
        1 => 128,
        2 => layout.rows,
        _ => 1,
    };

    let aligned = sizes[..sizes.len() - 1]
        .iter()
        .all(|s| s.is_multiple_of(unit));

    let (kind, block) = match (aligned, rank - dimension) {
        (true, _) => ("dma", 0),
        (false, 1) => {
            let cap = (CONCAT_BYTES / (n * bytes) / layout.rows * layout.rows).max(layout.rows);
            (
                "lanes",
                block(out.shape[rank - 2], layout.rows, cap)
                    .ok_or("a concatenation on TPU: its rows have no block of whole tiles")?,
            )
        }
        _ => {
            let cap = (CONCAT_BYTES / (m * bytes) / 128 * 128).max(128);
            (
                "rows",
                block(out.shape[rank - 1], 128, cap)
                    .ok_or("a concatenation on TPU: its columns have no block of whole tiles")?,
            )
        }
    };

    let spec = format!(
        "\"kind\": \"{kind}\", \"shape\": {:?}, \"dimension\": {dimension}, \"sizes\": {sizes:?}, \"operands\": {operands:?}, \"inputs\": {}, \"block\": {block}, \"dtype\": \"{}\"",
        out.shape,
        step.inputs.len(),
        out.dtype.full_name()
    );

    let mut kernel = Kernel::new(
        "lumen.codegen.layout.tpu.concatenate",
        format!("concatenate_{kind}_{}x{}", sizes.len(), out.dtype),
        spec,
        Rows::Step,
    );

    kernel.context = Some(step.label.to_string());
    Ok(kernel)
}

/// Most bytes of the tile-aligned window one step of the slice kernel stages.
const SLICE_BYTES: usize = 8 << 20;

/// The kernel of the lone slice `step` of a TPU-tiled tensor whose rows or columns start inside a tile (no elementwise
/// plan reads it in whole tiles): per slab of the result, a DMA of the tile-aligned window of rows and columns the
/// slice lies in, the slice's rows and columns then cut from it in registers.
pub(crate) fn slice(step: &StepBase) -> Result<Kernel, String> {
    let Primitive::Slice { start_indices, .. } = &step.primitive else {
        unreachable!("a slice")
    };

    let (x, out) = (&step.inputs[0].1, &step.output.1);
    let rank = out.shape.len();

    if rank < 2 {
        return Err(format!("a slice of {x} to fewer than 2 dimensions does not run on TPU yet"));
    }

    let rows = Layout::new(x.dtype, &x.shape)?.rows;

    // The window `[first, first + size)` of whole `unit`s holding `[start, start + n)` (reaching past the axis's
    // end into its last tile's padding, as the kernel's blocks pad it).
    let window = |start: usize, n: usize, unit: usize| {
        let first = start / unit * unit;
        (first, (start + n).div_ceil(unit) * unit - first)
    };

    let (r, c) = (start_indices[rank - 2], start_indices[rank - 1]);
    let (m, n) = (out.shape[rank - 2], out.shape[rank - 1]);
    let (r0, rw) = window(r, m, rows);
    let (c0, cw) = window(c, n, 128);
    let padded = rw.div_ceil(rows) * rows * cw.div_ceil(128) * 128 * x.dtype.size_of();

    if padded > SLICE_BYTES {
        return Err(format!("a slice of {x} to {out} with a window wider than {SLICE_BYTES} bytes does not run on TPU yet"));
    }

    let spec = format!(
        "\"shape\": {:?}, \"out\": {:?}, \"starts\": {:?}, \"window\": [{r0}, {rw}, {c0}, {cw}], \"dtype\": \"{}\"",
        x.shape,
        out.shape,
        start_indices,
        x.dtype.full_name()
    );

    let mut kernel = Kernel::new(
        "lumen.codegen.layout.tpu.slice",
        format!("slice_{}_{rw}x{cw}", x.dtype),
        spec,
        Rows::Step,
    );

    kernel.context = Some(step.label.to_string());
    Ok(kernel)
}
