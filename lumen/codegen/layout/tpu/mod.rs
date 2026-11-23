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
    /// back; other reshapes would move elements between tiles and are rejected.
    fn reshape(from: &TensorType, to: &TensorType) -> Result<(String, String), String> {
        let (f, t) = (&from.shape, &to.shape);

        let failed = || {
            Err(format!(
                "reshape on TPU: {from} to {to} moves elements between tiles other than by splitting or merging its last dimension"
            ))
        };

        let split = t.len() == f.len() + 1;

        let (flat, tiled) = match split {
            true => (f, t),
            false => (t, f),
        };

        if tiled.len() != flat.len() + 1 || tiled.len() < 3 {
            return failed();
        }

        let r = flat.len();
        let (a, w) = (tiled[r - 1], tiled[r]);

        if flat[..r - 1] != tiled[..r - 1] || flat[r - 1] != a * w {
            return failed();
        }

        let m = flat[r - 2];
        let lead: usize = flat[..r - 2].iter().product();
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

    let Some(&Primitive::Concatenate { dimension, .. }) =
        body.nodes().first().map(|n| &n.primitive)
    else {
        unreachable!("a concatenate fusion of one concatenation")
    };

    let out = &step.output.1;
    let rank = out.shape.len();

    if rank < 2 {
        return Err(format!(
            "a concatenation into {out} of fewer than 2 dimensions does not run on TPU yet"
        ));
    }

    let layout = Layout::new(out.dtype, &out.shape)?;
    let sizes: Vec<usize> = step
        .inputs
        .iter()
        .map(|(_, t)| t.shape[dimension])
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
        "\"kind\": \"{kind}\", \"shape\": {:?}, \"dimension\": {dimension}, \"sizes\": {sizes:?}, \"block\": {block}, \"dtype\": \"{}\"",
        out.shape,
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
