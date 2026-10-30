//! Encodes reshapes of TPU-tiled tensors that split or merge the last dimension.

use crate::codegen::dot_general::tpu::block;
use crate::codegen::layout::LayoutCodegen;
use crate::codegen::utils::tpu::Tpu;
use crate::device::tpu::layout::Layout;
use crate::graph::{Graph, TensorType};
use crate::graph_compiler::plan::Step;
use crate::kernels::tpu::{Launch, launch};

/// TPU layout: the fixed reshape Pallas program, specialized by a spec of the split or merged last dim;
/// transposes fuse into the elementwise loops.
impl LayoutCodegen for Tpu {
    /// The kernel's name and spec.
    type Code = (String, String);
    type Options = ();

    /// None: a transpose fuses into the elementwise loops.
    fn transpose(_: &Graph, _: &[bool], _: &()) -> Result<(String, String), String> {
        Err("a transpose on TPU fuses into the elementwise loops".into())
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

/// Launches the reshape `step` of TPU-tiled tensors on TPU `index`.
pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let (kernel, spec) = Tpu::reshape(&step.inputs[0].1, &step.output.1)?;
    let rows = [inputs[0], output.cast_const()];

    let launched = Launch {
        module: "lumen.codegen.layout.tpu",
        kernel,
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
        sparse: None,
    };

    launch(index, launched)
}
