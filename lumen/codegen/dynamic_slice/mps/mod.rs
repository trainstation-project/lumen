//! Metal dynamic slices: the dynamic-slice and dynamic-update-slice kernels and their launches, the copy
//! an out-of-place update starts from, and the source of a fused chain of dynamic-update-slices.

use crate::codegen::utils::mps::Mps;
use crate::runtime::kernel::Launch as _;
use crate::runtime::kernel::mps::MpsLaunch;
use std::fmt::Write;

use indoc::formatdoc;

use super::DynamicSliceCodegen;
use crate::Tensor;
use crate::codegen::utils::mps::*;
use crate::graph::{Graph, TensorType, UPDATES};
use crate::graph_compiler::passes::updates::{overlaps, starts};
use crate::runtime::mps::{elementwise_grid, u32_arg};
use crate::tensor::contiguous_strides;

/// Copies the `ty`-shaped tensor at `src` to `dst` on the GPU.
pub(crate) fn copy(
    src: *const u8,
    dst: *mut u8,
    ty: &crate::graph::TensorType,
    keep: Vec<Tensor>,
    name: &'static str,
) -> Result<(), String> {
    let n = ty.numel();
    let kernel = format!("dynamic_slice_copy_{}", ty.dtype.size_of());
    let grid = elementwise_grid(n, ty.dtype);

    Mps::launch(MpsLaunch {
        kernel: &kernel,
        buffers: &[src, dst.cast_const()],
        args: &[u32_arg(n as u32)],
        grid,
        keep,
        name,
    })
}

/// Highest operand rank the Metal kernels take.
pub(crate) const MAX_RANK: usize = 8;

/// Metal dynamic slices: a generated kernel for a chain of updates; one update runs Metal's fixed kernel.
impl DynamicSliceCodegen for Mps {
    /// The kernel's name and source.
    type Code = (String, String);
    type Options = ();

    /// The kernel of `updates` fusion `body`: every update (inputs `1..`, one shape) written into `out`
    /// (already holding operand `in0`) at its constant start indices, clamped so it stays in bounds; thread
    /// `t` writes element `t` of every update. Its name is `UPDATES`-prefixed.
    fn updates(body: &Graph, _: &()) -> Result<(String, String), String> {
        let x = body.type_of(body.inputs()[0]);
        let update = body.type_of(body.inputs()[1]);
        let t = metal_type(x.dtype);

        let (spans, strides) = (
            contiguous_strides(&update.shape),
            contiguous_strides(&x.shape),
        );

        let rank = x.shape.len();

        // Element `j`'s coordinates in the update, then its offset in the operand before a start is added.
        let mut code: String = (0..rank)
            .filter(|&d| update.shape[d] != 1)
            .map(|d| {
                format!(
                    "const ulong c{d} = (j / {}ul) % {}ul;\n",
                    spans[d], update.shape[d]
                )
            })
            .collect();

        let coordinate = |d: usize| match update.shape[d] {
            1 => "0ul".to_string(),
            _ => format!("c{d}"),
        };

        let terms: Vec<String> = (0..rank)
            .filter(|&d| update.shape[d] != 1)
            .map(|d| format!("c{d} * {}ul", strides[d]))
            .collect();

        let at = match terms.is_empty() {
            true => "0ul".to_string(),
            false => terms.join(" + "),
        };

        writeln!(code, "const ulong at = {at};").unwrap();
        let starts: Vec<Vec<i64>> = starts(body).into_iter().map(|(_, s)| s).collect();
        let chain = overlaps(&x.shape, &update.shape, &starts);

        let mut params: Vec<String> = (0..=chain.len())
            .map(|k| format!("device const {t} *in{k} [[buffer({k})]]"))
            .collect();

        params.push(format!("device {t} *out [[buffer({})]]", chain.len() + 1));

        for (k, (start, later)) in chain.iter().enumerate() {
            let offset: usize = (0..rank).map(|d| start[d] * strides[d]).sum();

            // Elements a later overlapping update writes over are left to it.
            let covered: Vec<String> = later
                .iter()
                .map(|&l| {
                    let inside: Vec<String> = (0..rank)
                        .filter(|&d| update.shape[d] != 1 || start[d] != chain[l].0[d])
                        .map(|d| {
                            let (c, s, t) = (coordinate(d), start[d], chain[l].0[d]);

                            format!(
                                "{c} + {s}ul >= {t}ul && {c} + {s}ul < {}ul",
                                t + update.shape[d]
                            )
                        })
                        .collect();

                    match inside.is_empty() {
                        true => "true".to_string(),
                        false => format!("({})", inside.join(" && ")),
                    }
                })
                .collect();

            let store = format!("out[at + {offset}ul] = in{}[j];", k + 1);

            match covered.is_empty() {
                true => writeln!(code, "{store}"),
                false => writeln!(code, "if (!({})) {{ {store} }}", covered.join(" || ")),
            }
            .unwrap();
        }

        let (count, params) = (update.numel(), params.join(",\n"));

        let (name, source) = named(formatdoc!(
            "
            kernel void NAME({params},
                             uint t [[thread_position_in_grid]]) {{
                if (t >= {count}u) {{
                    return;
                }}
                const ulong j = t;
                {code}
            }}
            "
        ));

        let renamed = name.replacen("fusion", UPDATES, 1);
        let source = source.replace(&name, &renamed);
        Ok((renamed, source))
    }

    /// None: one update runs the fixed `dynamic_update_slice` kernel.
    fn update(
        operand: &TensorType,
        _: &TensorType,
        _: &[bool],
        _: &(),
    ) -> Result<(String, String), String> {
        Err(format!(
            "a dynamic_update_slice into {operand} runs Metal's fixed kernel"
        ))
    }
}
