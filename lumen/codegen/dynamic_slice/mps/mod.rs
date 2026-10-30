//! Metal dynamic slices: the dynamic-slice and dynamic-update-slice kernels and their launches, the copy
//! an out-of-place update starts from, and the source of a fused chain of dynamic-update-slices.

use std::fmt::Write;

use indoc::formatdoc;

use super::DynamicSliceCodegen;
use crate::Tensor;
use crate::codegen::mps::*;
use crate::graph::Primitive;
use crate::graph::{Graph, TensorType, UPDATES};
use crate::graph_compiler::common::updates::{overlaps, starts};
use crate::graph_compiler::plan::Step;
use crate::kernels::mps::{Grid, dims_arg, elementwise_grid, launch, u32_arg, u64_arg};
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

    launch(
        &kernel,
        &[src, dst.cast_const()],
        &[u32_arg(n as u32)],
        grid,
        keep,
        name,
    )
}

/// Highest operand rank the Metal kernels take.
const MAX_RANK: usize = 8;

/// Encodes a `DynamicSlice` or `DynamicUpdateSlice` step; an out-of-place update first copies
/// the operand into the output.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let x = &step.inputs[0].1;
    let rank = x.shape.len();

    if rank > MAX_RANK {
        return Err(format!(
            "{}: {rank} dimensions are more than its MPS kernels take ({MAX_RANK})",
            step.label
        ));
    }

    let bytes = x.dtype.size_of();

    let (source, block, first) = match &step.primitive {
        Primitive::DynamicSlice { slice_sizes } => (inputs[0], slice_sizes.clone(), 1),
        Primitive::DynamicUpdateSlice => (inputs[1], step.inputs[1].1.shape.clone(), 2),
        _ => unreachable!("a dynamic slice"),
    };

    let update = matches!(step.primitive, Primitive::DynamicUpdateSlice);

    if update && output.cast_const() != inputs[0] {
        copy(inputs[0], output, x, keep.clone(), step.label)?;
    }

    let index_dtype = step
        .inputs
        .get(first)
        .map_or("i32".into(), |(_, t)| t.dtype.to_string());

    // Unused index slots point at the source buffer; the kernel only reads the first `rank`.
    let mut buffers = vec![source, output.cast_const()];
    buffers.extend((0..MAX_RANK).map(|d| *inputs.get(first + d).unwrap_or(&source)));
    let count: usize = block.iter().product();

    let args = [
        dims_arg(x.shape.clone()),
        dims_arg(block.clone()),
        u32_arg(rank as u32),
        u64_arg(count),
    ];

    // Use 32-bit offsets when the operand fits, 64-bit otherwise.
    let width = match x.numel() <= u32::MAX as usize {
        true => 32,
        false => 64,
    };

    let kernel = match update {
        true => format!("dynamic_update_slice_{bytes}_{index_dtype}_{width}"),
        false => format!("dynamic_slice_{bytes}_{index_dtype}_{width}"),
    };

    let grid = Grid::Threads([count.div_ceil(16 / bytes), 1, 1]);
    launch(&kernel, &buffers, &args, grid, keep, step.label)
}

/// Metal dynamic slices: a generated kernel for a chain of updates; one update runs Metal's fixed kernel.
impl DynamicSliceCodegen for Mps {
    /// The kernel's name and source.
    type Code = (String, String);

    /// The kernel of `updates` fusion `body`: every update (inputs `1..`, one shape) written into `out`
    /// (already holding operand `in0`) at its constant start indices, clamped so it stays in bounds; thread
    /// `t` writes element `t` of every update. Its name is `UPDATES`-prefixed.
    fn updates(body: &Graph) -> Result<(String, String), String> {
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
    ) -> Result<(String, String), String> {
        Err(format!(
            "a dynamic_update_slice into {operand} runs Metal's fixed kernel"
        ))
    }
}
