//! Triton layout: transposes fused with elementwise ops, a tile of the output a program so both the reads
//! and the writes are coalesced, as the `BODY` of the fixed `transpose_kernel` (`kernel.py`); reshapes
//! are views the elementwise kernels read through.

use super::{LayoutCodegen, transpose_tiling};
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::triton::Emitter;
use crate::codegen::store::triton::gather_index;
use crate::codegen::utils::triton::{Tensors, Triton, body_function};
use crate::graph::{Graph, Primitive, TensorType};
use crate::tensor::contiguous_strides;

/// Triton layout: a tiled transpose kernel; reshapes are views folded into the elementwise kernels.
impl LayoutCodegen for Triton {
    /// The `BODY` function's name and Python source.
    type Code = (String, String);
    type Options = Tensors;

    /// The tiled transpose's `BODY(EXTRA, BLOCK_ID_Z, BLOCK_A, BLOCK_B, MASK)`: the outputs on the tile of
    /// `transpose_kernel`'s program, rows `BLOCK_A` of the merged dims `a` (contiguous in each transposed
    /// input) by columns `BLOCK_B` of the merged innermost dims `b` (contiguous in the output), of batch
    /// `BLOCK_ID_Z`, loaded and stored through helpers for the tensors' `layouts`. Each hero
    /// transpose's input is loaded at addresses contiguous along `BLOCK_A` and the outputs stored along
    /// `BLOCK_B`, so both are coalesced: Triton stages the tile between the two layouts through shared
    /// memory. The first output is `out` stored through `STORE`; output `k` after it is `out{k}`, stored
    /// through `STORE{k}`; inputs flagged in `by_value` arrive as scalars.
    fn transpose(
        body: &Graph,
        by_value: &[bool],
        layouts: &Tensors,
    ) -> Result<(String, String), String> {
        let tiling = transpose_tiling(body).ok_or("a body with no tiled transpose")?;
        let outputs = body.outputs().len();
        let sa = tiling.a_stride;
        let [_, (_, b1)] = tiling.runs;
        let sizes: Vec<usize> = tiling.batch.iter().map(|d| d.0).collect();
        let strides: Vec<usize> = tiling.batch.iter().map(|d| d.1).collect();
        let base = gather_index("BLOCK_ID_Z", &sizes, &strides);
        let mut emitter = Emitter::new(body, by_value);
        emitter.mask = "MASK".into();

        // Each hero's input read at the tile's positions: its `a` dims are contiguous in it, so the
        // addresses run along `BLOCK_A`.
        for (k, &h) in tiling.heroes.iter().enumerate() {
            let node = body
                .nodes()
                .iter()
                .find(|n| n.output == h)
                .expect("a hero transpose");

            let Primitive::Transpose { permutation } = &node.primitive else {
                unreachable!("a hero transpose")
            };

            let xs = contiguous_strides(&body.type_of(node.inputs[0]).shape);
            let stride = |d: usize| xs[permutation[d]];
            let batch: Vec<usize> = tiling.batch_dims.iter().map(|&d| stride(d)).collect();
            let start = gather_index("BLOCK_ID_Z", &sizes, &batch);

            let at = format!(
                "{start} + BLOCK_A[:, None] + BLOCK_B[None, :] * {}",
                stride(b1)
            );

            let value = emitter.value(node.inputs[0], at);
            emitter.lines.push_str(&format!("    TILE_{k} = {value}\n"));

            emitter
                .values
                .insert((h, "BLOCK".into()), format!("TILE_{k}"));
        }

        let stores: String = (0..outputs)
            .map(|k| {
                let result = emitter.value(body.outputs()[k], "BLOCK".into());

                match k {
                    0 => format!("    STORE(out, BLOCK, {result}, MASK)\n"),
                    k => format!("    STORE{k}(out{k}, BLOCK, {result}, MASK)\n"),
                }
            })
            .collect();

        let lines = std::mem::take(&mut emitter.lines);

        let code = format!(
            "    BLOCK = {base} + BLOCK_A[:, None] * {sa} + BLOCK_B[None, :]\n{lines}{stores}"
        );

        Ok(body_function(
            "transpose",
            layouts,
            "BLOCK_ID_Z, BLOCK_A, BLOCK_B, MASK",
            &code,
        ))
    }

    /// None: a reshape is a view the elementwise kernels read through.
    fn reshape(from: &TensorType, to: &TensorType) -> Result<(String, String), String> {
        Err(format!(
            "a reshape of {from} to {to} on Triton is a view the elementwise kernels read through"
        ))
    }
}
