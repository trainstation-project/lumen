//! Metal source for transposes fused with elementwise ops, staged through threadgroup tiles so
//! both the reads and the writes are coalesced.
use std::collections::HashMap;
use std::fmt::Write;

use indoc::formatdoc;

use crate::codegen::emitter::Emit;
use crate::codegen::emitter::mps::Emitter;
use crate::codegen::layout::{LayoutCodegen, TILE};
pub(crate) use crate::codegen::layout::{TransposeTiling, transpose_tiling};
use crate::codegen::store::mps::{gather_index, written};
use crate::codegen::utils::mps::*;
use crate::graph::{Graph, TensorType, Var};

impl TransposeTiling {
    /// Launch grid: one threadgroup per `TILE` x `TILE` tile of each batch.
    pub(crate) fn grid(&self) -> crate::runtime::mps::Grid {
        let batches = self.batch.iter().map(|d| d.0).product();
        crate::runtime::mps::Grid::Groups([self.a.div_ceil(TILE), self.b.div_ceil(TILE), batches])
    }
}

/// Metal layout: tiled transposes fused with elementwise ops; plain layout steps run the fixed strided copy.
impl LayoutCodegen for Mps {
    /// The kernel's name and source.
    type Code = (String, String);
    type Options = HashMap<usize, (usize, Vec<usize>)>;

    /// The tiled transpose kernel: loads each hero transpose's input into a tile at input order, then
    /// computes and stores the outputs reading the tile transposed.
    fn transpose(
        body: &Graph,
        by_value: &[bool],
        views: &HashMap<usize, (usize, Vec<usize>)>,
    ) -> Result<(String, String), String> {
        let out = body.outputs()[0];
        let out_type = metal_type(body.type_of(out).dtype);
        let (params, _) = io_params(body, by_value, out_type, &[]);
        let tiling = transpose_tiling(body).ok_or("a body with no tiled transpose")?;
        let (a, b, sa) = (tiling.a, tiling.b, tiling.a_stride);
        let (sizes, strides): (Vec<usize>, Vec<usize>) = tiling.batch.iter().copied().unzip();
        let base = gather_index("group.z", &sizes, &strides);
        let mut source = String::new();

        for (k, &h) in tiling.heroes.iter().enumerate() {
            let t = metal_type(body.type_of(h).dtype);
            // One column of padding avoids threadgroup memory bank conflicts on the transposed read.
            writeln!(source, "    threadgroup {t} tile{k}[{TILE}][{}];", TILE + 1).unwrap();
        }

        writeln!(source, "    uint base = {base};").unwrap();

        writeln!(
            source,
            "    uint a0 = group.x * {TILE}, b0 = group.y * {TILE};"
        )
        .unwrap();

        let each = |code: &str, x: &str, y: &str| {
            formatdoc!(
                "
                for (uint dy = 0; dy < {TILE}; dy += 16) {{
                    for (uint dx = 0; dx < {TILE}; dx += 16) {{
                        uint {x} = {x}0 + tid.x + dx, {y} = {y}0 + tid.y + dy;
                        if (a >= {a}u || b >= {b}u) {{
                            continue;
                        }}
                        uint j = base + a * {sa}u + b;
                        {code}
                    }}
                }}
                "
            )
        };

        let mut load = Emitter::new(body, by_value);
        let producer = |v: Var| body.nodes().iter().find(|n| n.output == v).unwrap();
        let mut stores = String::new();

        for (k, &h) in tiling.heroes.iter().enumerate() {
            let node = producer(h);
            let i = load.operand_index(node, "j".into());
            let v = load.value(node.inputs[0], i);
            writeln!(stores, "        tile{k}[tid.y + dy][tid.x + dx] = {v};").unwrap();
        }

        source.push_str(&each(&format!("{}{stores}", load.lines), "a", "b"));
        writeln!(source, "threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
        let mut emitter = Emitter::new(body, by_value);

        for (k, &h) in tiling.heroes.iter().enumerate() {
            emitter.invariant[h] = true;

            emitter
                .row_locals
                .insert(h, format!("tile{k}[tid.x + dx][tid.y + dy]"));
        }

        let result = emitter.value(out, "j".into());

        let extra: Vec<String> = body.outputs()[1..]
            .iter()
            .map(|&v| emitter.value(v, "j".into()))
            .collect();

        let (stores, writes) = written(body, views, &result, &extra);
        source.push_str(&each(&format!("{}{writes}", emitter.lines), "b", "a"));

        Ok(named(formatdoc!(
            "
            {stores}
            kernel void NAME({params}uint3 group [[threadgroup_position_in_grid]],
                             uint3 tid [[thread_position_in_threadgroup]]) {{
                {source}
            }}
            "
        )))
    }

    /// None: a reshape runs the fixed strided copy.
    fn reshape(from: &TensorType, to: &TensorType) -> Result<(String, String), String> {
        Err(format!(
            "a reshape of {from} to {to} runs Metal's strided copy"
        ))
    }
}
