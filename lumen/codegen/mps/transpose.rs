//! Metal source for transposes fused with elementwise ops, staged through threadgroup tiles so
//! both the reads and the writes are coalesced.
use super::*;

/// How a body's output is tiled for a transpose kernel.
pub(crate) struct TransposeTiling {
    /// Transpose nodes read into threadgroup tiles, all sharing the same tiling.
    heroes: Vec<Var>,
    /// Output elements along the merged dims that are contiguous in the transposed input.
    a: usize,
    /// Output elements along the merged innermost dims, contiguous in the output.
    b: usize,
    /// Output stride of the `a` dims.
    a_stride: usize,
    /// `(size, output stride)` of each remaining dim, flattened into the grid's z axis.
    batch: Vec<(usize, usize)>,
}

impl TransposeTiling {
    /// Launch grid: one threadgroup per `TILE` x `TILE` tile of each batch.
    pub(crate) fn grid(&self) -> crate::kernels::mps::Grid {
        let batches = self.batch.iter().map(|d| d.0).product();
        crate::kernels::mps::Grid::Groups([self.a.div_ceil(TILE), self.b.div_ceil(TILE), batches])
    }
}

/// Side of the square tile one threadgroup transposes.
pub(super) const TILE: usize = 32;

/// Smallest extent on both tiled sides worth staging through a tile.
pub(super) const MIN_TILED: usize = 16;

/// Most transposes one kernel stages, each needing its own threadgroup tile.
pub(crate) const MAX_HEROES: usize = 4;

/// Tiling for a body whose output is computed elementwise from transposes that swap a contiguous
/// input axis with the output's innermost axis, or `None` if no transpose qualifies.
pub(crate) fn transpose_tiling(body: &Graph) -> Option<TransposeTiling> {
    use Primitive::*;
    if gemm_dot(body).is_some()
        || reduction_root(body).is_some()
        || !row_reductions(body).is_empty()
    {
        return None;
    }
    let out = body.type_of(body.outputs()[0]);
    if out.numel() > u32::MAX as usize {
        return None;
    }
    let nodes = body.nodes();
    let mut used = vec![false; body.types.len()];
    let mut elementwise_only = vec![true; body.types.len()];
    for &v in body.outputs() {
        used[v] = true;
    }
    // A node is at the output when it has the output's shape and reaches it only through
    // elementwise ops, so the tile can be read back at output positions.
    let mut at_output = vec![false; body.types.len()];
    for node in nodes.iter().rev() {
        let v = node.output;
        at_output[v] = used[v] && elementwise_only[v] && body.type_of(v).shape == out.shape;
        let elementwise = matches!(
            node.primitive,
            Add | Sub
                | Mul
                | Div
                | Max
                | Eq
                | Lt
                | Neg
                | Exp
                | Log
                | Sqrt
                | Tanh
                | Logistic
                | Cast { .. }
                | Select
        );
        for &x in &node.inputs {
            used[x] = true;
            elementwise_only[x] &= elementwise && at_output[v];
        }
    }
    let tile = |node: &Node| -> Option<[(usize, usize); 2]> {
        let Transpose { permutation } = &node.primitive else {
            return None;
        };
        if !at_output[node.output] {
            return None;
        }
        let xs = contiguous_strides(&body.type_of(node.inputs[0]).shape);
        // Merge adjacent output dims that stay contiguous in the input into runs of
        // (first dim, last dim, input stride).
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();
        for d in (0..out.shape.len()).filter(|&d| out.shape[d] != 1) {
            let stride = xs[permutation[d]];
            match runs.last_mut() {
                Some(run) if run.2 == stride * out.shape[d] => *run = (run.0, d, stride),
                _ => runs.push((d, d, stride)),
            }
        }
        let a = runs.iter().position(|r| r.2 == 1)?;
        let b = runs.len() - 1;
        let size = |r: (usize, usize, usize)| out.shape[r.0..=r.1].iter().product::<usize>();
        (a < b && size(runs[a]).min(size(runs[b])) >= MIN_TILED)
            .then(|| [(runs[a].0, runs[a].1), (runs[b].0, runs[b].1)])
    };
    let mut heroes = Vec::new();
    let mut tiled = None;
    for node in nodes {
        let Some(t) = tile(node) else { continue };
        if *tiled.get_or_insert(t) == t && heroes.len() < MAX_HEROES {
            heroes.push(node.output);
        }
    }
    let [(a0, a1), (b0, b1)] = tiled?;
    let out_strides = contiguous_strides(&out.shape);
    let size = |first: usize, last: usize| out.shape[first..=last].iter().product();
    let batch = (0..out.shape.len())
        .filter(|&d| out.shape[d] != 1 && !(a0..=a1).contains(&d) && !(b0..=b1).contains(&d))
        .map(|d| (out.shape[d], out_strides[d]))
        .collect();
    Some(TransposeTiling {
        heroes,
        a: size(a0, a1),
        b: size(b0, b1),
        a_stride: out_strides[a1],
        batch,
    })
}

/// Generates the tiled transpose kernel: loads each hero transpose's input into a tile at input
/// order, then computes and stores the outputs reading the tile transposed.
pub(super) fn transpose_kernel(
    body: &Graph,
    by_value: &[bool],
    tiling: &TransposeTiling,
    views: &HashMap<usize, (usize, Vec<usize>)>,
) -> (String, String) {
    let out = body.outputs()[0];
    let out_type = metal_type(body.type_of(out).dtype);
    let (params, _) = io_params(body, by_value, out_type, &[]);
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
    named(formatdoc!(
        "
        {stores}
        kernel void NAME({params}uint3 group [[threadgroup_position_in_grid]],
                         uint3 tid [[thread_position_in_threadgroup]]) {{
            {source}
        }}
        "
    ))
}
