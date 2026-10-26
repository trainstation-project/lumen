//! Block planning for TPU loops with layout primitives: maps every value's dims onto the output's
//! grid and picks block sizes every operand can be read in as whole tiles.

use crate::device::tpu::layout::Layout;
use crate::graph::{Graph, Primitive, TensorType};

/// Lanes of a TPU vreg: a tiled array's last-dim unit.
const LANES: usize = 128;

/// Block unit of a rank-1 array, which is tiled in flat chunks.
const FLAT: usize = 1024;

/// Most elements one block may hold.
const BLOCK_ELEMENTS: usize = 1 << 18;

/// Where one dim of a value's block sits relative to the output's grid.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Dim {
    /// Follows output dim `d`, shifted by `offset` elements.
    Out { d: usize, offset: usize },
    /// A single index fixed at `offset`, independent of the grid.
    One { offset: usize },
}

impl Dim {
    /// The same dim read `by` elements further along.
    fn shifted(self, by: usize) -> Dim {
        match self {
            Dim::Out { d, offset } => Dim::Out {
                d,
                offset: offset + by,
            },
            Dim::One { offset } => Dim::One {
                offset: offset + by,
            },
        }
    }
}

/// Block sizes along each output dim and, per value, how its dims follow them.
#[cfg_attr(not(lumen_tpu_linked), allow(dead_code))]
pub(crate) struct Plan {
    /// Block size along each output dim.
    pub(crate) blocks: Vec<usize>,
    /// Each value's per-dim map, `None` for values with no block (scalars, unreached values).
    pub(crate) maps: Vec<Option<Vec<Dim>>>,
}

#[cfg_attr(not(lumen_tpu_linked), allow(dead_code))]
impl Plan {
    /// The block shape a value with dim map `map` is read in.
    pub(crate) fn block(&self, map: &[Dim]) -> Vec<usize> {
        map.iter()
            .map(|m| match *m {
                Dim::Out { d, .. } => self.blocks[d],
                Dim::One { .. } => 1,
            })
            .collect()
    }
}

/// Whether `p` only moves or generates elements, which needs the indexed loop.
pub(crate) fn layout(p: &Primitive) -> bool {
    matches!(
        p,
        Primitive::BroadcastInDim { .. }
            | Primitive::Reshape { .. }
            | Primitive::Transpose { .. }
            | Primitive::Slice { .. }
            | Primitive::Iota { .. }
    )
}

/// Walks the loop back from its output, giving each value the dim map it is read at;
/// fails when a value is read at two different places or a primitive can't be mapped.
fn maps(body: &Graph) -> Result<Vec<Option<Vec<Dim>>>, String> {
    let root = body.outputs()[0];
    let rank = body.type_of(root).shape.len();
    let mut maps: Vec<Option<Vec<Dim>>> = vec![None; body.types.len()];
    maps[root] = Some((0..rank).map(|d| Dim::Out { d, offset: 0 }).collect());
    let assign = |maps: &mut Vec<Option<Vec<Dim>>>, v: usize, map: Vec<Dim>| match &maps[v] {
        Some(m) if *m != map => Err("a value read at two places of the block".to_string()),
        _ => {
            maps[v] = Some(map);
            Ok(())
        }
    };
    for node in body.nodes().iter().rev() {
        let Some(m) = maps[node.output].clone() else {
            continue;
        };
        let x = node.inputs.first().copied();
        match &node.primitive {
            Primitive::BroadcastInDim {
                shape,
                broadcast_dimensions,
            } => {
                let x = x.expect("an operand");
                let xt = body.type_of(x);
                // A broadcast size-1 dim stays at index 0 rather than following the grid.
                let xm = broadcast_dimensions
                    .iter()
                    .enumerate()
                    .map(|(k, &d)| match xt.shape[k] == 1 && shape[d] != 1 {
                        true => Dim::One { offset: 0 },
                        false => m[d],
                    })
                    .collect();
                assign(&mut maps, x, xm)?;
            }
            Primitive::Reshape { .. } => {
                let x = x.expect("an operand");
                let (xs, os) = (&body.type_of(x).shape, &body.type_of(node.output).shape);
                // Only reshapes that add or drop unit dims are mappable.
                let wide = |s: &[usize]| (0..s.len()).filter(|&k| s[k] != 1).collect::<Vec<_>>();
                let (xw, ow) = (wide(xs), wide(os));
                if xw.len() != ow.len() || xw.iter().zip(&ow).any(|(&a, &b)| xs[a] != os[b]) {
                    return Err("a reshape of more than unit dimensions".into());
                }
                let mut xm = vec![Dim::One { offset: 0 }; xs.len()];
                for (&a, &b) in xw.iter().zip(&ow) {
                    xm[a] = m[b];
                }
                assign(&mut maps, x, xm)?;
            }
            Primitive::Transpose { permutation } => {
                let x = x.expect("an operand");
                let mut xm = vec![Dim::One { offset: 0 }; permutation.len()];
                for (i, &p) in permutation.iter().enumerate() {
                    xm[p] = m[i];
                }
                assign(&mut maps, x, xm)?;
            }
            Primitive::Slice { start_indices, .. } => {
                let x = x.expect("an operand");
                let xm = m
                    .iter()
                    .zip(start_indices)
                    .map(|(d, &s)| d.shifted(s))
                    .collect();
                assign(&mut maps, x, xm)?;
            }
            Primitive::Iota { .. } | Primitive::Full { .. } => {}
            p if super::fusible(p) => {
                for &v in &node.inputs {
                    if !body.type_of(v).shape.is_empty() {
                        assign(&mut maps, v, m.clone())?;
                    }
                }
            }
            p => return Err(format!("{} in a loop", p.name())),
        }
    }
    Ok(maps)
}

/// Whether `t` read through `map` with output blocks `blocks` takes whole tiles at block-aligned offsets.
fn readable(t: &TensorType, map: &[Dim], blocks: &[usize]) -> bool {
    let Ok(layout) = Layout::new(t.dtype, &t.shape) else {
        return false;
    };
    let rank = t.shape.len();
    map.iter().enumerate().all(|(k, dim)| {
        let (block, offset) = match *dim {
            Dim::Out { d, offset } => (blocks[d], offset),
            Dim::One { offset } => (1, offset),
        };
        // Rank-1 arrays are tiled flat; otherwise the last two dims are tiled by lanes and sublane rows.
        let unit = match (rank, rank - k) {
            (1, _) => FLAT,
            (_, 1) => LANES,
            (_, 2) => layout.rows,
            _ => 1,
        };
        let whole = block == t.shape[k] || block.is_multiple_of(unit);
        whole && offset.is_multiple_of(block) && offset + block <= t.shape[k]
    })
}

/// Block sizes for a dim of `n`: `n` itself if at most `cap`, then the multiples of `unit` up to
/// `cap` that divide `n`, largest first.
fn candidates(n: usize, unit: usize, cap: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    if n <= cap {
        out.push(n);
    }
    out.extend(
        (1..=cap / unit)
            .rev()
            .map(|i| i * unit)
            .filter(|&b| b < n && n.is_multiple_of(b)),
    );
    out
}

/// Picks blocks for the loop `body` that every operand and the output read in whole tiles, with
/// the last block dividing `window` (the output's column offset in a wider matrix).
pub(crate) fn plan(body: &Graph, window: usize) -> Result<Plan, String> {
    let maps = maps(body)?;
    let root = body.outputs()[0];
    let out = body.type_of(root);
    let rank = out.shape.len();
    let reads: Vec<(&TensorType, &Vec<Dim>)> = body
        .inputs()
        .iter()
        .chain([&root])
        .filter_map(|&v| {
            let t = body.type_of(v);
            (!t.shape.is_empty()).then(|| maps[v].as_ref().map(|m| (t, m)))
        })
        .collect::<Option<_>>()
        .ok_or("an operand no member reads")?;
    let fits = |blocks: &[usize]| {
        reads.iter().all(|(t, m)| readable(t, m, blocks)) && window.is_multiple_of(blocks[rank - 1])
    };
    let blocks = match rank {
        0 => return Err("a scalar loop of layout".into()),
        1 => candidates(out.shape[0], FLAT, 64 * FLAT)
            .into_iter()
            .map(|b| vec![b])
            .find(|b| fits(b)),
        _ => {
            let (m, n) = (out.shape[rank - 2], out.shape[rank - 1]);
            let mut found = None;
            let mut rows = candidates(m, 8, 512);
            rows.extend(
                candidates(m, FLAT, 8 * FLAT)
                    .into_iter()
                    .filter(|b| *b > 512),
            );
            // Prefer the widest column block, then the tallest row block that fits.
            'search: for bn in candidates(n, LANES, 2048) {
                for &bm in &rows {
                    let mut blocks = vec![1; rank];
                    blocks[rank - 2] = bm;
                    blocks[rank - 1] = bn;
                    if bm * bn <= BLOCK_ELEMENTS && fits(&blocks) {
                        found = Some(blocks);
                        break 'search;
                    }
                }
            }
            found
        }
    };
    let blocks = blocks.ok_or("no block of whole tiles for every operand")?;
    Ok(Plan { blocks, maps })
}
