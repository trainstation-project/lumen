use crate::device::tpu::layout::Layout;
use crate::graph::{Graph, Primitive, TensorType};

const LANES: usize = 128;

const FLAT: usize = 1024;

const BLOCK_ELEMENTS: usize = 1 << 18;

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Dim {
    Out { d: usize, offset: usize },
    One { offset: usize },
}

impl Dim {
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

#[cfg_attr(not(lumen_tpu_linked), allow(dead_code))]
pub(crate) struct Plan {
    pub(crate) blocks: Vec<usize>,
    pub(crate) maps: Vec<Option<Vec<Dim>>>,
}

#[cfg_attr(not(lumen_tpu_linked), allow(dead_code))]
impl Plan {
    pub(crate) fn block(&self, map: &[Dim]) -> Vec<usize> {
        map.iter()
            .map(|m| match *m {
                Dim::Out { d, .. } => self.blocks[d],
                Dim::One { .. } => 1,
            })
            .collect()
    }
}

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
