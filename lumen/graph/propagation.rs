//! How tensor metadata flows through each primitive, as `Primitive::infer` gives its type. `Primitive::names`
//! infers a result's axis names from its operands' (PyTorch's named-tensor name inference); the tracer calls it
//! for every node with a named operand. `Primitive::strategies` lists how a primitive runs on local shards of
//! operands placed over a device mesh (PyTorch DTensor's single-dim strategies), which `lumen/graph/sharding.py`
//! chooses from. The graph itself never holds names or placements.

use crate::graph::{BinaryOp, Primitive};

/// An operand as a name rule sees it: its shape, and its axis names when it is named.
pub type Named<'a> = (&'a [usize], Option<&'a [String]>);

/// The one name tuple every one of `named` shares (None if there are none), or an error naming `what` if two
/// differ.
fn agreed(named: &[&[String]], what: &str) -> Result<Option<Vec<String>>, String> {
    for names in &named[1..] {
        if names != &named[0] {
            return Err(format!(
                "{what}: operands named {} and {} do not match: align them first (align_to)",
                tuple(named[0]),
                tuple(names)
            ));
        }
    }

    Ok(named.first().map(|n| n.to_vec()))
}

/// `names` written as a Python tuple, as error messages show them: `('a', 'b')`.
fn tuple<S: AsRef<str>>(names: &[S]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("'{}'", n.as_ref())).collect();

    match quoted.len() {
        1 => format!("({},)", quoted[0]),
        _ => format!("({})", quoted.join(", ")),
    }
}

/// `names` unless one repeats (a result can't have two axes of one name), else None.
fn distinct(names: Vec<String>) -> Option<Vec<String>> {
    let unique: std::collections::BTreeSet<&String> = names.iter().collect();
    (unique.len() == names.len()).then_some(names)
}

impl Primitive {
    /// The axis names of this primitive's result (of `shape`) on `operands` (at least one named), or None for
    /// none; an error when the operands' names conflict. Every primitive is classified here, so a new one
    /// must say how names flow through it.
    pub fn names(
        &self,
        operands: &[Named],
        shape: &[usize],
    ) -> Result<Option<Vec<String>>, String> {
        use Primitive::*;

        let first = || operands[0].1.map(<[String]>::to_vec);

        Ok(match self {
            // Elementwise ops (and collectives that keep their operand's shape): the names their named
            // operands of the result's shape share (the tracer aligns operands by name before such an op).
            Add
            | Sub
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
            | Unary(_)
            | Binary(_)
            | Select
            | HostWait
            | DeviceWait
            | ToHost
            | ToDevice
            | AllReduce { .. }
            | Broadcast { .. }
            | Reduce { .. } => {
                let named: Vec<&[String]> = operands
                    .iter()
                    .filter(|(s, _)| *s == shape)
                    .filter_map(|(_, n)| *n)
                    .collect();

                agreed(&named, "elementwise op")?
            }

            // A reduction: the operand's names but the reduced axes'.
            ReduceSum { axes, .. } | ReduceMax { axes } => first().map(|names| {
                names
                    .into_iter()
                    .enumerate()
                    .filter(|(d, _)| !axes.contains(d))
                    .map(|(_, n)| n)
                    .collect()
            }),

            // An op whose result has its first operand's axes, perhaps resized: that operand's names.
            Cumsum { .. }
            | Slice { .. }
            | DynamicSlice { .. }
            | DynamicUpdateSlice
            | ScatterAdd { .. } => first(),

            // A transpose: the operand's names, permuted as its axes.
            Transpose { permutation } => {
                first().map(|names| permutation.iter().map(|&d| names[d].clone()).collect())
            }

            // A reshape or broadcast keeps names only when it changes nothing; a named split, merge or
            // alignment names its result itself.
            Reshape { .. } => first().filter(|_| operands[0].0 == shape),
            BroadcastInDim {
                broadcast_dimensions,
                ..
            } => first().filter(|_| {
                operands[0].0 == shape && broadcast_dimensions.iter().copied().eq(0..shape.len())
            }),

            // The batch axes' names (the same in both operands), then each operand's free axes', as the result's
            // dims are ordered; None unless both are named, or when a name would repeat.
            DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                ..
            } => {
                let (Some(lhs), Some(rhs)) = (operands[0].1, operands[1].1) else {
                    return Ok(None);
                };

                let batch: Vec<&String> = lhs_batch.iter().map(|&d| &lhs[d]).collect();
                let rhs_names: Vec<&String> = rhs_batch.iter().map(|&d| &rhs[d]).collect();

                if batch != rhs_names {
                    return Err(format!(
                        "dot_general: batch axes named {} and {} do not match",
                        tuple(&batch),
                        tuple(&rhs_names)
                    ));
                }

                let free = |names: &[String], batch: &[usize], contracting: &[usize]| {
                    names
                        .iter()
                        .enumerate()
                        .filter(|(d, _)| !batch.contains(d) && !contracting.contains(d))
                        .map(|(_, n)| n.clone())
                        .collect::<Vec<_>>()
                };

                let mut names: Vec<String> = batch.into_iter().cloned().collect();
                names.extend(free(lhs, lhs_batch, lhs_contracting));
                names.extend(free(rhs, rhs_batch, rhs_contracting));
                distinct(names)
            }

            // `gather` along `axis`: the operand's names with that axis' replaced by the indices' (a 0-d index
            // adds none); None unless both are named, or when a name would repeat.
            Gather { axis } => {
                let (x, (index_shape, index_names)) = (operands[0].1, operands[1]);

                let ids = match index_names {
                    Some(n) => Some(n),
                    None if index_shape.is_empty() => Some(&[][..]),
                    None => None,
                };

                let (Some(x), Some(ids)) = (x, ids) else {
                    return Ok(None);
                };

                distinct([&x[..*axis], ids, &x[axis + 1..]].concat())
            }

            // `take_along_axis`: the result has the indices' shape and the operand's axes, so the indices'
            // names, or the operand's when the indices have none.
            TakeAlongAxis { .. } => operands[1].1.or(operands[0].1).map(<[String]>::to_vec),

            // `concatenate`: the names its named operands share.
            Concatenate { .. } => {
                let named: Vec<&[String]> = operands.iter().filter_map(|(_, n)| *n).collect();
                agreed(&named, "concatenate")?
            }

            // Collectives that move or reshape data across ranks, point-to-point transfers, factories, fused
            // and custom kernels: an unnamed result.
            AllGather { .. }
            | ReduceScatter { .. }
            | AllToAll { .. }
            | GatherTo { .. }
            | ScatterFrom { .. }
            | Send { .. }
            | Recv { .. }
            | Full { .. }
            | Iota { .. }
            | RandomBits { .. }
            | LinearCrossEntropy
            | Fusion { .. }
            | FusionOutput { .. }
            | CustomCall { .. }
            | PallasCall { .. } => None,
        })
    }
}

/// The reduction a partial value is pending.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Sum,
    Max,
    Min,
    Avg,
}

/// How a value is laid out along one mesh dimension (DTensor's `Shard`, `Replicate` and `Partial`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Split along this tensor dimension, one equal block per process.
    Shard(usize),
    /// Every process holds the whole value.
    Replicate,
    /// Every process holds a part; the value is their reduction, not yet done.
    Partial(Op),
}

/// A sharding strategy on one mesh dimension: each operand's placement on it, and the result's.
pub type Strategy = (Vec<Placement>, Placement);

impl Placement {
    /// Whether this is a partial value a linear op keeps partial: a sum's or an average's.
    fn summed(self) -> bool {
        matches!(self, Placement::Partial(Op::Sum | Op::Avg))
    }
}

/// How far an elementwise op is linear in its partial operands, so keeps them partial.
#[derive(Clone, Copy)]
enum Linear {
    /// Not linear: no partial operand stays partial.
    No,
    /// Linear in all operands at once (`add`): partial sums alike stay partial.
    Sum,
    /// Linear in each operand alone (`mul`): one partial sum, the others replicated.
    Scale,
    /// Linear in its first operand (`div`): a partial sum there, the others replicated.
    Numerator,
    /// The reduction itself (`max`, `min`): partial values of that reduction alike stay partial.
    Same(Op),
    /// An identity (a transfer or wait): any partial value stays partial.
    Any,
}

/// `placements` without repeats, in order.
fn unique(placements: &[Placement]) -> Vec<Placement> {
    let mut out = Vec::new();

    for &p in placements {
        if !out.contains(&p) {
            out.push(p);
        }
    }

    out
}

/// An elementwise op's strategies: every operand sharded as one of them is, and partial ones as `linear` allows.
fn elementwise(placements: &[Placement], linear: Linear) -> Vec<Strategy> {
    use Placement::*;

    let n = placements.len();
    let distinct = unique(placements);

    let mut out: Vec<Strategy> = distinct
        .iter()
        .filter(|p| matches!(p, Shard(_)))
        .map(|&p| (vec![p; n], p))
        .collect();

    for &p in &distinct {
        let Partial(op) = p else {
            continue;
        };

        match linear {
            Linear::Any => out.push((vec![p; n], p)),
            Linear::Sum if p.summed() => out.push((vec![p; n], p)),
            Linear::Same(q) if q == op => out.push((vec![p; n], p)),
            Linear::Scale if p.summed() => {
                out.extend((0..n).map(|k| {
                    let ins = (0..n).map(|j| if j == k { p } else { Replicate }).collect();
                    (ins, p)
                }));
            }
            Linear::Numerator if p.summed() => {
                let mut ins = vec![Replicate; n];
                ins[0] = p;
                out.push((ins, p));
            }
            _ => {}
        }
    }

    out
}

/// A reduction by `op` over `axes` of an operand placed `p`: a shard along a reduced axis made partial, along a
/// kept one kept (renumbered); a value partial by the same reduction (or an average, for a sum) kept.
fn reduce(p: Placement, axes: &[usize], op: Op) -> Vec<Strategy> {
    use Placement::*;

    match p {
        Shard(d) if axes.contains(&d) => vec![(vec![p], Partial(op))],
        Shard(d) => vec![(vec![p], Shard(d - axes.iter().filter(|&&a| a < d).count()))],
        Partial(q) if q == op || (op == Op::Sum && p.summed()) => vec![(vec![p], p)],
        _ => Vec::new(),
    }
}

/// The dims of shapes `a` and `b` (of one element count, none zero) in runs of equal products, in order: a
/// reshape from `a` to `b` merges or splits each run's dims of `a` into its dims of `b`.
fn groups(a: &[usize], b: &[usize]) -> Vec<(Vec<usize>, Vec<usize>)> {
    let (mut i, mut j, mut out) = (0, 0, Vec::new());

    while i < a.len() || j < b.len() {
        let (mut mine, mut theirs, mut x, mut y) = (Vec::new(), Vec::new(), 1, 1);

        if i < a.len() {
            mine.push(i);
            x = a[i];
            i += 1;
        }

        if j < b.len() {
            theirs.push(j);
            y = b[j];
            j += 1;
        }

        while x != y {
            if x < y {
                mine.push(i);
                x *= a[i];
                i += 1;
            } else {
                theirs.push(j);
                y *= b[j];
                j += 1;
            }
        }

        out.push((mine, theirs));
    }

    out
}

impl Primitive {
    /// Ways to run this primitive on local shards along one mesh dimension, its operands placed there as
    /// `placements` with global shapes `shapes`, its result of global `shape`: each the operands' placements it
    /// takes and its result's. The all-replicated one is always possible and goes unlisted; a primitive with
    /// none runs on replicated operands. Every primitive is classified here.
    pub fn strategies(
        &self,
        placements: &[Placement],
        shapes: &[&[usize]],
        shape: &[usize],
    ) -> Vec<Strategy> {
        use Placement::*;
        use Primitive::*;

        // The strategy keeping the single operand's placement, when `ok`.
        let keep = |ok: bool| match ok {
            true => vec![(vec![placements[0]], placements[0])],
            false => Vec::new(),
        };

        match self {
            Add | Sub | Neg => elementwise(placements, Linear::Sum),
            Mul => elementwise(placements, Linear::Scale),
            Div => elementwise(placements, Linear::Numerator),
            Max => elementwise(placements, Linear::Same(Op::Max)),
            Binary(BinaryOp::Min) => elementwise(placements, Linear::Same(Op::Min)),
            ToHost | ToDevice | HostWait => elementwise(placements, Linear::Any),
            Eq
            | Lt
            | Exp
            | Log
            | Sqrt
            | Tanh
            | Logistic
            | Cast { .. }
            | Unary(_)
            | Binary(_)
            | Select
            | DeviceWait
            | AllReduce { .. }
            | Broadcast { .. }
            | Reduce { .. } => elementwise(placements, Linear::No),

            ReduceSum { axes, .. } => reduce(placements[0], axes, Op::Sum),
            ReduceMax { axes } => reduce(placements[0], axes, Op::Max),

            // A cumulative sum: sharded along another axis, or a partial sum.
            Cumsum { axis, .. } => {
                let p = placements[0];
                keep(matches!(p, Shard(d) if d != *axis) || p.summed())
            }

            // A slice: sharded along an axis it keeps whole, or partial (it is linear).
            Slice {
                start_indices,
                limit_indices,
                ..
            } => keep(match placements[0] {
                Shard(d) => start_indices[d] == 0 && limit_indices[d] == shapes[0][d],
                Partial(_) => true,
                Replicate => false,
            }),

            // A transpose: a shard moves with its axis; a partial value stays partial.
            Transpose { permutation } => match placements[0] {
                p @ Shard(d) => {
                    let to = permutation
                        .iter()
                        .position(|&a| a == d)
                        .expect("a permutation");
                    vec![(vec![p], Shard(to))]
                }
                p => keep(matches!(p, Partial(_))),
            },

            // A reshape: a shard along the outermost (not size-1) dim of a run of dims merged or split into
            // others stays a shard along the outermost of those (each process's block is one contiguous part of
            // the run's elements either way); a partial value stays partial.
            Reshape { .. } => match placements[0] {
                Partial(_) => keep(true),
                p @ Shard(d) if shape.iter().product::<usize>() != 0 => {
                    let outer = |dims: &[usize], sizes: &[usize]| {
                        dims.iter().copied().find(|&k| sizes[k] != 1)
                    };

                    groups(shapes[0], shape)
                        .into_iter()
                        .find(|(mine, _)| mine.contains(&d))
                        .filter(|(mine, _)| outer(mine, shapes[0]) == Some(d))
                        .and_then(|(_, theirs)| outer(&theirs, shape))
                        .map(|to| vec![(vec![p], Shard(to))])
                        .unwrap_or_default()
                }
                _ => Vec::new(),
            },

            // A broadcast: a shard of an axis it keeps (not stretched from size 1) stays a shard of the axis it
            // becomes; a partial value stays partial (each reduction commutes with copying).
            BroadcastInDim {
                broadcast_dimensions,
                ..
            } => match placements[0] {
                Partial(_) => keep(true),
                p @ Shard(d) => {
                    let to = broadcast_dimensions[d];

                    match shapes[0][d] == shape[to] {
                        true => vec![(vec![p], Shard(to))],
                        false => Vec::new(),
                    }
                }
                Replicate => Vec::new(),
            },

            // DTensor's einsum strategies: a batch axis sharded in both operands and the result; a contracted
            // axis sharded in both, the result a partial sum; a free axis of one sharded there and in the result,
            // the other replicated; a partial sum with the other replicated (it is bilinear).
            DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                ..
            } => {
                let free = |rank: usize, batch: &[usize], contracting: &[usize]| {
                    (0..rank)
                        .filter(|d| !batch.contains(d) && !contracting.contains(d))
                        .collect::<Vec<_>>()
                };

                let lhs_free = free(shapes[0].len(), lhs_batch, lhs_contracting);
                let rhs_free = free(shapes[1].len(), rhs_batch, rhs_contracting);
                let batch = lhs_batch.len();
                let mut out = Vec::new();

                for (k, (&a, &b)) in lhs_batch.iter().zip(rhs_batch).enumerate() {
                    out.push((vec![Shard(a), Shard(b)], Shard(k)));
                }

                for (&a, &b) in lhs_contracting.iter().zip(rhs_contracting) {
                    out.push((vec![Shard(a), Shard(b)], Partial(Op::Sum)));
                }

                for (k, &a) in lhs_free.iter().enumerate() {
                    out.push((vec![Shard(a), Replicate], Shard(batch + k)));
                }

                for (k, &b) in rhs_free.iter().enumerate() {
                    out.push((vec![Replicate, Shard(b)], Shard(batch + lhs_free.len() + k)));
                }

                let (lhs, rhs) = (placements[0], placements[1]);

                if lhs.summed() {
                    out.push((vec![lhs, Replicate], lhs));
                }

                if rhs.summed() {
                    out.push((vec![Replicate, rhs], rhs));
                }

                out
            }

            // A dynamic slice: sharded along an axis it takes whole (so starts at 0), or partial (it is linear);
            // its start indices replicated.
            DynamicSlice { slice_sizes } => {
                let indices = vec![Replicate; placements.len() - 1];

                let ok = match placements[0] {
                    Shard(d) => slice_sizes[d] == shapes[0][d],
                    Partial(_) => true,
                    Replicate => false,
                };

                match ok {
                    true => vec![([vec![placements[0]], indices].concat(), placements[0])],
                    false => Vec::new(),
                }
            }

            // A dynamic update: the operand and the update sharded alike along an axis the update covers whole,
            // or both partial sums (it is linear in the two together); its start indices replicated.
            DynamicUpdateSlice => {
                let indices = vec![Replicate; placements.len() - 2];

                unique(&placements[..2])
                    .into_iter()
                    .filter(|&p| match p {
                        Shard(d) => shapes[1][d] == shapes[0][d],
                        p => p.summed(),
                    })
                    .map(|p| ([vec![p, p], indices.clone()].concat(), p))
                    .collect()
            }

            // A gather along `axis` (the indices' axes replace it): the operand sharded along another axis, the
            // indices replicated; the operand sharded along `axis`, the indices replicated, each process looking
            // up the indices in its block and zeroing the rest, so the result is a partial sum (DTensor's
            // `MaskPartial`; `sharding.py` records the masked lookup); the indices sharded, the operand
            // replicated; a partial sum of the operand, the indices replicated (it is linear in the operand).
            Gather { axis } => {
                let ids = shapes[1].len();
                let mut out = Vec::new();

                for p in unique(placements) {
                    match p {
                        Shard(d) if p == placements[0] && d != *axis => {
                            let to = if d < *axis { d } else { d + ids - 1 };
                            out.push((vec![p, Replicate], Shard(to)));
                        }
                        Shard(_) if p == placements[0] => {
                            out.push((vec![p, Replicate], Partial(Op::Sum)));
                        }
                        Shard(k) if p == placements[1] => {
                            out.push((vec![Replicate, p], Shard(axis + k)));
                        }
                        p if p == placements[0] && p.summed() => out.push((vec![p, Replicate], p)),
                        _ => {}
                    }
                }

                out
            }

            // `take_along_axis` (the indices pick along `axis`, each other axis matched): the operand and the
            // indices sharded alike along another axis both have whole; a partial sum of the operand, the indices
            // replicated.
            TakeAlongAxis { axis } => {
                let mut out = Vec::new();

                for p in unique(placements) {
                    match p {
                        Shard(d) if d != *axis && shapes[0][d] == shapes[1][d] => {
                            out.push((vec![p, p], p));
                        }
                        p if p == placements[0] && p.summed() => out.push((vec![p, Replicate], p)),
                        _ => {}
                    }
                }

                out
            }

            // `scatter_add` along `axis` (the transpose of a gather: the updates have the indices' axes in place
            // of it): the operand sharded along another axis and the updates along that axis' place in them,
            // the indices replicated; the operand and the updates partial sums, the indices replicated (it is
            // linear in the two together).
            ScatterAdd { axis } => {
                let ids = shapes[1].len();
                let mut out = Vec::new();

                for p in unique(placements) {
                    match p {
                        Shard(d) if p == placements[0] && d != *axis => {
                            let at = if d < *axis { d } else { d + ids - 1 };
                            out.push((vec![p, Replicate, Shard(at)], p));
                        }
                        p if p.summed() => out.push((vec![p, Replicate, p], p)),
                        _ => {}
                    }
                }

                out
            }

            // A concatenation: every operand sharded alike along another axis, or partial alike (each reduction
            // commutes with concatenating).
            Concatenate { dimension, .. } => unique(placements)
                .into_iter()
                .filter(|&p| match p {
                    Shard(d) => d != *dimension,
                    Partial(_) => true,
                    Replicate => false,
                })
                .map(|p| (vec![p; placements.len()], p))
                .collect(),

            // Data-moving collectives, transfers, factories, fused and custom kernels: no strategy, so
            // replicated operands.
            AllGather { .. }
            | ReduceScatter { .. }
            | AllToAll { .. }
            | GatherTo { .. }
            | ScatterFrom { .. }
            | Send { .. }
            | Recv { .. }
            | Full { .. }
            | Iota { .. }
            | RandomBits { .. }
            | LinearCrossEntropy
            | Fusion { .. }
            | FusionOutput { .. }
            | CustomCall { .. }
            | PallasCall { .. } => Vec::new(),
        }
    }
}
