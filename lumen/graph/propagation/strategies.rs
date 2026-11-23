//! `Primitive::strategies`: how each primitive runs on local shards of operands placed over a device mesh (PyTorch
//! DTensor's single-dim strategies), which `sharding.py` chooses from.

use super::rules::{Linear, elementwise, groups, reduce, unique};
use super::{Op, Placement, Strategy};
use crate::graph::{BinaryOp, Primitive};

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

            // A swap of adjacent last-axis pairs: sharded along another axis (a shard of the last could split a
            // pair), or partial by any reduction (each commutes with a permutation).
            SwapPairs { .. } => {
                let p = placements[0];
                let last = shapes[0].len() - 1;
                keep(matches!(p, Shard(d) if d != last) || matches!(p, Partial(_)))
            }

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
