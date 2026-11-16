//! How axis names flow through each primitive: `Primitive::names` infers a result's names from its operands',
//! as `Primitive::infer` infers its type (PyTorch's named-tensor name inference). The tracer calls it for every
//! node with a named operand; the graph itself never holds names.

use crate::graph::Primitive;

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
