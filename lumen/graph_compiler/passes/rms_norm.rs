//! Detection of RMS norms written out as primitives, `h / sqrt(mean(h * h) + eps)` or `h * rsqrt(..)`, so
//! autograd differentiates each from its rows' saved `1 / rms` (`lumen/autograd/rms_norm.py`), as it
//! does `F.rms_norm`.

use crate::Scalar;
use crate::graph::{BinaryOp, Graph, Node, Primitive, UnaryOp, Var};

/// An RMS norm traced as primitives.
pub(crate) struct RmsNorm {
    /// The norm's operand: `h`, or the narrower value `h` is a widening cast of (read by the norm alone).
    pub(crate) x: Var,
    /// The normalized value `h / rms`, or (with `weight`) its rounding to `x`'s dtype times the weight.
    pub(crate) out: Var,
    /// The `[n]` weight the normalized rows are multiplied by, `F.rms_norm`'s, if any.
    pub(crate) weight: Option<Var>,
    /// Each row's `rms` (`sqrt(mean(h * h) + eps)`), or its `1 / rms` where written as `rsqrt`.
    pub(crate) rms: Var,
    /// Whether `rms` holds `1 / rms`.
    pub(crate) inverse: bool,
    /// The values computed between `x` (and `weight`) and `out`, `out` included, sorted.
    pub(crate) members: Vec<Var>,
}

/// Each RMS norm over the last axis written out in `graph`, its inner values read by it alone.
pub(crate) fn rms_norms(graph: &Graph) -> Vec<RmsNorm> {
    let nodes = graph.nodes();
    let mut producer = vec![None; graph.types.len()];
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);

        for &v in &node.inputs {
            readers[v].push(i);
        }
    }

    let node = |v: Var| producer[v].map(|k| &nodes[k]);

    // `v` through reshapes and broadcasts, which keep or spread a row's value.
    let through = |mut v: Var| {
        while let Some(n) = node(v).filter(|n| {
            matches!(
                n.primitive,
                Primitive::Reshape { .. } | Primitive::BroadcastInDim { .. }
            )
        }) {
            v = n.inputs[0];
        }

        v
    };

    // The scalar `v` holds, a constant broadcast or reshaped.
    let constant = |v: Var| match node(through(v)).map(|n| &n.primitive) {
        Some(Primitive::Full {
            fill_value, shape, ..
        }) if shape.is_empty() => match *fill_value {
            Scalar::Float(f) => Some(f),
            Scalar::Int(i) => Some(i as f64),
            Scalar::Bool(_) => None,
        },
        _ => None,
    };

    // The operand of `n` other than the one `pick` accepts, trying both orders.
    let other = |n: &Node, pick: &dyn Fn(Var) -> bool| match n.inputs[..] {
        [a, b] if pick(a) => Some(b),
        [a, b] if pick(b) => Some(a),
        _ => None,
    };

    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs();

    nodes
        .iter()
        .filter_map(|out| {
            // `h / sqrt(a)` or `h * rsqrt(a)`, `a = mean(h * h) + eps` per row.
            let (h, rms, inverse) = match out.primitive {
                Primitive::Div => {
                    let r = through(out.inputs[1]);
                    node(r).filter(|n| n.primitive == Primitive::Sqrt)?;
                    (out.inputs[0], r, false)
                }
                Primitive::Mul => {
                    let rsqrt = |v: Var| {
                        node(through(v))
                            .is_some_and(|n| n.primitive == Primitive::Unary(UnaryOp::Rsqrt))
                    };

                    let h = other(out, &rsqrt)?;
                    let r = through(if out.inputs[0] == h { out.inputs[1] } else { out.inputs[0] });
                    (h, r, true)
                }
                _ => return None,
            };

            let ty = graph.type_of(h);
            let (rank, n) = (ty.shape.len(), *ty.shape.last()?);

            if !ty.dtype.is_float() || graph.type_of(out.output) != ty || rank == 0 || n < 2 {
                return None;
            }

            let a = node(rms)?.inputs[0];
            let add = node(a).filter(|n| n.primitive == Primitive::Add)?;
            let mean = other(add, &|v| constant(v).is_some())?;
            let scale = node(mean)?;

            let sum = match scale.primitive {
                Primitive::Mul => other(scale, &|v| constant(v).is_some_and(|c| close(c, 1.0 / n as f64)))?,
                Primitive::Div if constant(scale.inputs[1]).is_some_and(|c| close(c, n as f64)) => scale.inputs[0],
                _ => return None,
            };

            let sum = node(through(sum)).filter(|s| {
                matches!(&s.primitive, Primitive::ReduceSum { axes, .. } if axes == &[rank - 1])
            })?;

            let square = node(sum.inputs[0])?;

            let squares = match square.primitive {
                Primitive::Mul => square.inputs == [h, h],
                Primitive::Binary(BinaryOp::Pow) => {
                    square.inputs[0] == h && constant(square.inputs[1]) == Some(2.0)
                }
                _ => false,
            };

            if !squares {
                return None;
            }

            // A widening cast of a narrower value read by the norm alone starts the norm there.
            let x = match node(h).map(|c| (&c.primitive, c.inputs[0])) {
                Some((Primitive::Cast { .. }, x))
                    if graph.type_of(x).dtype.is_float()
                        && graph.type_of(x).dtype.size_of() < ty.dtype.size_of()
                        && !graph.outputs().contains(&h)
                        && readers[h]
                            .iter()
                            .all(|&k| std::ptr::eq(&nodes[k], out) || nodes[k].output == square.output) =>
                {
                    x
                }
                _ => h,
            };

            // A weight times the normalized rows (rounded to `x`'s dtype first where `h` widened it),
            // `F.rms_norm`'s, joins the norm: its gradient then needs no normalized value kept.
            let sole = |v: Var| match (&readers[v][..], graph.outputs().contains(&v)) {
                (&[k], false) => Some(&nodes[k]),
                _ => None,
            };

            let narrow = graph.type_of(x).dtype;

            let rounded = match narrow == ty.dtype {
                true => Some(out.output),
                false => sole(out.output)
                    .filter(|c| c.primitive == Primitive::Cast { new_dtype: narrow })
                    .map(|c| c.output),
            };

            let weighted = rounded.and_then(|r| {
                let mul = sole(r).filter(|m| m.primitive == Primitive::Mul)?;
                let b = other(mul, &|v| v == r)?;
                let spread = node(b).filter(|n| {
                    matches!(&n.primitive, Primitive::BroadcastInDim { broadcast_dimensions, .. } if broadcast_dimensions == &[rank - 1])
                })?;
                let w = spread.inputs[0];
                let wt = graph.type_of(w);

                (wt.shape == [n] && wt.dtype == narrow && sole(b).is_some()).then_some((mul.output, w))
            });

            let (end, weight) = match weighted {
                Some((end, w)) => (end, Some(w)),
                None => (out.output, None),
            };

            // The values between the operands and the output, each read by them alone.
            let mut members: Vec<Var> = Vec::new();
            let mut stack = vec![end];

            while let Some(v) = stack.pop() {
                if v == x || Some(v) == weight || members.contains(&v) {
                    continue;
                }

                let k = producer[v]?;
                members.push(v);
                stack.extend(&nodes[k].inputs);
            }

            let inside = |k: usize| members.contains(&nodes[k].output);

            let sealed = members.iter().filter(|&&v| v != end).all(|&v| {
                !graph.outputs().contains(&v) && readers[v].iter().all(|&k| inside(k))
            });

            if !sealed {
                return None;
            }

            members.sort_unstable();

            Some(RmsNorm {
                x,
                out: end,
                weight,
                rms,
                inverse,
                members,
            })
        })
        .collect()
}
