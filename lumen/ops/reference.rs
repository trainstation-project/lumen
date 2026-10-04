//! The reference executor: runs a [`Graph`] on the host, one primitive at
//! a time, and defines what each primitive computes. Integers are held
//! as `i128` (wide enough for every lumen integer dtype) and floats as
//! `f64`; each result is wrapped or rounded to its dtype, so integer ops
//! wrap around and float ops round as if computed in that dtype. Device
//! executors are checked against it.

use crate::graph::primitive::free_dims;
use crate::graph::{Graph, Primitive, TensorType};
use crate::tensor::dtype::{bf16, dispatch_dtype, f16};
use crate::tensor::for_each_index;
use crate::{DType, Device, Element, Scalar, Tensor, TensorOptions};

/// A graph value's elements in row-major order.
#[derive(Debug, Clone)]
enum Values {
    Int(Vec<i128>),
    Float(Vec<f64>),
}

/// Run `graph` on `inputs`, returning its outputs on the first input's
/// device (the CPU if it has none).
pub fn run(graph: &Graph, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
    if inputs.len() != graph.inputs().len() {
        return Err(format!(
            "the graph takes {} inputs, got {}",
            graph.inputs().len(),
            inputs.len()
        ));
    }
    for (&var, t) in graph.inputs().iter().zip(inputs) {
        let ty = graph.type_of(var);
        if t.dtype() != ty.dtype || t.shape() != ty.shape {
            return Err(format!(
                "input %{var} must be {ty}, got {}",
                TensorType::new(t.dtype(), t.shape())
            ));
        }
    }
    let outputs = eval_graph(graph, inputs.iter().map(load).collect());
    let device = inputs.first().map_or(Device::Cpu, Tensor::device);
    Ok(graph
        .outputs()
        .iter()
        .zip(&outputs)
        .map(|(&v, values)| store(values, graph.type_of(v), device))
        .collect())
}

/// The values of `graph`'s outputs, given its inputs'.
fn eval_graph(graph: &Graph, inputs: Vec<Values>) -> Vec<Values> {
    let mut env: Vec<Option<Values>> = vec![None; graph.types.len()];
    // A multi-output fusion's other outputs, by its value.
    let mut extra: Vec<Vec<Values>> = vec![Vec::new(); graph.types.len()];
    for (&var, values) in graph.inputs().iter().zip(inputs) {
        env[var] = Some(values);
    }
    for node in graph.nodes() {
        let args: Vec<&Values> = node
            .inputs
            .iter()
            .map(|&v| {
                env[v]
                    .as_ref()
                    .expect("graph values are defined before use")
            })
            .collect();
        let types: Vec<&TensorType> = node.inputs.iter().map(|&v| graph.type_of(v)).collect();
        let value = match &node.primitive {
            Primitive::Fusion { body, .. } if body.outputs().len() > 1 => {
                let mut outputs = eval_graph(body, args.iter().map(|&v| v.clone()).collect());
                extra[node.output] = outputs.split_off(1);
                outputs.remove(0)
            }
            Primitive::FusionOutput { index, .. } => extra[node.inputs[0]][index - 1].clone(),
            p => eval(p, &args, &types, graph.type_of(node.output)),
        };
        env[node.output] = Some(value);
    }
    graph
        .outputs()
        .iter()
        .map(|&v| env[v].clone().expect("graph values are defined before use"))
        .collect()
}

fn load(t: &Tensor) -> Values {
    let dtype = t.dtype();
    dispatch_dtype!(dtype, T => from_scalars(t.to_vec::<T>().into_iter().map(Element::to_scalar), dtype))
}

fn from_scalars(scalars: impl Iterator<Item = Scalar>, dtype: DType) -> Values {
    if dtype.is_float() {
        Values::Float(scalars.map(Scalar::to_f64).collect())
    } else {
        Values::Int(
            scalars
                .map(|s| match s {
                    Scalar::Bool(b) => b as i128,
                    // u64 comes back as the i64 with the same bits.
                    Scalar::Int(i) => wrap(i as i128, dtype),
                    Scalar::Float(_) => unreachable!("integer dtype"),
                })
                .collect(),
        )
    }
}

/// Primitive `p` as a host kernel over raw row-major buffers, the way a
/// CPU [`Plan`](crate::graph::Plan) runs each step: reads `inputs` (of
/// `types`) and writes `output` (of `out`).
///
/// # Safety
/// Each input must point to host memory holding its type's elements, all
/// initialized, and `output` to writable host memory for `out`'s elements.
pub(crate) unsafe fn eval_raw(
    p: &Primitive,
    inputs: &[*const u8],
    types: &[&TensorType],
    output: *mut u8,
    out: &TensorType,
) {
    let args: Vec<Values> = inputs
        .iter()
        .zip(types)
        .map(|(&ptr, ty)| {
            dispatch_dtype!(ty.dtype, T => {
                let ptr = ptr.cast::<T>();
                // SAFETY: the caller guarantees `numel` initialized elements.
                let scalars = (0..ty.numel()).map(|i| unsafe { ptr.add(i).read_unaligned() }.to_scalar());
                from_scalars(scalars, ty.dtype)
            })
        })
        .collect();
    let values = eval(p, &args.iter().collect::<Vec<_>>(), types, out);
    dispatch_dtype!(out.dtype, T => {
        let ptr = output.cast::<T>();
        for (i, x) in to_elements::<T>(&values).enumerate() {
            // SAFETY: the caller guarantees room for `numel` elements.
            unsafe { ptr.add(i).write_unaligned(x) };
        }
    })
}

/// `values` as elements of type `T`.
fn to_elements<T: Element>(values: &Values) -> Box<dyn Iterator<Item = T> + '_> {
    match values {
        // Wrapped to the dtype already, so the i64 cast keeps the bits.
        Values::Int(v) => Box::new(v.iter().map(|&x| T::from_scalar(Scalar::Int(x as i64)))),
        Values::Float(v) => Box::new(v.iter().map(|&x| T::from_scalar(Scalar::Float(x)))),
    }
}

fn store(values: &Values, ty: &TensorType, device: Device) -> Tensor {
    dispatch_dtype!(ty.dtype, T => {
        let data: Vec<T> = to_elements(values).collect();
        Tensor::from_slice(&data, TensorOptions::new().device(device)).reshape(&ty.shape)
    })
}

/// `x` wrapped around to integer dtype `dtype`.
fn wrap(x: i128, dtype: DType) -> i128 {
    use DType::*;
    match dtype {
        Bool => (x != 0) as i128,
        U8 => x as u8 as i128,
        U16 => x as u16 as i128,
        U32 => x as u32 as i128,
        U64 => x as u64 as i128,
        I8 => x as i8 as i128,
        I16 => x as i16 as i128,
        I32 => x as i32 as i128,
        I64 => x as i64 as i128,
        _ => unreachable!("{dtype} is not an integer dtype"),
    }
}

/// `x` rounded to float dtype `dtype`.
fn round(x: f64, dtype: DType) -> f64 {
    match dtype {
        DType::F16 => f16::from_f64(x).to_f64(),
        DType::BF16 => bf16::from_f64(x).to_f64(),
        DType::F32 => x as f32 as f64,
        DType::F64 => x,
        _ => unreachable!("{dtype} is not a float dtype"),
    }
}

/// `x` converted to integer dtype `dtype`: truncated toward zero and
/// clamped to its range, NaN to 0 (StableHLO's `convert`).
fn saturate(x: f64, dtype: DType) -> i128 {
    use DType::*;
    match dtype {
        Bool => (x != 0.0) as i128,
        U8 => x as u8 as i128,
        U16 => x as u16 as i128,
        U32 => x as u32 as i128,
        U64 => x as u64 as i128,
        I8 => x as i8 as i128,
        I16 => x as i16 as i128,
        I32 => x as i32 as i128,
        I64 => x as i64 as i128,
        _ => unreachable!("{dtype} is not an integer dtype"),
    }
}

/// The smallest value of `dtype`: the identity of `max`.
fn lowest(dtype: DType) -> Values {
    match dtype {
        d if d.is_float() => Values::Float(vec![f64::NEG_INFINITY]),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            Values::Int(vec![-(1 << (8 * dtype.size_of() - 1))])
        }
        _ => Values::Int(vec![0]),
    }
}

/// NaN-propagating max, as in StableHLO.
fn fmax(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else {
        x.max(y)
    }
}

/// The values of `out` built from `x` by an index map: element `idx` of
/// the result is element `source(idx)` (a row-major flat index) of `x`.
fn gather(x: &Values, out: &TensorType, source: impl Fn(&[usize]) -> usize) -> Values {
    let mut index = Vec::with_capacity(out.numel());
    if out.numel() > 0 {
        for_each_index(&out.shape, |idx| index.push(source(&idx)));
    }
    match x {
        Values::Int(v) => Values::Int(index.iter().map(|&i| v[i]).collect()),
        Values::Float(v) => Values::Float(index.iter().map(|&i| v[i]).collect()),
    }
}

/// The row-major flat index of `idx` in `shape`.
fn ravel(idx: impl IntoIterator<Item = usize>, shape: &[usize]) -> usize {
    idx.into_iter()
        .zip(shape)
        .fold(0, |flat, (i, n)| flat * n + i)
}

fn eval(p: &Primitive, args: &[&Values], types: &[&TensorType], out: &TensorType) -> Values {
    use Primitive::*;
    use Values::{Float, Int};
    let dtype = out.dtype;
    match p {
        Add | Sub | Mul | Div | Max | Eq | Lt => match (args[0], args[1]) {
            (Int(x), Int(y)) => {
                let f = |a: i128, b: i128| match p {
                    Add => a + b,
                    Sub => a - b,
                    Mul => a.wrapping_mul(b),
                    // Integer division by zero gives -1 (all ones), as in XLA.
                    Div if b == 0 => -1,
                    Div => a / b,
                    Max => a.max(b),
                    Eq => (a == b) as i128,
                    Lt => (a < b) as i128,
                    _ => unreachable!(),
                };
                Int(x
                    .iter()
                    .zip(y)
                    .map(|(&a, &b)| wrap(f(a, b), dtype))
                    .collect())
            }
            (Float(x), Float(y)) => {
                let pairs = x.iter().zip(y);
                match p {
                    Eq => Int(pairs.map(|(a, b)| (a == b) as i128).collect()),
                    Lt => Int(pairs.map(|(a, b)| (a < b) as i128).collect()),
                    _ => Float(
                        pairs
                            .map(|(&a, &b)| {
                                let r = match p {
                                    Add => a + b,
                                    Sub => a - b,
                                    Mul => a * b,
                                    Div => a / b,
                                    _ => fmax(a, b),
                                };
                                round(r, dtype)
                            })
                            .collect(),
                    ),
                }
            }
            _ => unreachable!("operands of one dtype"),
        },
        Neg => match args[0] {
            Int(x) => Int(x.iter().map(|&a| wrap(-a, dtype)).collect()),
            Float(x) => Float(x.iter().map(|&a| -a).collect()),
        },
        Exp | Log | Sqrt | Tanh | Logistic => {
            let Float(x) = args[0] else {
                unreachable!("float operand")
            };
            let f = |a: f64| match p {
                Exp => a.exp(),
                Log => a.ln(),
                Sqrt => a.sqrt(),
                Tanh => a.tanh(),
                _ => 1.0 / (1.0 + (-a).exp()),
            };
            Float(x.iter().map(|&a| round(f(a), dtype)).collect())
        }
        Cast { .. } => match (args[0], dtype.is_float()) {
            (Int(x), true) => Float(x.iter().map(|&a| round(a as f64, dtype)).collect()),
            (Int(x), false) => Int(x.iter().map(|&a| wrap(a, dtype)).collect()),
            (Float(x), true) => Float(x.iter().map(|&a| round(a, dtype)).collect()),
            (Float(x), false) => Int(x.iter().map(|&a| saturate(a, dtype)).collect()),
        },
        Select => {
            let Int(pred) = args[0] else {
                unreachable!("bool pred")
            };
            match (args[1], args[2]) {
                (Int(x), Int(y)) => Int(pick(pred, x, y)),
                (Float(x), Float(y)) => Float(pick(pred, x, y)),
                _ => unreachable!("cases of one dtype"),
            }
        }
        ReduceSum { axes, .. } | ReduceMax { axes } => {
            let x = types[0];
            let init = match p {
                ReduceSum { .. } if dtype.is_float() => Float(vec![0.0]),
                ReduceSum { .. } => Int(vec![0]),
                _ => lowest(dtype),
            };
            // The result index of each operand element: its index with the
            // reduced axes dropped.
            let mut target = Vec::with_capacity(x.numel());
            if x.numel() > 0 {
                for_each_index(&x.shape, |idx| {
                    let kept = free_dims(idx.len(), axes).map(|d| idx[d]);
                    target.push(ravel(kept, &out.shape));
                });
            }
            let sum = matches!(p, ReduceSum { .. });
            match (args[0], init) {
                (Int(v), Int(init)) => {
                    let mut acc = vec![init[0]; out.numel()];
                    for (&t, &a) in target.iter().zip(v) {
                        acc[t] = if sum {
                            wrap(acc[t] + a, dtype)
                        } else {
                            acc[t].max(a)
                        };
                    }
                    Int(acc)
                }
                (Float(v), Float(init)) => {
                    let mut acc = vec![init[0]; out.numel()];
                    // Accumulated in the dtype, rounding each step.
                    for (&t, &a) in target.iter().zip(v) {
                        acc[t] = if sum {
                            round(acc[t] + a, dtype)
                        } else {
                            fmax(acc[t], a)
                        };
                    }
                    Float(acc)
                }
                _ => unreachable!(),
            }
        }
        DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            rhs_batch,
            accum_dtype,
            ..
        } => {
            let (lhs, rhs) = (types[0], types[1]);
            let lhs_free: Vec<usize> = free_dims(
                lhs.shape.len(),
                &[lhs_batch.as_slice(), lhs_contracting].concat(),
            )
            .collect();
            let rhs_free: Vec<usize> = free_dims(
                rhs.shape.len(),
                &[rhs_batch.as_slice(), rhs_contracting].concat(),
            )
            .collect();
            let contracted: Vec<usize> = lhs_contracting.iter().map(|&d| lhs.shape[d]).collect();
            // For each result element, the (lhs, rhs) flat index pairs whose
            // products it sums.
            let mut terms: Vec<Vec<(usize, usize)>> = Vec::with_capacity(out.numel());
            if out.numel() > 0 {
                for_each_index(&out.shape, |idx| {
                    let (batch, rest) = idx.split_at(lhs_batch.len());
                    let (l_free, r_free) = rest.split_at(lhs_free.len());
                    let mut l = vec![0; lhs.shape.len()];
                    let mut r = vec![0; rhs.shape.len()];
                    for (i, &b) in batch.iter().enumerate() {
                        l[lhs_batch[i]] = b;
                        r[rhs_batch[i]] = b;
                    }
                    for (&d, &i) in lhs_free.iter().zip(l_free) {
                        l[d] = i;
                    }
                    for (&d, &i) in rhs_free.iter().zip(r_free) {
                        r[d] = i;
                    }
                    let mut pairs = Vec::new();
                    if contracted.iter().product::<usize>() > 0 {
                        for_each_index(&contracted, |c| {
                            for (k, &i) in c.iter().enumerate() {
                                l[lhs_contracting[k]] = i;
                                r[rhs_contracting[k]] = i;
                            }
                            pairs.push((
                                ravel(l.iter().copied(), &lhs.shape),
                                ravel(r.iter().copied(), &rhs.shape),
                            ));
                        });
                    }
                    terms.push(pairs);
                });
            }
            match (args[0], args[1]) {
                (Int(x), Int(y)) => Int(terms
                    .iter()
                    .map(|t| {
                        let sum = t.iter().fold(0i128, |acc, &(i, j)| {
                            wrap(acc + x[i].wrapping_mul(y[j]), dtype)
                        });
                        wrap(sum, dtype)
                    })
                    .collect()),
                (Float(x), Float(y)) => Float(
                    terms
                        .iter()
                        // In the accumulation dtype, each product and sum
                        // rounded to it; the result to output_dtype, once.
                        .map(|t| {
                            let acc = t.iter().fold(0.0, |acc, &(i, j)| {
                                round(acc + round(x[i] * y[j], *accum_dtype), *accum_dtype)
                            });
                            round(acc, dtype)
                        })
                        .collect(),
                ),
                _ => unreachable!("operands of one dtype"),
            }
        }
        Reshape { .. } => args[0].clone(),
        BroadcastInDim {
            broadcast_dimensions,
            ..
        } => {
            let x = types[0];
            gather(args[0], out, |idx| {
                let source = broadcast_dimensions
                    .iter()
                    .zip(&x.shape)
                    .map(|(&d, &n)| if n == 1 { 0 } else { idx[d] });
                ravel(source, &x.shape)
            })
        }
        Transpose { permutation } => {
            let x = types[0];
            gather(args[0], out, |idx| {
                let mut source = vec![0; idx.len()];
                for (i, &d) in permutation.iter().enumerate() {
                    source[d] = idx[i];
                }
                ravel(source, &x.shape)
            })
        }
        Slice { start_indices, .. } => {
            let x = types[0];
            gather(args[0], out, |idx| {
                ravel(idx.iter().zip(start_indices).map(|(i, s)| i + s), &x.shape)
            })
        }
        // The operands' elements one after another; an index's element is
        // in the last operand starting at or before its coordinate.
        Concatenate { dimension } => {
            let d = *dimension;
            let all = match args[0] {
                Int(_) => Int(args
                    .iter()
                    .flat_map(|a| match a {
                        Int(v) => v.clone(),
                        Float(_) => unreachable!("operands of one dtype"),
                    })
                    .collect()),
                Float(_) => Float(
                    args.iter()
                        .flat_map(|a| match a {
                            Float(v) => v.clone(),
                            Int(_) => unreachable!("operands of one dtype"),
                        })
                        .collect(),
                ),
            };
            let (mut starts, mut bases) = (Vec::new(), Vec::new());
            let (mut start, mut base) = (0, 0);
            for t in types {
                starts.push(start);
                bases.push(base);
                start += t.shape[d];
                base += t.numel();
            }
            gather(&all, out, |idx| {
                let k = starts.partition_point(|&s| s <= idx[d]) - 1;
                let source = idx
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| if i == d { c - starts[k] } else { c });
                bases[k] + ravel(source, &types[k].shape)
            })
        }
        Full { fill_value, .. } => {
            let one = match (*fill_value, dtype.is_float()) {
                (v, true) => Float(vec![round(v.to_f64(), dtype)]),
                (Scalar::Float(f), false) => Int(vec![saturate(f, dtype)]),
                (Scalar::Int(i), false) => Int(vec![wrap(i as i128, dtype)]),
                (Scalar::Bool(b), false) => Int(vec![b as i128]),
            };
            gather(&one, out, |_| 0)
        }
        Iota { dimension, .. } => {
            let n = out.shape[*dimension];
            let range = TensorType::new(dtype, &[n]);
            let values = if dtype.is_float() {
                Float((0..n).map(|i| round(i as f64, dtype)).collect())
            } else {
                Int((0..n).map(|i| wrap(i as i128, dtype)).collect())
            };
            gather(&values, out, |idx| ravel([idx[*dimension]], &range.shape))
        }
        Fusion { body, .. } => {
            let inputs = args.iter().map(|&v| v.clone()).collect();
            eval_graph(body, inputs).remove(0)
        }
        FusionOutput { .. } => unreachable!("evaluated with its fusion (eval_graph)"),
    }
}

fn pick<T: Copy>(pred: &[i128], x: &[T], y: &[T]) -> Vec<T> {
    pred.iter()
        .zip(x.iter().zip(y))
        .map(|(&p, (&a, &b))| if p != 0 { a } else { b })
        .collect()
}
