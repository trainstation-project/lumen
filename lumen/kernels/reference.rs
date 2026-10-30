//! Reference interpreter: evaluates graphs element by element on the host in i128/f64, rounding to each result
//! dtype, as the ground truth backends are tested against.
use crate::graph::primitive::free_dims;
use crate::graph::{Graph, Primitive, TensorType};
use crate::tensor::dtype::{dispatch_dtype, f8e4m3b11fnuz, f8e4m3fn, f8e5m2, to_bf16, to_f16};
use crate::tensor::for_each_index;
use crate::{DType, Device, Element, Scalar, Tensor, TensorOptions};

/// A tensor's elements flattened row-major: integers and bools widened to i128, floats to f64.
#[derive(Debug, Clone)]
enum Values {
    Int(Vec<i128>),
    Float(Vec<f64>),
}

/// Evaluates `graph` on `inputs` and returns its outputs as tensors on the first input's device.
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
    let device = inputs.first().map_or(Device::Cpu(0), Tensor::device);

    Ok(graph
        .outputs()
        .iter()
        .zip(&outputs)
        .map(|(&v, values)| store(values, graph.type_of(v), device))
        .collect())
}

/// Evaluates every node of `graph` in order and returns the values of its outputs.
fn eval_graph(graph: &Graph, inputs: Vec<Values>) -> Vec<Values> {
    let mut env: Vec<Option<Values>> = vec![None; graph.types.len()];
    // extra outputs of multi-output fusions, read back by their FusionOutput nodes
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

/// Reads a tensor's elements into `Values`.
fn load(t: &Tensor) -> Values {
    let dtype = t.dtype();
    dispatch_dtype!(dtype, T => from_scalars(t.to_vec::<T>().into_iter().map(Element::to_scalar), dtype))
}

/// Collects scalars into `Values`, wrapping integers to `dtype`'s range.
fn from_scalars(scalars: impl Iterator<Item = Scalar>, dtype: DType) -> Values {
    if dtype.is_float() {
        Values::Float(scalars.map(Scalar::to_f64).collect())
    } else {
        Values::Int(
            scalars
                .map(|s| match s {
                    Scalar::Bool(b) => b as i128,
                    Scalar::Int(i) => wrap(i as i128, dtype),
                    Scalar::Float(_) => unreachable!("integer dtype"),
                })
                .collect(),
        )
    }
}

/// Evaluates primitive `p` on raw host buffers and writes the result to `output`.
///
/// # Safety
/// Each `inputs[i]` must point to `types[i].numel()` readable elements of `types[i].dtype`, and `output` to
/// `out.numel()` writable elements of `out.dtype`.
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
                let scalars = (0..ty.numel()).map(|i| unsafe { ptr.add(i).read_unaligned() }.to_scalar());
                from_scalars(scalars, ty.dtype)
            })
        })
        .collect();

    let values = eval(p, &args.iter().collect::<Vec<_>>(), types, out);

    dispatch_dtype!(out.dtype, T => {
        let ptr = output.cast::<T>();

        for (i, x) in to_elements::<T>(&values).enumerate() {
            unsafe { ptr.add(i).write_unaligned(x) };
        }
    })
}

/// Converts values to elements of `T`.
fn to_elements<T: Element>(values: &Values) -> Box<dyn Iterator<Item = T> + '_> {
    match values {
        Values::Int(v) => Box::new(v.iter().map(|&x| T::from_scalar(Scalar::Int(x as i64)))),
        Values::Float(v) => Box::new(v.iter().map(|&x| T::from_scalar(Scalar::Float(x)))),
    }
}

/// Builds a tensor of type `ty` on `device` from values.
fn store(values: &Values, ty: &TensorType, device: Device) -> Tensor {
    dispatch_dtype!(ty.dtype, T => {
        let data: Vec<T> = to_elements(values).collect();
        Tensor::from_slice(&data, TensorOptions::new().device(device)).reshape(&ty.shape)
    })
}

/// Wraps an integer to `dtype` with two's complement overflow, including the 2- and 4-bit types.
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
        I2 | I4 => {
            let shift = 128 - dtype.bits();
            (x << shift) >> shift
        }
        U2 | U4 => x & ((1 << dtype.bits()) - 1),
        _ => unreachable!("{dtype} is not an integer dtype"),
    }
}

/// Rounds `x` to the nearest value representable in float `dtype`.
fn round(x: f64, dtype: DType) -> f64 {
    match dtype {
        DType::F16 => to_f16(x).to_f64(),
        DType::BF16 => to_bf16(x).to_f64(),
        DType::F32 => x as f32 as f64,
        DType::F64 => x,
        DType::F8E4M3FN => f8e4m3fn::from_f64(x).to_f64(),
        DType::F8E5M2 => f8e5m2::from_f64(x).to_f64(),
        DType::F8E4M3B11FNUZ => f8e4m3b11fnuz::from_f64(x).to_f64(),
        _ => unreachable!("{dtype} is not a float dtype"),
    }
}

/// Converts a float to integer `dtype` as a cast does: saturating, NaN to 0.
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
        I2 | I4 => (x as i128).clamp(-(1 << (dtype.bits() - 1)), (1 << (dtype.bits() - 1)) - 1),
        U2 | U4 => (x as i128).clamp(0, (1 << dtype.bits()) - 1),
        _ => unreachable!("{dtype} is not an integer dtype"),
    }
}

/// Returns the identity of max for `dtype` as a single value.
fn lowest(dtype: DType) -> Values {
    match dtype {
        d if d.is_float() => Values::Float(vec![f64::NEG_INFINITY]),
        DType::I2 | DType::I4 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            Values::Int(vec![-(1 << (dtype.bits() - 1))])
        }
        _ => Values::Int(vec![0]),
    }
}

/// Minimum that propagates NaN.
fn fmin(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else {
        x.min(y)
    }
}

/// Returns the bit width of `dtype`, counting bool as 1.
fn bits(dtype: DType) -> u32 {
    match dtype {
        DType::Bool => 1,
        d => d.bits() as u32,
    }
}

/// Reinterprets the low `w` bits of `x` as a signed integer.
fn signed_bits(x: i128, w: u32) -> i128 {
    let bits = unsigned_bits(x, w) as i128;

    if w > 0 && bits >= 1 << (w - 1) {
        bits - (1 << w)
    } else {
        bits
    }
}

/// Returns the low `w` bits of `x`.
fn unsigned_bits(x: i128, w: u32) -> u128 {
    (x as u128) & ((1u128 << w) - 1)
}

/// Inverse error function via Giles' single-precision polynomial approximation.
fn erf_inv(x: f64) -> f64 {
    if x.abs() == 1.0 {
        return x * f64::INFINITY;
    }

    let w = -((1.0 - x) * (1.0 + x)).ln();

    let p = if w < 5.0 {
        let w = w - 2.5;

        [
            2.810_226_36e-08,
            3.432_739_39e-07,
            -3.523_387_7e-06,
            -4.391_506_54e-06,
            0.000_218_580_87,
            -0.001_253_725_03,
            -0.004_177_681_64,
            0.246_640_727,
            1.501_409_41,
        ]
        .iter()
        .fold(0.0, |p, c| c + p * w)
    } else {
        let w = w.sqrt() - 3.0;

        [
            -0.000_200_214_257,
            0.000_100_950_558,
            0.001_349_343_22,
            -0.003_673_428_44,
            0.005_739_507_73,
            -0.007_622_461_3,
            0.009_438_870_47,
            1.001_674_06,
            2.832_976_82,
        ]
        .iter()
        .fold(0.0, |p, c| c + p * w)
    };

    p * x
}

/// Rounds `x` to a float with `exponent_bits` and `mantissa_bits`, like XLA's ReducePrecision: round to nearest
/// even, overflow to infinity and underflow to signed zero.
fn reduce_precision(x: f64, exponent_bits: u32, mantissa_bits: u32) -> f64 {
    if x.is_nan() {
        return x;
    }

    let mut b = x.to_bits();

    // round half to even on the dropped mantissa bits
    if mantissa_bits < 52 {
        let last = 1u64 << (52 - mantissa_bits);
        let bias = (last >> 1) - 1 + ((b & last) >> (52 - mantissa_bits));
        b = (b + bias) & !(last - 1);
    }

    // clamp the exponent to the reduced range
    if exponent_bits < 11 {
        let reduced_bias = (1u64 << (exponent_bits - 1)) - 1;
        let exponent = b & 0x7FF0_0000_0000_0000;
        let sign = b & 0x8000_0000_0000_0000;

        if exponent > (1023 + reduced_bias) << 52 {
            b = sign | 0x7FF0_0000_0000_0000;
        } else if exponent <= (1023 - reduced_bias) << 52 {
            b = sign;
        }
    }

    f64::from_bits(b)
}

/// Returns the next representable value of float `dtype` after `x` towards `y`.
fn nextafter(x: f64, y: f64, dtype: DType) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }

    if x == y {
        return y;
    }

    // step the bit pattern by one ulp; from zero go to the smallest subnormal with the direction's sign
    let step = |bits: u64, sign: u64, up: bool| -> u64 {
        match bits & !sign {
            0 => (if y > x { 0 } else { sign }) | 1,
            _ if up == (bits & sign == 0) => bits + 1,
            _ => bits - 1,
        }
    };

    let up = y > x;

    match dtype {
        DType::F32 => f64::from(f32::from_bits(
            step(u64::from((x as f32).to_bits()), 1 << 31, up) as u32,
        )),
        DType::F16 => {
            crate::tensor::dtype::f16::from_bits(
                step(u64::from(to_f16(x).to_bits()), 1 << 15, up) as u16
            )
            .to_f64()
        }
        DType::BF16 => {
            crate::tensor::dtype::bf16::from_bits(
                step(u64::from(to_bf16(x).to_bits()), 1 << 15, up) as u16,
            )
            .to_f64()
        }
        _ => f64::from_bits(step(x.to_bits(), 1 << 63, up)),
    }
}

/// Maximum that propagates NaN.
fn fmax(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else {
        x.max(y)
    }
}

/// Builds an `out`-shaped tensor whose element at each index is `x[source(index)]`.
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

/// Returns the row-major flat offset of `idx` within `shape`.
fn ravel(idx: impl IntoIterator<Item = usize>, shape: &[usize]) -> usize {
    idx.into_iter()
        .zip(shape)
        .fold(0, |flat, (i, n)| flat * n + i)
}

/// Reads integer indices into an axis of size `n`, clamped into range.
fn taken(indices: &Values, n: usize) -> Vec<usize> {
    let Values::Int(v) = indices else {
        unreachable!("integer indices")
    };

    v.iter()
        .map(|&k| k.clamp(0, (n as i128 - 1).max(0)) as usize)
        .collect()
}

/// Reads dynamic slice start indices, clamped so a slice of `sizes` stays inside `x`.
fn starts(x: &TensorType, indices: &[&Values], sizes: &[usize]) -> Vec<usize> {
    indices
        .iter()
        .zip(&x.shape)
        .zip(sizes)
        .map(|((index, &n), &size)| {
            let Values::Int(i) = index else {
                unreachable!("an integer index")
            };

            i[0].clamp(0, (n - size) as i128) as usize
        })
        .collect()
}

/// Evaluates one primitive on `args` of `types`, producing a result of type `out`.
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
                    // integer division by zero yields -1, matching XLA
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
        Unary(op) => {
            use crate::graph::UnaryOp as U;
            let w = bits(types[0].dtype);

            match args[0] {
                Int(x) => Int(x
                    .iter()
                    .map(|&a| match op {
                        U::Abs => wrap(a.abs(), dtype),
                        U::Sign => a.signum(),
                        U::Not if types[0].dtype == DType::Bool => 1 - a,
                        U::Not => wrap(!a, dtype),
                        U::PopulationCount => i128::from(unsigned_bits(a, w).count_ones()),
                        U::Clz => i128::from(unsigned_bits(a, w).leading_zeros() - (128 - w)),
                        _ => unreachable!("{} of an integer", op.name()),
                    })
                    .collect()),
                Float(x) => {
                    let f = |a: f64| match op {
                        U::Sin => a.sin(),
                        U::Cos => a.cos(),
                        U::Tan => a.tan(),
                        U::Asin => a.asin(),
                        U::Acos => a.acos(),
                        U::Atan => a.atan(),
                        U::Sinh => a.sinh(),
                        U::Cosh => a.cosh(),
                        U::Asinh => a.asinh(),
                        U::Acosh => a.acosh(),
                        U::Atanh => a.atanh(),
                        U::Floor => a.floor(),
                        U::Ceil => a.ceil(),
                        U::RoundEven => a.round_ties_even(),
                        U::Log1p => a.ln_1p(),
                        U::Expm1 => a.exp_m1(),
                        U::Exp2 => a.exp2(),
                        U::Log2 => a.log2(),
                        U::Rsqrt => 1.0 / a.sqrt(),
                        U::Cbrt => a.cbrt(),
                        U::ErfInv => erf_inv(a),
                        U::ReducePrecision {
                            exponent_bits,
                            mantissa_bits,
                        } => reduce_precision(a, *exponent_bits, *mantissa_bits),
                        U::Abs => a.abs(),
                        U::Sign if a == 0.0 || a.is_nan() => a,
                        U::Sign => a.signum(),
                        _ => unreachable!("{} of a float", op.name()),
                    };

                    match op {
                        U::IsFinite => Int(x.iter().map(|&a| a.is_finite() as i128).collect()),
                        _ => Float(x.iter().map(|&a| round(f(a), dtype)).collect()),
                    }
                }
            }
        }
        Binary(op) => {
            use crate::graph::BinaryOp as B;
            let w = bits(types[0].dtype);

            match (args[0], args[1]) {
                (Int(x), Int(y)) => Int(x
                    .iter()
                    .zip(y)
                    .map(|(&a, &b)| match op {
                        // integer remainder by zero yields the dividend, matching XLA
                        B::Rem if b == 0 => a,
                        B::Rem => a % b,
                        B::Min => a.min(b),
                        B::And => wrap(a & b, dtype),
                        B::Or => wrap(a | b, dtype),
                        B::Xor => wrap(a ^ b, dtype),
                        // shift amounts outside 0..bits saturate: 0, or -1 for arithmetic shifts of negatives
                        _ if b < 0 || b >= i128::from(w) => match op {
                            B::ShiftRightArithmetic if signed_bits(a, w) < 0 => wrap(-1, dtype),
                            _ => 0,
                        },
                        B::ShiftLeft => wrap(a << b, dtype),
                        B::ShiftRightLogical => wrap((unsigned_bits(a, w) >> b) as i128, dtype),
                        B::ShiftRightArithmetic => wrap(signed_bits(a, w) >> b, dtype),
                        _ => unreachable!("{} of integers", op.name()),
                    })
                    .collect()),
                (Float(x), Int(y)) => Float(
                    x.iter()
                        .zip(y)
                        .map(|(&a, &b)| round(a.powf(b as f64), dtype))
                        .collect(),
                ),
                (Float(x), Float(y)) => Float(
                    x.iter()
                        .zip(y)
                        .map(|(&a, &b)| match op {
                            B::Pow => round(a.powf(b), dtype),
                            B::Rem => round(a % b, dtype),
                            B::Min => fmin(a, b),
                            B::Nextafter => nextafter(a, b, dtype),
                            _ => unreachable!("{} of floats", op.name()),
                        })
                        .collect(),
                ),
                _ => unreachable!("operands of one dtype"),
            }
        }
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

            // flat output index each input element accumulates into
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
        CustomCall { label, .. } => {
            panic!("{label}: a custom op's function runs in a plan (lumen.compile), on tensors")
        }
        PallasCall { label, .. } => panic!("{label}: a lumen.pallas kernel runs on a TPU"),
        AllReduce { .. } | AllGather { .. } | ReduceScatter { .. } => {
            panic!(
                "{}: collectives across processes run on CUDA (NCCL)",
                p.name()
            )
        }
        DynamicSlice { slice_sizes } => {
            let x = types[0];
            let starts = starts(x, &args[1..], slice_sizes);
            let mut at = Vec::with_capacity(out.numel());

            if out.numel() > 0 {
                for_each_index(&out.shape, |idx| {
                    at.push(ravel(idx.iter().zip(&starts).map(|(i, s)| i + s), &x.shape));
                });
            }

            match args[0] {
                Int(v) => Int(at.iter().map(|&i| v[i]).collect()),
                Float(v) => Float(at.iter().map(|&i| v[i]).collect()),
            }
        }
        Gather { axis } => {
            let (x, indices) = (types[0], types[1]);
            let picked = taken(args[1], x.shape[*axis]);

            gather(args[0], out, |idx| {
                let k = picked[ravel(
                    idx[*axis..*axis + indices.shape.len()].iter().copied(),
                    &indices.shape,
                )];

                let at = idx[..*axis]
                    .iter()
                    .copied()
                    .chain(std::iter::once(k))
                    .chain(idx[*axis + indices.shape.len()..].iter().copied());

                ravel(at, &x.shape)
            })
        }
        TakeAlongAxis { axis } => {
            let x = types[0];
            let picked = taken(args[1], x.shape[*axis]);

            gather(args[0], out, |idx| {
                let k = picked[ravel(idx.iter().copied(), &out.shape)];

                let at = idx
                    .iter()
                    .enumerate()
                    .map(|(d, &i)| if d == *axis { k } else { i });

                ravel(at, &x.shape)
            })
        }
        ScatterAdd { axis } => {
            let (x, indices, u) = (types[0], types[1], types[2]);
            let picked = taken(args[1], x.shape[*axis]);
            let mut at = Vec::with_capacity(u.numel());

            if u.numel() > 0 {
                for_each_index(&u.shape, |idx| {
                    let k = picked[ravel(
                        idx[*axis..*axis + indices.shape.len()].iter().copied(),
                        &indices.shape,
                    )];

                    let at_x = idx[..*axis]
                        .iter()
                        .copied()
                        .chain(std::iter::once(k))
                        .chain(idx[*axis + indices.shape.len()..].iter().copied());

                    at.push(ravel(at_x, &x.shape));
                });
            }

            match (args[0], args[2]) {
                (Int(v), Int(w)) => {
                    let mut v = v.clone();

                    at.iter()
                        .zip(w)
                        .for_each(|(&i, &e)| v[i] = wrap(v[i] + e, dtype));

                    Int(v)
                }
                (Float(v), Float(w)) => {
                    let mut v = v.clone();

                    at.iter()
                        .zip(w)
                        .for_each(|(&i, &e)| v[i] = round(v[i] + e, dtype));

                    Float(v)
                }
                _ => unreachable!("the updates have the operand's dtype"),
            }
        }
        DynamicUpdateSlice => {
            let (x, u) = (types[0], types[1]);
            let starts = starts(x, &args[2..], &u.shape);
            let mut at = Vec::with_capacity(u.numel());

            if u.numel() > 0 {
                for_each_index(&u.shape, |idx| {
                    at.push(ravel(idx.iter().zip(&starts).map(|(i, s)| i + s), &x.shape));
                });
            }

            match (args[0], args[1]) {
                (Int(v), Int(w)) => {
                    let mut v = v.clone();
                    at.iter().zip(w).for_each(|(&i, &e)| v[i] = e);
                    Int(v)
                }
                (Float(v), Float(w)) => {
                    let mut v = v.clone();
                    at.iter().zip(w).for_each(|(&i, &e)| v[i] = e);
                    Float(v)
                }
                _ => unreachable!("an update of the operand's dtype"),
            }
        }
        Cumsum { axis, reverse, .. } => {
            // walk each line along `axis` of the `(outer, n, b)` view, backwards when reversed
            let shape = &types[0].shape;
            let n = shape[*axis];
            let b: usize = shape[axis + 1..].iter().product();
            let lines = out.numel().checked_div(n).unwrap_or(0);

            let order = |line: usize| {
                let (i, j) = (line / b, line % b);

                (0..n).map(move |k| {
                    let k = if *reverse { n - 1 - k } else { k };
                    (i * n + k) * b + j
                })
            };

            match args[0] {
                Int(v) => {
                    let mut acc = vec![0; v.len()];

                    for line in 0..lines {
                        let mut s = 0;

                        for e in order(line) {
                            s = wrap(s + v[e], dtype);
                            acc[e] = s;
                        }
                    }

                    Int(acc)
                }
                Float(v) => {
                    let mut acc = vec![0.0; v.len()];

                    for line in 0..lines {
                        let mut s = 0.0;

                        for e in order(line) {
                            s = round(s + v[e], dtype);
                            acc[e] = s;
                        }
                    }

                    Float(acc)
                }
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
            // per output element, the (lhs, rhs) flat offset pairs summed over the contracting dims
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
                        .map(|t| {
                            // round each product and partial sum to the accumulator dtype, as a kernel would
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
        Reshape { .. } | HostWait | DeviceWait | ToHost | ToDevice => args[0].clone(),
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
        Concatenate { dimension, .. } => {
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

            // per operand: its start along `d` and its offset in the flattened concatenation
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
        RandomBits { offset, .. } => {
            let (Int(seed), Int(start)) = (args[0], args[1]) else {
                unreachable!("uint64 scalars")
            };

            let (seed, start) = (seed[0] as u64, (start[0] as u64).wrapping_add(*offset));

            Int((0..out.numel() as u64)
                .map(|i| philox_bits(seed, start.wrapping_add(i)) as i128)
                .collect())
        }
        LinearCrossEntropy => {
            let (Float(h), Float(w), Int(target)) = (args[0], args[1], args[2]) else {
                unreachable!("float rows and weights, int classes")
            };

            let (b, d, v) = (types[0].shape[0], types[0].shape[1], types[1].shape[0]);
            // float32 logits, then loss = logsumexp - logit[target]; output is losses followed by logsumexps
            let (mut loss, mut lse) = (Vec::with_capacity(b), Vec::with_capacity(b));

            for i in 0..b {
                let logits: Vec<f64> = (0..v)
                    .map(|j| {
                        (0..d).fold(0.0, |acc, k| {
                            let product = round(h[i * d + k] * w[j * d + k], DType::F32);
                            round(acc + product, DType::F32)
                        })
                    })
                    .collect();

                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = logits.iter().map(|&x| (x - max).exp()).sum();
                let class = target[i].clamp(0, v as i128 - 1) as usize;
                let l = round(max + sum.ln(), DType::F32);
                loss.push(round(l - logits[class], DType::F32));
                lse.push(l);
            }

            loss.extend(lse);
            Float(loss)
        }
        Fusion { body, .. } => {
            let inputs = args.iter().map(|&v| v.clone()).collect();
            eval_graph(body, inputs).remove(0)
        }
        FusionOutput { .. } => unreachable!("evaluated with its fusion (eval_graph)"),
    }
}

/// Elementwise select: `x` where `pred` is nonzero, else `y`.
fn pick<T: Copy>(pred: &[i128], x: &[T], y: &[T]) -> Vec<T> {
    pred.iter()
        .zip(x.iter().zip(y))
        .map(|(&p, (&a, &b))| if p != 0 { a } else { b })
        .collect()
}

/// Returns the first 32-bit word of Philox4x32-10 for a 64-bit key and counter.
pub(crate) fn philox_bits(key: u64, counter: u64) -> u32 {
    let counter = [counter as u32, (counter >> 32) as u32, 0, 0];
    philox4x32([key as u32, (key >> 32) as u32], counter)[0]
}

/// Philox4x32 with 10 rounds: maps a 2-word key and 4-word counter to 4 random words.
pub(crate) fn philox4x32(mut k: [u32; 2], mut c: [u32; 4]) -> [u32; 4] {
    let mulhilo = |a: u32, b: u32| {
        let p = a as u64 * b as u64;
        ((p >> 32) as u32, p as u32)
    };

    for round in 0..10 {
        if round > 0 {
            k = [
                k[0].wrapping_add(0x9E37_79B9),
                k[1].wrapping_add(0xBB67_AE85),
            ];
        }

        let (hi0, lo0) = mulhilo(0xD251_1F53, c[0]);
        let (hi1, lo1) = mulhilo(0xCD9E_8D57, c[2]);
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
    }

    c
}
