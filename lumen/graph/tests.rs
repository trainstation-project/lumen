use super::Primitive::*;
use super::{Graph, Primitive, TensorType};
use crate::ops::reference;
use crate::tensor::dtype::bf16;
use crate::{DType, Scalar, Tensor};

fn ty(dtype: DType, shape: &[usize]) -> TensorType {
    TensorType::new(dtype, shape)
}

fn infer(p: Primitive, args: &[TensorType]) -> Result<TensorType, String> {
    p.infer(&args.iter().collect::<Vec<_>>())
}

// ---------------------------------------------------------------------
// shape and dtype rules
// ---------------------------------------------------------------------

#[test]
fn elementwise_is_strict() {
    let x = ty(DType::F32, &[2, 3]);
    assert_eq!(infer(Add, &[x.clone(), x.clone()]).unwrap(), x);
    assert_eq!(
        infer(Lt, &[x.clone(), x.clone()]).unwrap().dtype,
        DType::Bool
    );
    // No implicit broadcasting or promotion.
    assert!(infer(Add, &[x.clone(), ty(DType::F32, &[3])]).is_err());
    assert!(infer(Add, &[x.clone(), ty(DType::F16, &[2, 3])]).is_err());
    assert!(infer(Exp, &[ty(DType::I32, &[2])]).is_err());
    // Nothing computes bfloat16 math as traced: convert it first.
    for p in [Exp, Log, Sqrt, Tanh, Logistic] {
        let e = infer(p, &[ty(DType::BF16, &[2])]).unwrap_err();
        assert!(e.contains("convert to float32"), "{e}");
    }
    assert!(infer(Neg, &[ty(DType::BF16, &[2])]).is_ok());
    assert!(infer(Add, &[ty(DType::Bool, &[2]), ty(DType::Bool, &[2])]).is_err());
    assert!(infer(Add, &[x]).is_err());
}

#[test]
fn select_rules() {
    let (p, x) = (ty(DType::Bool, &[4]), ty(DType::F32, &[4]));
    assert_eq!(
        infer(Select, &[p.clone(), x.clone(), x.clone()]).unwrap(),
        x
    );
    assert!(infer(Select, &[x.clone(), x.clone(), x.clone()]).is_err());
    assert!(infer(Select, &[p, x, ty(DType::F64, &[4])]).is_err());
}

#[test]
fn slice_rules() {
    let x = ty(DType::F32, &[4, 6]);
    let slice = |s: &[usize], l: &[usize]| Slice {
        start_indices: s.to_vec(),
        limit_indices: l.to_vec(),
        view: false,
    };
    let y = infer(slice(&[1, 2], &[3, 6]), std::slice::from_ref(&x)).unwrap();
    assert_eq!((y.dtype, y.shape), (DType::F32, vec![2, 4]));
    assert_eq!(
        infer(slice(&[2, 2], &[2, 2]), std::slice::from_ref(&x))
            .unwrap()
            .shape,
        [0, 0]
    );
    assert!(infer(slice(&[0], &[4]), std::slice::from_ref(&x)).is_err());
    assert!(infer(slice(&[3, 0], &[2, 6]), std::slice::from_ref(&x)).is_err());
    assert!(infer(slice(&[0, 0], &[4, 7]), &[x]).is_err());
}

#[test]
fn concatenate_rules() {
    let (a, b) = (ty(DType::F32, &[2, 3, 4]), ty(DType::F32, &[2, 5, 4]));
    let cat = |dimension| Concatenate {
        dimension,
        in_place: false,
    };
    let y = infer(cat(1), &[a.clone(), b.clone(), a.clone()]).unwrap();
    assert_eq!((y.dtype, y.shape), (DType::F32, vec![2, 11, 4]));
    assert_eq!(
        infer(cat(0), std::slice::from_ref(&a)).unwrap().shape,
        [2, 3, 4]
    );
    assert!(infer(cat(0), &[a.clone(), b.clone()]).is_err());
    assert!(infer(cat(3), std::slice::from_ref(&a)).is_err());
    assert!(infer(cat(1), &[a.clone(), ty(DType::F16, &[2, 3, 4])]).is_err());
    assert!(infer(cat(1), &[a, ty(DType::F32, &[2, 3])]).is_err());
    assert!(infer(cat(0), &[]).is_err());
}

#[test]
fn reduce_drops_axes() {
    let x = ty(DType::F32, &[2, 3, 4]);
    let sum = ReduceSum {
        axes: vec![0, 2],
        accum_dtype: DType::F32,
    };
    assert_eq!(infer(sum, std::slice::from_ref(&x)).unwrap().shape, [3]);
    assert!(infer(ReduceMax { axes: vec![3] }, std::slice::from_ref(&x)).is_err());
    assert!(infer(ReduceMax { axes: vec![1, 1] }, &[x]).is_err());
    // A sum accumulates in its accum_dtype, its result's: the operand's,
    // or float32 for narrower floats.
    let sum = |accum_dtype| ReduceSum {
        axes: vec![0],
        accum_dtype,
    };
    let wide = infer(sum(DType::F32), &[ty(DType::BF16, &[4])]).unwrap();
    assert_eq!(wide.dtype, DType::F32);
    for (d, accum) in [
        (DType::F32, DType::F16),
        (DType::F32, DType::F64),
        (DType::I8, DType::I32),
    ] {
        let e = infer(sum(accum), &[ty(d, &[4])]).unwrap_err();
        assert!(e.contains("accum_dtype"), "{e}");
    }
    assert_eq!(
        sum(DType::F32).to_string(),
        "reduce_sum[axes=(0,) accum_dtype=f32]"
    );
}

#[test]
fn dot_general_shape() {
    // Batched matmul: [b, m, k] x [b, k, n] -> [b, m, n].
    let p = DotGeneral {
        lhs_contracting: vec![2],
        rhs_contracting: vec![1],
        lhs_batch: vec![0],
        rhs_batch: vec![0],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };
    let (l, r) = (ty(DType::F32, &[5, 2, 3]), ty(DType::F32, &[5, 3, 4]));
    assert_eq!(infer(p.clone(), &[l.clone(), r]).unwrap().shape, [5, 2, 4]);
    assert!(infer(p, &[l.clone(), ty(DType::F32, &[5, 2, 4])]).is_err());
}

#[test]
fn dot_general_accumulates_in_accum_dtype() {
    let dot = |accum_dtype, output_dtype| DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype,
        output_dtype,
    };
    let operands = |d| [ty(d, &[2, 3]), ty(d, &[3, 4])];
    // It accumulates in the operands' dtype, or float32 for narrower
    // floats, and outputs the operands' dtype or the accumulation's.
    for (d, accum, output) in [
        (DType::F32, DType::F32, DType::F32),
        (DType::BF16, DType::BF16, DType::BF16),
        (DType::BF16, DType::F32, DType::BF16),
        (DType::BF16, DType::F32, DType::F32),
        (DType::F16, DType::F32, DType::F16),
        (DType::I32, DType::I32, DType::I32),
    ] {
        assert_eq!(
            infer(dot(accum, output), &operands(d)).unwrap().dtype,
            output
        );
    }
    for (d, accum, output, word) in [
        (DType::F32, DType::F16, DType::F32, "accum_dtype"),
        (DType::F32, DType::F64, DType::F32, "accum_dtype"),
        (DType::BF16, DType::F16, DType::BF16, "accum_dtype"),
        (DType::I8, DType::I32, DType::I8, "accum_dtype"),
        (DType::BF16, DType::F32, DType::F16, "output_dtype"),
        (DType::BF16, DType::BF16, DType::F32, "output_dtype"),
    ] {
        let e = infer(dot(accum, output), &operands(d)).unwrap_err();
        assert!(e.contains(word), "{e}");
    }
    assert_eq!(
        dot(DType::F32, DType::BF16).to_string(),
        "dot_general[dimension_numbers=(((1,), (0,)), ((), ())) accum_dtype=f32 output_dtype=bf16]"
    );
    // Each product and partial sum rounded to accum_dtype, the result to
    // output_dtype once: bfloat16 holds 1 + 2^-8 as 1 (each 2^-8 added is
    // lost); float32 adds both, 1 + 2^-7, which bfloat16 holds too.
    let b = |v: f32| bf16::from_f32(v);
    let x =
        Tensor::from_slice(&[b(1.0), b(1.0 / 256.0), b(1.0 / 256.0)], DType::BF16).reshape(&[1, 3]);
    let y = Tensor::from_slice(&[b(1.0); 3], DType::BF16).reshape(&[3, 1]);
    let pair = [x, y];
    assert_eq!(
        run(dot(DType::BF16, DType::BF16), &pair).to_vec::<bf16>(),
        [b(1.0)]
    );
    let narrow = run(dot(DType::F32, DType::BF16), &pair).to_vec::<bf16>();
    assert_eq!(narrow, [b(1.0 + 1.0 / 128.0)]);
    assert_eq!(
        run(dot(DType::F32, DType::F32), &pair).to_vec::<f32>(),
        [1.0 + 1.0 / 128.0]
    );
}

#[test]
fn layout_primitives() {
    let x = ty(DType::I32, &[3, 1]);
    let b = BroadcastInDim {
        shape: vec![2, 3, 4],
        broadcast_dimensions: vec![1, 2],
    };
    assert_eq!(infer(b, std::slice::from_ref(&x)).unwrap().shape, [2, 3, 4]);
    let bad = BroadcastInDim {
        shape: vec![3, 4],
        broadcast_dimensions: vec![1, 0],
    };
    assert!(infer(bad, std::slice::from_ref(&x)).is_err());
    let t = Transpose {
        permutation: vec![1, 0],
    };
    assert_eq!(infer(t, std::slice::from_ref(&x)).unwrap().shape, [1, 3]);
    let r = Reshape { new_sizes: vec![3] };
    assert_eq!(infer(r, std::slice::from_ref(&x)).unwrap().shape, [3]);
    assert!(infer(Reshape { new_sizes: vec![4] }, &[x]).is_err());
}

#[test]
fn display_is_jaxpr_like() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[2, 3]));
    let s = g
        .apply(
            ReduceSum {
                axes: vec![1],
                accum_dtype: DType::F32,
            },
            &[x],
        )
        .unwrap();
    g.set_outputs(&[s]).unwrap();
    assert_eq!(
        g.to_string(),
        "{ lambda %0:f32[2,3]. let\n    %1:f32[2] = reduce_sum[axes=(1,) accum_dtype=f32] %0\n  in (%1) }"
    );
    assert!(g.apply(Neg, &[7]).is_err());
}

// ---------------------------------------------------------------------
// reference executor
// ---------------------------------------------------------------------

/// Run a one-node graph of `p` on `inputs`.
fn run(p: Primitive, inputs: &[Tensor]) -> Tensor {
    let mut g = Graph::new();
    let vars: Vec<_> = inputs
        .iter()
        .map(|t| g.input(ty(t.dtype(), t.shape())))
        .collect();
    let out = g.apply(p, &vars).unwrap();
    g.set_outputs(&[out]).unwrap();
    reference::run(&g, inputs).unwrap().remove(0)
}

#[test]
fn softmax_graph() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[2, 3]));
    let m = g.apply(ReduceMax { axes: vec![1] }, &[x]).unwrap();
    let bcast = BroadcastInDim {
        shape: vec![2, 3],
        broadcast_dimensions: vec![0],
    };
    let m = g.apply(bcast.clone(), &[m]).unwrap();
    let e = g.apply(Sub, &[x, m]).unwrap();
    let e = g.apply(Exp, &[e]).unwrap();
    let s = g
        .apply(
            ReduceSum {
                axes: vec![1],
                accum_dtype: DType::F32,
            },
            &[e],
        )
        .unwrap();
    let s = g.apply(bcast, &[s]).unwrap();
    let y = g.apply(Div, &[e, s]).unwrap();
    g.set_outputs(&[y]).unwrap();

    let data = [1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0];
    let x = Tensor::from_slice(&data, DType::F32).reshape(&[2, 3]);
    let y = reference::run(&g, &[x]).unwrap().remove(0).to_vec::<f32>();
    let e: Vec<f32> = [-2.0f32, -1.0, 0.0].iter().map(|v| v.exp()).collect();
    let total: f32 = e.iter().sum();
    for (a, b) in y[..3].iter().zip(&e) {
        assert!((a - b / total).abs() < 1e-6);
    }
    assert_eq!(y[3..], [1.0 / 3.0; 3]);
}

#[test]
fn dot_general_values() {
    let l = Tensor::from_slice(&[1i32, 2, 3, 4, 5, 6], DType::I32).reshape(&[2, 3]);
    let r = Tensor::from_slice(&[1i32, 0, 0, 1, 1, 1], DType::I32).reshape(&[3, 2]);
    let p = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::I32,
        output_dtype: DType::I32,
    };
    let y = run(p, &[l, r]);
    assert_eq!(y.shape(), [2, 2]);
    assert_eq!(y.to_vec::<i32>(), [4, 5, 10, 11]);
}

#[test]
fn integers_wrap_and_converts_saturate() {
    let x = Tensor::from_slice(&[250u8, 3], DType::U8);
    assert_eq!(run(Add, &[x.clone(), x]).to_vec::<u8>(), [244, 6]);
    let big = Tensor::from_slice(&[u64::MAX, 1], DType::U64);
    let one = Tensor::from_slice(&[1u64, 2], DType::U64);
    assert_eq!(run(Lt, &[one, big.clone()]).to_vec::<bool>(), [true, false]);
    assert_eq!(run(Max, &[big.clone(), big]).to_vec::<u64>(), [u64::MAX, 1]);
    let f = Tensor::from_slice(&[300.0f32, -1.5, f32::NAN], DType::F32);
    let to_i8 = Cast {
        new_dtype: DType::I8,
    };
    assert_eq!(
        run(to_i8, std::slice::from_ref(&f)).to_vec::<i8>(),
        [127, -1, 0]
    );
    let to_f16 = Cast {
        new_dtype: DType::F16,
    };
    assert!(run(to_f16, &[f]).to_vec::<half::f16>()[0] == half::f16::from_f32(300.0));
}

#[test]
fn layout_values() {
    let x = Tensor::from_slice(&[1i64, 2, 3, 4, 5, 6], DType::I64).reshape(&[2, 3]);
    let t = Transpose {
        permutation: vec![1, 0],
    };
    assert_eq!(run(t, &[x]).to_vec::<i64>(), [1, 4, 2, 5, 3, 6]);
    let v = Tensor::from_slice(&[7i64, 8], DType::I64);
    let b = BroadcastInDim {
        shape: vec![2, 3],
        broadcast_dimensions: vec![0],
    };
    assert_eq!(run(b, &[v]).to_vec::<i64>(), [7, 7, 7, 8, 8, 8]);
    let iota = Iota {
        dtype: DType::F32,
        shape: vec![2, 3],
        dimension: 1,
    };
    assert_eq!(
        run(iota, &[]).to_vec::<f32>(),
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    );
    let full = Full {
        shape: vec![2],
        fill_value: Scalar::Float(2.5),
        dtype: DType::I32,
    };
    assert_eq!(run(full, &[]).to_vec::<i32>(), [2, 2]);
}

#[test]
fn rejects_mismatched_inputs() {
    let mut g = Graph::new();
    g.input(ty(DType::F32, &[2]));
    let x = Tensor::from_slice(&[1.0f64, 2.0], DType::F64);
    assert!(reference::run(&g, &[x]).is_err());
    assert!(reference::run(&g, &[]).is_err());
}
