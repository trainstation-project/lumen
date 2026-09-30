use super::Primitive::*;
use super::{Graph, Primitive, TensorType, reference};
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
fn reduce_drops_axes() {
    let x = ty(DType::F32, &[2, 3, 4]);
    let sum = ReduceSum { axes: vec![0, 2] };
    assert_eq!(infer(sum, std::slice::from_ref(&x)).unwrap().shape, [3]);
    assert!(infer(ReduceMax { axes: vec![3] }, std::slice::from_ref(&x)).is_err());
    assert!(infer(ReduceMax { axes: vec![1, 1] }, &[x]).is_err());
}

#[test]
fn dot_general_shape() {
    // Batched matmul: [b, m, k] x [b, k, n] -> [b, m, n].
    let p = DotGeneral {
        lhs_contracting: vec![2],
        rhs_contracting: vec![1],
        lhs_batch: vec![0],
        rhs_batch: vec![0],
    };
    let (l, r) = (ty(DType::F32, &[5, 2, 3]), ty(DType::F32, &[5, 3, 4]));
    assert_eq!(infer(p.clone(), &[l.clone(), r]).unwrap().shape, [5, 2, 4]);
    assert!(infer(p, &[l.clone(), ty(DType::F32, &[5, 2, 4])]).is_err());
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
    let s = g.apply(ReduceSum { axes: vec![1] }, &[x]).unwrap();
    g.set_outputs(&[s]).unwrap();
    assert_eq!(
        g.to_string(),
        "{ lambda %0:f32[2,3]. let\n    %1:f32[2] = reduce_sum[axes=(1,)] %0\n  in (%1) }"
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
    let s = g.apply(ReduceSum { axes: vec![1] }, &[e]).unwrap();
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
    let to_i8 = ConvertElementType {
        new_dtype: DType::I8,
    };
    assert_eq!(
        run(to_i8, std::slice::from_ref(&f)).to_vec::<i8>(),
        [127, -1, 0]
    );
    let to_f16 = ConvertElementType {
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

// ---------------------------------------------------------------------
// plans
// ---------------------------------------------------------------------

use super::Plan;
use super::plan::Buffer;

/// Values in `[-2, 2)` from a fixed-seed LCG, so tests are deterministic.
fn data(shape: &[usize], seed: u64) -> Tensor {
    let mut state = seed;
    let values: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0
        })
        .collect();
    Tensor::from_slice(&values, DType::F32).reshape(shape)
}

/// `plan.run` agrees exactly with `reference::run`.
fn check_against_reference(g: &Graph, inputs: &[Tensor]) -> Plan {
    let plan = Plan::compile(g);
    let expected = reference::run(g, inputs).unwrap();
    let actual = plan.run(inputs).unwrap();
    assert_eq!(expected.len(), actual.len());
    for (e, a) in expected.iter().zip(&actual) {
        assert_eq!(e.shape(), a.shape());
        assert_eq!(e.to_vec::<f32>(), a.to_vec::<f32>(), "{plan}");
    }
    plan
}

/// The MLP block from run_graph.py: matmul, relu, matmul, softmax.
fn mlp() -> Graph {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[4, 8]));
    let w1 = g.input(ty(DType::F32, &[8, 16]));
    let w2 = g.input(ty(DType::F32, &[16, 3]));
    let matmul = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
    };
    let h = g.apply(matmul.clone(), &[x, w1]).unwrap();
    let zero = Full {
        shape: vec![4, 16],
        fill_value: Scalar::Float(0.0),
        dtype: DType::F32,
    };
    let zero = g.apply(zero, &[]).unwrap();
    let h = g.apply(Max, &[h, zero]).unwrap();
    let z = g.apply(matmul, &[h, w2]).unwrap();
    let m = g.apply(ReduceMax { axes: vec![1] }, &[z]).unwrap();
    let m = g
        .apply(
            Reshape {
                new_sizes: vec![4, 1],
            },
            &[m],
        )
        .unwrap();
    let bcast = BroadcastInDim {
        shape: vec![4, 3],
        broadcast_dimensions: vec![0, 1],
    };
    let m = g.apply(bcast.clone(), &[m]).unwrap();
    let e = g.apply(Sub, &[z, m]).unwrap();
    let e = g.apply(Exp, &[e]).unwrap();
    let s = g.apply(ReduceSum { axes: vec![1] }, &[e]).unwrap();
    let s = g
        .apply(
            Reshape {
                new_sizes: vec![4, 1],
            },
            &[s],
        )
        .unwrap();
    let s = g.apply(bcast, &[s]).unwrap();
    let y = g.apply(Div, &[e, s]).unwrap();
    g.set_outputs(&[y]).unwrap();
    g
}

#[test]
fn plan_matches_reference() {
    let inputs = [data(&[4, 8], 1), data(&[8, 16], 2), data(&[16, 3], 3)];
    let plan = check_against_reference(&mlp(), &inputs);
    // Reshapes alias their operands: no step.
    assert!(
        plan.steps()
            .iter()
            .all(|s| !matches!(s.primitive, Reshape { .. }))
    );
    assert_eq!(plan.steps().last().unwrap().output.0, Buffer::Output(0));
}

#[test]
fn plan_reuses_workspace() {
    // x -> exp -> exp -> ... (10 times): only two intermediates are ever
    // live at once, so the workspace holds two, not nine.
    let mut g = Graph::new();
    let mut v = g.input(ty(DType::F32, &[256]));
    for _ in 0..10 {
        v = g.apply(Tanh, &[v]).unwrap();
    }
    g.set_outputs(&[v]).unwrap();
    let plan = check_against_reference(&g, &[data(&[256], 4)]);
    assert_eq!(plan.workspace_bytes(), 2 * 1024);
}

#[test]
fn plan_copies_aliased_outputs_and_drops_dead_code() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[2, 3]));
    let e = g.apply(Exp, &[x]).unwrap();
    let _dead = g.apply(Log, &[e]).unwrap();
    let flat = g.apply(Reshape { new_sizes: vec![6] }, &[e]).unwrap();
    g.set_outputs(&[e, flat, x]).unwrap();
    let plan = check_against_reference(&g, &[data(&[2, 3], 5)]);
    // exp into out0, then copies of it (reshaped) into out1 and of the
    // input into out2; the log is dropped.
    let ops: Vec<_> = plan.steps().iter().map(|s| s.primitive.name()).collect();
    assert_eq!(ops, ["exp", "reshape", "reshape"]);
    assert_eq!(plan.workspace_bytes(), 0);
}
