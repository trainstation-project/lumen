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
    let cat = |dimension| Concatenate { dimension };
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
pub(crate) fn data(shape: &[usize], seed: u64) -> Tensor {
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
pub(crate) fn mlp() -> Graph {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[4, 8]));
    let w1 = g.input(ty(DType::F32, &[8, 16]));
    let w2 = g.input(ty(DType::F32, &[16, 3]));
    let matmul = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
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
    let s = g
        .apply(
            ReduceSum {
                axes: vec![1],
                accum_dtype: DType::F32,
            },
            &[e],
        )
        .unwrap();
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

/// A plan that owns its memory: the input that is not a parameter and
/// the outputs in the workspace (an output that is that input in its
/// place, one that is a parameter the parameter); each run copies the
/// input in and overwrites the outputs, views of the workspace.
#[test]
fn owned_plans_place_inputs_and_outputs_in_the_workspace() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[4]));
    let w = g.input(ty(DType::F32, &[4]));
    let y = g.apply(Mul, &[x, w]).unwrap();
    g.set_outputs(&[y, x, w]).unwrap();
    let options = super::PlanOptions {
        parameters: Some(vec![false, true]),
        ..Default::default()
    };
    let plan = Plan::compile_with(&g, &options);
    assert!(matches!(plan.steps()[0].inputs[0].0, Buffer::Workspace(_)));
    assert_eq!(plan.steps()[0].inputs[1].0, Buffer::Input(1));
    let workspace = Tensor::zeros(&[plan.workspace_bytes()], DType::U8);
    let w = Tensor::from_slice(&[2.0f32; 4], DType::F32);
    let run = |x: &[f32]| {
        plan.run_in(&workspace, &[Tensor::from_slice(x, DType::F32), w.clone()])
            .unwrap()
    };
    let first = run(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(first[0].to_vec::<f32>(), [2.0, 4.0, 6.0, 8.0]);
    assert_eq!(first[1].to_vec::<f32>(), [1.0, 2.0, 3.0, 4.0]);
    assert!(first[0].shares_storage_with(&workspace) && first[2].shares_storage_with(&w));
    let second = run(&[0.0, 1.0, 0.0, 1.0]);
    assert_eq!(second[0].to_vec::<f32>(), [0.0, 2.0, 0.0, 2.0]);
    assert_eq!(first[0].to_vec::<f32>(), [0.0, 2.0, 0.0, 2.0]); // overwritten
    let small = Tensor::zeros(&[plan.workspace_bytes() - 1], DType::U8);
    assert!(plan.run_in(&small, &[w.clone(), w.clone()]).is_err());
}

/// A plan of `g` donating its first input, and the output's buffer.
fn donating(g: &Graph) -> (Plan, Buffer) {
    let options = super::PlanOptions {
        scratch: None,
        donate: vec![0],
        parameters: None,
        views: Vec::new(),
        scalars: Vec::new(),
    };
    let plan = Plan::compile_with(g, &options);
    let out = plan
        .steps()
        .iter()
        .map(|s| s.output.0)
        .find(|b| !matches!(b, Buffer::Workspace(_)));
    (plan, out.expect("a step writes the output"))
}

#[test]
fn donated_inputs_hold_outputs_written_in_place() {
    let ty = TensorType::new(DType::F32, &[2, 3]);
    // exp(x) + x: the add reads x at the element it writes.
    let mut g = Graph::new();
    let x = g.input(ty.clone());
    let e = g.apply(Exp, &[x]).unwrap();
    let y = g.apply(Add, &[e, x]).unwrap();
    g.set_outputs(&[y]).unwrap();
    let (plan, out) = donating(&g);
    assert_eq!(out, Buffer::Input(0), "{plan}");
    let input = data(&[2, 3], 1);
    let expected = reference::run(&g, std::slice::from_ref(&input)).unwrap();
    let result = plan.run(std::slice::from_ref(&input)).unwrap();
    assert!(result[0].shares_storage_with(&input));
    assert_eq!(result[0].to_vec::<f32>(), expected[0].to_vec::<f32>());

    // A transpose reads other elements: it cannot overwrite its operand.
    let mut g = Graph::new();
    let x = g.input(TensorType::new(DType::F32, &[3, 3]));
    let t = g
        .apply(
            Transpose {
                permutation: vec![1, 0],
            },
            &[x],
        )
        .unwrap();
    g.set_outputs(&[t]).unwrap();
    assert_eq!(donating(&g).1, Buffer::Output(0));

    // exp(x) is written before x is read again: it takes new memory, and
    // x * 2, the last reader, takes x's.
    let mut g = Graph::new();
    let x = g.input(ty.clone());
    let e = g.apply(Exp, &[x]).unwrap();
    let two = g
        .apply(
            Full {
                shape: vec![2, 3],
                fill_value: Scalar::Float(2.0),
                dtype: DType::F32,
            },
            &[],
        )
        .unwrap();
    let d = g.apply(Mul, &[x, two]).unwrap();
    g.set_outputs(&[e, d]).unwrap();
    let plan = Plan::compile_with(
        &g,
        &super::PlanOptions {
            scratch: None,
            donate: vec![0],
            parameters: None,
            views: Vec::new(),
            scalars: Vec::new(),
        },
    );
    let outputs: Vec<Buffer> = plan
        .steps()
        .iter()
        .map(|s| s.output.0)
        .filter(|b| !matches!(b, Buffer::Workspace(_)))
        .collect();
    assert_eq!(outputs, [Buffer::Output(0), Buffer::Input(0)], "{plan}");
}

#[test]
fn kernel_scratch_is_placed_in_the_workspace() {
    let mut g = Graph::new();
    let x = g.input(TensorType::new(DType::F32, &[4, 64]));
    let e = g.apply(Exp, &[x]).unwrap();
    let s = g
        .apply(
            ReduceSum {
                axes: vec![1],
                accum_dtype: DType::F32,
            },
            &[e],
        )
        .unwrap();
    g.set_outputs(&[s]).unwrap();
    let options = super::PlanOptions {
        scratch: Some(|p, _, _| {
            if matches!(p, ReduceSum { .. }) {
                100
            } else {
                0
            }
        }),
        donate: Vec::new(),
        parameters: None,
        views: Vec::new(),
        scalars: Vec::new(),
    };
    let plan = Plan::compile_with(&g, &options);
    let reduce = &plan.steps()[1];
    let (offset, bytes) = reduce.scratch.expect("the reduction's scratch");
    assert_eq!(bytes, 100);
    // Clear of the exp's result, which the reduction reads.
    let Buffer::Workspace(e_at) = reduce.inputs[0].0 else {
        panic!("{plan}")
    };
    assert!(
        offset >= e_at + 4 * 64 * 4 || offset + bytes <= e_at,
        "{plan}"
    );
    assert!(plan.workspace_bytes() >= offset + bytes);
    assert!(plan.to_string().contains("(scratch ws+"));
}

#[test]
fn plans_run_on_meta_tensors_compute_nothing() {
    let g = mlp();
    let meta = |t: Tensor| t.to(crate::Device::Meta);
    let inputs = [
        meta(data(&[4, 8], 1)),
        meta(data(&[8, 16], 2)),
        meta(data(&[16, 3], 3)),
    ];
    let outputs = Plan::compile(&g)
        .run_on(&inputs, crate::Device::Meta)
        .unwrap();
    let out = &outputs[0];
    assert_eq!(
        (out.device(), out.shape(), out.dtype()),
        (crate::Device::Meta, &[4, 3][..], DType::F32)
    );
    // Inputs must be on the device asked for.
    assert!(
        Plan::compile(&g)
            .run_on(&inputs, crate::Device::Cpu)
            .is_err()
    );
}

/// Plans on MPS, whose steps are Metal kernels, against the reference
/// executor, for every primitive and dtype (MPS has no float64).
#[cfg(lumen_mps_linked)]
pub(crate) mod mps {
    use super::super::{Graph, Plan, Primitive, TensorType};
    use super::Primitive::*;
    use super::{data, mlp};
    use crate::ops::reference;
    use crate::tensor::dtype::dispatch_dtype;
    use crate::{DType, Device, Element, Scalar, Tensor, TensorOptions};

    const DTYPES: [DType; 12] = [
        DType::Bool,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
    ];

    pub(crate) fn available() -> bool {
        crate::device::mps::is_available()
    }

    /// `shape` of `dtype` from a fixed seed: floats in [-4, 4), integers in
    /// [-100, 100) (wrapped for unsigned dtypes), bools alternating.
    pub(crate) fn values(dtype: DType, shape: &[usize], seed: u64) -> Tensor {
        let x = data(shape, seed).to_vec::<f32>();
        dispatch_dtype!(dtype, T => {
            let v: Vec<T> = x
                .iter()
                .map(|&f| {
                    T::from_scalar(if dtype.is_float() {
                        Scalar::Float(f as f64 * 2.0)
                    } else {
                        Scalar::Int((f * 50.0) as i64)
                    })
                })
                .collect();
            Tensor::from_slice(&v, dtype).reshape(shape)
        })
    }

    /// Values of 16-bit float `dtype` that sum exactly in it in any order:
    /// -1, 0 or 1, one element in `period` nonzero (a sum of up to 128 of
    /// them a sum of about 128 at most: integers both 16-bit floats hold).
    /// They accumulate in their dtype, so another order than the
    /// reference's rounds differently; on these, no order rounds.
    pub(crate) fn exact_values(dtype: DType, shape: &[usize], seed: u64, period: usize) -> Tensor {
        let x = data(shape, seed).to_vec::<f32>();
        let v: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &f)| match i % period {
                0 => (f / 2.0).round() as f64,
                _ => 0.0,
            })
            .collect();
        dispatch_dtype!(dtype, T => {
            let v: Vec<T> = v.iter().map(|&f| T::from_scalar(Scalar::Float(f))).collect();
            Tensor::from_slice(&v, dtype).reshape(shape)
        })
    }

    fn as_f64(t: &Tensor) -> Vec<f64> {
        dispatch_dtype!(t.dtype(), T => t.to_vec::<T>().into_iter().map(|v| v.to_scalar().to_f64()).collect())
    }

    /// The graph compiled for MPS (fused, `crate::compiler::mps`) agrees
    /// with the reference on the CPU: exactly for integers and bools, within
    /// the dtype's rounding for floats (Metal's math functions and the
    /// reference's, in double, differ in the last place).
    pub(crate) fn check(g: &Graph, inputs: &[Tensor]) {
        check_within(g, inputs, 0.0);
    }

    /// [`check`], allowing floats a further `slack` of absolute error: the
    /// float accumulation error of a large sum.
    pub(crate) fn check_within(g: &Graph, inputs: &[Tensor], slack: f64) {
        let expected = reference::run(g, inputs).unwrap();
        let on_mps: Vec<Tensor> = inputs.iter().map(|t| t.to(Device::Mps)).collect();
        let plan = crate::compiler::compile(g, Device::Mps).unwrap();
        let actual = plan.run(&on_mps).unwrap();
        for (e, a) in expected.iter().zip(&actual) {
            assert_eq!(a.device(), Device::Mps);
            assert_eq!((e.dtype(), e.shape()), (a.dtype(), a.shape()));
            let tol = match e.dtype() {
                DType::F32 => 1e-5,
                DType::F16 => 2e-3,
                DType::BF16 => 1e-2,
                _ => 0.0,
            };
            for (x, y) in as_f64(e).into_iter().zip(as_f64(a)) {
                let bound = tol * (1.0 + x.abs()) + if tol > 0.0 { slack } else { 0.0 };
                let close = x == y || (x.is_nan() && y.is_nan()) || (x - y).abs() <= bound;
                assert!(close, "{plan}\nexpected {x}, got {y}");
            }
        }
    }

    /// `exp(x - max) / sum(exp(x - max))` over the last dimension of `x`, of
    /// `dtype` and `shape`, as `softmax` traces it.
    fn written_softmax(dtype: DType, shape: &[usize]) -> Graph {
        let mut g = Graph::new();
        let x = g.input(ty(dtype, shape));
        let last = shape.len() - 1;
        let bcast = BroadcastInDim {
            shape: shape.to_vec(),
            broadcast_dimensions: (0..last).collect(),
        };
        let m = g.apply(ReduceMax { axes: vec![last] }, &[x]).unwrap();
        let m = g.apply(bcast.clone(), &[m]).unwrap();
        let d = g.apply(Sub, &[x, m]).unwrap();
        let e = g.apply(Exp, &[d]).unwrap();
        let sum = ReduceSum {
            axes: vec![last],
            accum_dtype: dtype,
        };
        let s = g.apply(sum, &[e]).unwrap();
        let s = g.apply(bcast, &[s]).unwrap();
        let y = g.apply(Div, &[e, s]).unwrap();
        g.set_outputs(&[y]).unwrap();
        g
    }

    /// Check the one-node graph `p` on inputs of `types`.
    fn check_node(p: Primitive, types: &[TensorType]) {
        check_node_within(p, types, 0.0);
    }

    fn check_node_within(p: Primitive, types: &[TensorType], slack: f64) {
        let mut g = Graph::new();
        let vars: Vec<_> = types.iter().map(|ty| g.input(ty.clone())).collect();
        let out = g.apply(p.clone(), &vars).unwrap();
        g.set_outputs(&[out]).unwrap();
        // A 16-bit float sum or dot: on values it sums exactly.
        let count: Option<usize> = match &p {
            ReduceSum { axes, .. } => Some(axes.iter().map(|&d| types[0].shape[d]).product()),
            DotGeneral {
                lhs_contracting, ..
            } => Some(lhs_contracting.iter().map(|&d| types[0].shape[d]).product()),
            _ => None,
        };
        let exact = count.filter(|_| matches!(types[0].dtype, DType::F16 | DType::BF16));
        let inputs: Vec<Tensor> = types
            .iter()
            .enumerate()
            .map(|(i, ty)| match exact {
                Some(count) => exact_values(
                    ty.dtype,
                    &ty.shape,
                    i as u64 + 1,
                    count.div_ceil(128).max(1),
                ),
                None => values(ty.dtype, &ty.shape, i as u64 + 1),
            })
            .collect();
        check_within(&g, &inputs, slack);
    }

    fn ty(dtype: DType, shape: &[usize]) -> TensorType {
        TensorType::new(dtype, shape)
    }

    /// Check constant `p` (`full` or `iota`), with an input of its type so
    /// the plan runs on MPS: outputs the constant and `max(constant, x)`.
    fn check_constant(p: Primitive) {
        let out = p.infer(&[]).unwrap();
        let mut g = Graph::new();
        let x = g.input(out.clone());
        let c = g.apply(p, &[]).unwrap();
        let m = g.apply(Max, &[c, x]).unwrap();
        g.set_outputs(&[c, m]).unwrap();
        check(&g, &[values(out.dtype, &out.shape, 7)]);
    }

    #[test]
    fn elementwise() {
        if !available() {
            return;
        }
        for dtype in DTYPES {
            let x = ty(dtype, &[5, 7]);
            let two = [x.clone(), x.clone()];
            for p in [Max, Eq, Lt] {
                check_node(p, &two);
            }
            check_node(Select, &[ty(DType::Bool, &[5, 7]), x.clone(), x.clone()]);
            if dtype == DType::Bool {
                continue;
            }
            for p in [Add, Sub, Mul, Div] {
                check_node(p, &two);
            }
            check_node(Neg, std::slice::from_ref(&x));
            // No bfloat16 kernels for these (graph rejects them).
            if dtype.is_float() && dtype != DType::BF16 {
                for p in [Exp, Log, Sqrt, Tanh, Logistic] {
                    check_node(p, std::slice::from_ref(&x));
                }
            }
        }
    }

    #[test]
    fn conversions() {
        if !available() {
            return;
        }
        for from in DTYPES {
            for to in DTYPES {
                check_node(ConvertElementType { new_dtype: to }, &[ty(from, &[64])]);
            }
        }
        // Saturation and NaN.
        let f = Tensor::from_slice(&[300.0f32, -1e10, f32::NAN, 2.9], DType::F32);
        for to in [DType::I8, DType::U8, DType::I32, DType::U64, DType::Bool] {
            let mut g = Graph::new();
            let x = g.input(ty(DType::F32, &[4]));
            let y = g.apply(ConvertElementType { new_dtype: to }, &[x]).unwrap();
            g.set_outputs(&[y]).unwrap();
            check(&g, std::slice::from_ref(&f));
        }
    }

    /// 16-bit float dots accumulating in float32, written in their dtype or
    /// float32 (`matmul_<dtype>_f32_<output>`), large and small tiles.
    #[test]
    fn widened_contractions() {
        if !available() {
            return;
        }
        for dtype in [DType::F16, DType::BF16] {
            // Written in the operands' dtype, and in float32.
            for ((m, k, n), output_dtype) in [(37, 45, 29), (300, 70, 200)]
                .into_iter()
                .flat_map(|s| [(s, dtype), (s, DType::F32)])
            {
                let dot = DotGeneral {
                    lhs_contracting: vec![1],
                    rhs_contracting: vec![0],
                    lhs_batch: vec![],
                    rhs_batch: vec![],
                    accum_dtype: DType::F32,
                    output_dtype,
                };
                check_node(dot, &[ty(dtype, &[m, k]), ty(dtype, &[k, n])]);
            }
        }
    }

    #[test]
    fn large_reductions() {
        if !available() {
            return;
        }
        // Rows and columns, split into chunks and not, a middle axis, and
        // axes that are not consecutive (a threadgroup per output).
        let cases: [(&[usize], Vec<usize>); 7] = [
            (&[3, 50_000], vec![1]),
            (&[50_000, 3], vec![0]),
            (&[300, 700], vec![0, 1]),
            (&[4, 5_000, 3], vec![1]),
            (&[2_000, 300], vec![1]),
            (&[300, 2_000], vec![0]),
            (&[40, 7, 300], vec![0, 2]),
        ];
        for dtype in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I32,
            DType::U8,
            DType::I64,
            DType::Bool,
        ] {
            for (shape, axes) in &cases {
                let x = ty(dtype, shape);
                check_node(ReduceMax { axes: axes.clone() }, std::slice::from_ref(&x));
                if dtype != DType::Bool {
                    // float32 accumulation error: about 1e-7 of the sum of
                    // |x|, at most 4 an element (16-bit floats sum exactly
                    // representable values, check_node_within).
                    let count: usize = axes.iter().map(|&d| shape[d]).product();
                    let slack = 1e-6 * 4.0 * count as f64;
                    let sum = ReduceSum {
                        axes: axes.clone(),
                        accum_dtype: dtype,
                    };
                    check_node_within(sum, std::slice::from_ref(&x), slack);
                    // 16-bit floats accumulated in float32: on any values.
                    if matches!(dtype, DType::F16 | DType::BF16) {
                        let wide = ReduceSum {
                            axes: axes.clone(),
                            accum_dtype: DType::F32,
                        };
                        check_node_within(wide, std::slice::from_ref(&x), slack);
                    }
                }
            }
        }
    }

    #[test]
    fn reductions_and_contractions() {
        if !available() {
            return;
        }
        for dtype in DTYPES {
            let x = ty(dtype, &[2, 3, 4]);
            for axes in [vec![1], vec![0, 2], vec![0, 1, 2], vec![]] {
                check_node(ReduceMax { axes: axes.clone() }, std::slice::from_ref(&x));
                if dtype != DType::Bool {
                    check_node(
                        ReduceSum {
                            axes,
                            accum_dtype: dtype,
                        },
                        std::slice::from_ref(&x),
                    );
                }
            }
            if dtype == DType::Bool {
                continue;
            }
            // [b, m, k] x [b, k, n] and a contraction over two dimensions.
            let batched = DotGeneral {
                lhs_contracting: vec![2],
                rhs_contracting: vec![1],
                lhs_batch: vec![0],
                rhs_batch: vec![0],
                accum_dtype: dtype,
                output_dtype: dtype,
            };
            check_node(batched, &[ty(dtype, &[3, 2, 5]), ty(dtype, &[3, 5, 4])]);
            let two = DotGeneral {
                lhs_contracting: vec![0, 2],
                rhs_contracting: vec![2, 0],
                lhs_batch: vec![],
                rhs_batch: vec![],
                accum_dtype: dtype,
                output_dtype: dtype,
            };
            check_node(two, &[ty(dtype, &[3, 2, 4]), ty(dtype, &[4, 5, 3])]);
            // The tiled kernel: sizes that are not multiples of its tiles,
            // transposed operands, and two batch dimensions.
            let dot = |lc: usize, rc: usize, batch: Vec<usize>| DotGeneral {
                lhs_contracting: vec![lc],
                rhs_contracting: vec![rc],
                lhs_batch: batch.clone(),
                rhs_batch: batch,
                accum_dtype: dtype,
                output_dtype: dtype,
            };
            check_node(
                dot(1, 0, vec![]),
                &[ty(dtype, &[37, 45]), ty(dtype, &[45, 29])],
            );
            check_node(
                dot(0, 1, vec![]),
                &[ty(dtype, &[45, 37]), ty(dtype, &[29, 45])],
            );
            check_node(
                dot(2, 2, vec![0]),
                &[ty(dtype, &[2, 33, 17]), ty(dtype, &[2, 40, 17])],
            );
            check_node(
                dot(3, 2, vec![0, 1]),
                &[ty(dtype, &[2, 3, 5, 7]), ty(dtype, &[2, 3, 7, 4])],
            );
            // Enough output tiles for the floats' large-tile kernel, with
            // edges in both dimensions.
            if dtype.is_float() {
                check_node(
                    dot(1, 0, vec![]),
                    &[ty(dtype, &[500, 20]), ty(dtype, &[20, 700])],
                );
            }
        }
    }

    #[test]
    fn layout_and_constants() {
        if !available() {
            return;
        }
        for dtype in DTYPES {
            let bcast = BroadcastInDim {
                shape: vec![2, 3, 4],
                broadcast_dimensions: vec![0, 2],
            };
            check_node(bcast, &[ty(dtype, &[2, 1])]);
            let t = Transpose {
                permutation: vec![2, 0, 1],
            };
            check_node(t, &[ty(dtype, &[2, 3, 4])]);
            // Tiled (block swaps, sizes that are not multiples of the tile)
            // and gathered (other permutations) transposes.
            for (shape, permutation) in [
                (vec![3, 37, 45], vec![0, 2, 1]),
                (vec![70, 33], vec![1, 0]),
                (vec![4, 5, 6, 7], vec![0, 1, 3, 2]),
                (vec![4, 5, 6], vec![1, 0, 2]),
            ] {
                check_node(Transpose { permutation }, &[ty(dtype, &shape)]);
            }
            for (start, limit) in [
                ([1, 0, 2], [3, 5, 6]),
                ([0, 2, 0], [4, 3, 6]),
                ([2, 1, 3], [2, 4, 5]),
            ] {
                let slice = Slice {
                    start_indices: start.to_vec(),
                    limit_indices: limit.to_vec(),
                };
                check_node(slice, &[ty(dtype, &[4, 5, 6])]);
            }
            // A softmax written out over the last dimension (one row
            // kernel): rows shorter and longer than a threadgroup, one
            // element, none.
            if matches!(dtype, DType::F16 | DType::F32) {
                for shape in [vec![4, 7], vec![3, 1000], vec![2, 3, 1], vec![0, 5]] {
                    let g = written_softmax(dtype, &shape);
                    check(&g, &[values(dtype, &shape, 1)]);
                }
            }
            // Each dimension, operands of different sizes (one empty).
            for dimension in 0..3 {
                let shape = |n| {
                    let mut shape = vec![3, 4, 5];
                    shape[dimension] = n;
                    ty(dtype, &shape)
                };
                check_node(
                    Concatenate { dimension },
                    &[shape(2), shape(0), shape(3), shape(1)],
                );
            }
            if dtype != DType::Bool {
                for dimension in 0..3 {
                    check_constant(Iota {
                        dtype,
                        shape: vec![5, 6, 7],
                        dimension,
                    });
                }
            }
            let full = Full {
                shape: vec![3, 2],
                fill_value: Scalar::Float(-2.5),
                dtype,
            };
            check_constant(full);
            if dtype != DType::Bool {
                let iota = Iota {
                    dtype,
                    shape: vec![3, 4],
                    dimension: 0,
                };
                check_constant(iota);
            }
            // Outputs that alias an input are copied (a reshape step).
            let mut g = Graph::new();
            let x = g.input(ty(dtype, &[2, 3]));
            let flat = g.apply(Reshape { new_sizes: vec![6] }, &[x]).unwrap();
            g.set_outputs(&[flat, x]).unwrap();
            check(&g, &[values(dtype, &[2, 3], 9)]);
        }
    }

    #[test]
    fn mlp_plan() {
        if !available() {
            return;
        }
        let inputs = [data(&[4, 8], 1), data(&[8, 16], 2), data(&[16, 3], 3)];
        check(&mlp(), &inputs);
    }

    #[test]
    fn rejects_float64_and_mixed_devices() {
        if !available() {
            return;
        }
        let mut g = Graph::new();
        let x = g.input(ty(DType::F64, &[2]));
        let y = g.apply(Neg, &[x]).unwrap();
        g.set_outputs(&[y]).unwrap();
        let x = Tensor::zeros(
            &[2],
            TensorOptions::new().dtype(DType::F64).device(Device::Mps),
        );
        let err = Plan::compile(&g).run(&[x]).unwrap_err();
        assert!(err.contains("float64"), "{err}");

        let mut g = Graph::new();
        let a = g.input(ty(DType::F32, &[2]));
        let b = g.input(ty(DType::F32, &[2]));
        let c = g.apply(Add, &[a, b]).unwrap();
        g.set_outputs(&[c]).unwrap();
        let (a, b) = (
            Tensor::zeros(&[2], Device::Mps),
            Tensor::zeros(&[2], DType::F32),
        );
        assert!(Plan::compile(&g).run(&[a, b]).is_err());
    }
}
