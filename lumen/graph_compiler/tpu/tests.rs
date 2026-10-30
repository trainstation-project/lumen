//! Tests for the TPU graph compiler passes.
use crate::graph::Primitive::*;
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::kernels::reference;
use crate::{DType, Device, Scalar, Tensor};

/// Applies `p` to `inputs`, panicking on a type error.
fn apply(g: &mut Graph, p: Primitive, inputs: &[Var]) -> Var {
    g.apply(p, inputs).unwrap()
}

/// Sums a `[3, 4]` value over its rows into `[3]`, through a reshape round trip.
fn sum(g: &mut Graph, v: Var) -> Var {
    let axes = vec![1];

    let s = apply(
        g,
        ReduceSum {
            axes,
            accum_dtype: DType::F32,
        },
        &[v],
    );

    let r = apply(
        g,
        Reshape {
            new_sizes: vec![3, 1],
        },
        &[s],
    );

    apply(g, Reshape { new_sizes: vec![3] }, &[r])
}

/// Broadcasts a `[3]` value back over the rows of a `[3, 4]` value.
fn rows(g: &mut Graph, v: Var) -> Var {
    let broadcast = BroadcastInDim {
        shape: vec![3, 4],
        broadcast_dimensions: vec![0],
    };

    apply(g, broadcast, &[v])
}

/// Softmax-with-ties backward diamonds fuse into row kernels that still compute the same values.
#[test]
fn rows_fuse_where_what_others_read_of_them_is_written() {
    let mut g = Graph::new();
    let x = g.input(TensorType::new(DType::F32, &[3, 4]));
    let y = g.input(TensorType::new(DType::F32, &[3, 4]));
    let max = apply(&mut g, ReduceMax { axes: vec![1] }, &[x]);
    let peak = rows(&mut g, max);
    let shifted = apply(&mut g, Sub, &[x, peak]);
    let e = apply(&mut g, Exp, &[shifted]);
    let total = sum(&mut g, e);
    let totals = rows(&mut g, total);
    let p = apply(&mut g, Div, &[e, totals]);
    let at_max = apply(&mut g, Eq, &[x, peak]);

    let ties = apply(
        &mut g,
        Cast {
            new_dtype: DType::F32,
        },
        &[at_max],
    );

    let count = sum(&mut g, ties);
    let q = apply(&mut g, Div, &[p, totals]);
    let dq = apply(&mut g, Neg, &[q]);

    let one = Full {
        shape: vec![],
        fill_value: Scalar::Float(1.0),
        dtype: DType::F32,
    };

    let one = apply(&mut g, one, &[]);

    let ones = apply(
        &mut g,
        BroadcastInDim {
            shape: vec![3, 4],
            broadcast_dimensions: vec![],
        },
        &[one],
    );

    let dy = apply(&mut g, Mul, &[p, ones]);
    let dp = apply(&mut g, Mul, &[ones, y]);
    let a = apply(&mut g, Mul, &[dp, dq]);
    let b = apply(&mut g, Div, &[dp, totals]);
    let da = sum(&mut g, a);
    let da = rows(&mut g, da);
    let de = apply(&mut g, Add, &[b, da]);
    let ds = apply(&mut g, Mul, &[de, e]);
    let dm = apply(&mut g, Neg, &[ds]);
    let dm = sum(&mut g, dm);
    let dm = apply(&mut g, Div, &[dm, count]);
    let dm = rows(&mut g, dm);
    let dm = apply(&mut g, Mul, &[dm, ties]);
    let dx = apply(&mut g, Add, &[ds, dm]);
    g.set_outputs(&[dx, dy]).unwrap();

    // `e`, `totals` and `ties` are read by several diamonds, so a fusion must expose them as outputs.
    let (fusible, fits) = (
        crate::codegen::tpu::row::fusible,
        crate::codegen::tpu::row::fits,
    );

    let rows = crate::graph_compiler::common::diamonds::diamonds(&g, fusible);
    let fused = crate::graph_compiler::common::diamonds::fused(&g, &rows, fusible, fits);

    let inputs: Vec<Tensor> = (0..2)
        .map(|k| {
            let values: Vec<f32> = (0..12)
                .map(|i| ((i * 7 + k * 5) % 11) as f32 / 4.0)
                .collect();

            Tensor::from_slice(&values, Device::Cpu(0)).reshape(&[3, 4])
        })
        .collect();

    let expected = reference::run(&g, &inputs).unwrap();
    let actual = reference::run(&fused, &inputs).unwrap();

    for (e, a) in expected.iter().zip(&actual) {
        assert_eq!(e.to_vec::<f32>(), a.to_vec::<f32>(), "{fused}");
    }

    assert!(
        fused
            .nodes()
            .iter()
            .any(|n| matches!(n.primitive, Fusion { .. })),
        "{fused}"
    );
}
