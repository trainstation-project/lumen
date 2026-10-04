use super::cse::cse;
use crate::graph::Primitive::*;
use crate::graph::tests::data;
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::ops::reference;
use crate::{DType, Scalar};

fn ty(shape: &[usize]) -> TensorType {
    TensorType::new(DType::F32, shape)
}

fn apply(g: &mut Graph, p: Primitive, inputs: &[Var]) -> Var {
    g.apply(p, inputs).unwrap()
}

fn names(g: &Graph) -> Vec<&'static str> {
    g.nodes().iter().map(|n| n.primitive.name()).collect()
}

fn matmul() -> Primitive {
    DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    }
}

#[test]
fn repeated_computations_run_once() {
    // relu(x @ w) * (x @ w), with the matmul written twice.
    let mut g = Graph::new();
    let x = g.input(ty(&[4, 8]));
    let w = g.input(ty(&[8, 16]));
    let a = apply(&mut g, matmul(), &[x, w]);
    let zero = |g: &mut Graph| {
        let z = apply(
            g,
            Full {
                shape: vec![],
                fill_value: Scalar::Float(0.0),
                dtype: DType::F32,
            },
            &[],
        );
        let b = BroadcastInDim {
            shape: vec![4, 16],
            broadcast_dimensions: vec![],
        };
        apply(g, b, &[z])
    };
    let z = zero(&mut g);
    let relu = apply(&mut g, Max, &[a, z]);
    let b = apply(&mut g, matmul(), &[x, w]);
    // Operands swapped: the same product.
    let y = apply(&mut g, Mul, &[relu, b]);
    let y2 = apply(&mut g, Mul, &[b, relu]);
    let z2 = zero(&mut g);
    let s = apply(&mut g, Add, &[y, y2]);
    let s = apply(&mut g, Add, &[s, z2]);
    g.set_outputs(&[s]).unwrap();
    let out = cse(&g);
    assert_eq!(
        names(&out),
        [
            "dot_general",
            "full",
            "broadcast_in_dim",
            "max",
            "mul",
            "add",
            "add"
        ],
        "{out}"
    );
    let inputs = [data(&[4, 8], 1), data(&[8, 16], 2)];
    let (e, a) = (
        reference::run(&g, &inputs).unwrap(),
        reference::run(&out, &inputs).unwrap(),
    );
    assert_eq!(e[0].to_vec::<f32>(), a[0].to_vec::<f32>());
}

#[test]
fn different_parameters_or_order_are_kept() {
    let mut g = Graph::new();
    let x = g.input(ty(&[4, 4]));
    let t = |p: Vec<usize>| Transpose { permutation: p };
    let a = apply(&mut g, t(vec![1, 0]), &[x]);
    let b = apply(&mut g, t(vec![0, 1]), &[x]);
    // Not commutative: kept.
    let c = apply(&mut g, Sub, &[a, b]);
    let d = apply(&mut g, Sub, &[b, a]);
    g.set_outputs(&[c, d]).unwrap();
    assert_eq!(names(&cse(&g)), names(&g));
}

/// The reference's outputs of `g` and of `g` simplified: equal, exactly.
fn same_values(g: &Graph, simplified: &Graph, inputs: &[crate::Tensor]) {
    let want = reference::run(g, inputs).unwrap();
    let got = reference::run(simplified, inputs).unwrap();
    for (w, g) in want.iter().zip(&got) {
        assert_eq!(w.to_vec::<f32>(), g.to_vec::<f32>());
    }
}

fn full(shape: &[usize], value: f64) -> Primitive {
    Full {
        shape: shape.to_vec(),
        fill_value: Scalar::Float(value),
        dtype: DType::F32,
    }
}

/// Algebraic simplification: `x * 1`, `1 * x`, `x / 1`, `x + 0`, `0 + x`,
/// `x - 0`, `neg(neg(x))`, `select(p, x, x)` and a cast to `x`'s dtype are
/// `x`; a reshape of a reshape, a transpose of a transpose, a broadcast of a
/// broadcast one.
#[test]
fn algebraic_simplification() {
    let mut g = Graph::new();
    let x = g.input(ty(&[2, 3]));
    let one = apply(&mut g, full(&[2, 3], 1.0), &[]);
    let zero = apply(&mut g, full(&[2, 3], 0.0), &[]);
    let a = apply(&mut g, Mul, &[x, one]);
    let a = apply(&mut g, Mul, &[one, a]);
    let a = apply(&mut g, Div, &[a, one]);
    let a = apply(&mut g, Add, &[a, zero]);
    let a = apply(&mut g, Add, &[zero, a]);
    let a = apply(&mut g, Sub, &[a, zero]);
    let a = apply(&mut g, Neg, &[a]);
    let a = apply(&mut g, Neg, &[a]);
    let p = apply(&mut g, Lt, &[a, x]);
    let a = apply(&mut g, Select, &[p, a, a]);
    let a = apply(
        &mut g,
        Cast {
            new_dtype: DType::F32,
        },
        &[a],
    );
    let r = apply(
        &mut g,
        Reshape {
            new_sizes: vec![3, 2],
        },
        &[a],
    );
    let r = apply(&mut g, Reshape { new_sizes: vec![6] }, &[r]);
    let t = apply(
        &mut g,
        Transpose {
            permutation: vec![1, 0],
        },
        &[a],
    );
    let t = apply(
        &mut g,
        Transpose {
            permutation: vec![1, 0],
        },
        &[t],
    );
    let inner = BroadcastInDim {
        shape: vec![4, 2, 3],
        broadcast_dimensions: vec![1, 2],
    };
    let b = apply(&mut g, inner, &[x]);
    let outer = BroadcastInDim {
        shape: vec![5, 4, 2, 3],
        broadcast_dimensions: vec![1, 2, 3],
    };
    let b = apply(&mut g, outer, &[b]);
    g.set_outputs(&[r, t, b]).unwrap();
    let simplified = super::simplify::simplify(&g);
    assert_eq!(
        names(&simplified),
        ["reshape", "broadcast_in_dim"],
        "{simplified}"
    );
    same_values(&g, &simplified, &[data(&[2, 3], 1)]);
}

/// A transpose of a dot swapping its free dimensions (a weight's gradient
/// from autodiff) is the dot of its operands swapped, no copy, with
/// `matched` (the MPS compiler's, after matching attention); not without
/// it, nor when the dot is read twice.
#[test]
fn transposes_of_dots_swap_their_operands() {
    let build = |read_twice: bool| {
        let mut g = Graph::new();
        let a = g.input(ty(&[4, 6]));
        let b = g.input(ty(&[6, 5]));
        let d = apply(&mut g, matmul(), &[a, b]);
        let t = apply(
            &mut g,
            Transpose {
                permutation: vec![1, 0],
            },
            &[d],
        );
        let outputs = if read_twice { vec![t, d] } else { vec![t] };
        g.set_outputs(&outputs).unwrap();
        g
    };
    let inputs = [data(&[4, 6], 1), data(&[6, 5], 2)];
    let g = build(false);
    let swapped = super::simplify::simplify_with(&g, true);
    assert_eq!(names(&swapped), ["dot_general"], "{swapped}");
    same_values(&g, &swapped, &inputs);
    assert_eq!(
        names(&super::simplify::simplify(&g)),
        ["dot_general", "transpose"]
    );
    let twice = super::simplify::simplify_with(&build(true), true);
    assert_eq!(names(&twice), ["dot_general", "transpose"], "{twice}");
}

/// With `matched`, a transpose moves past the elementwise primitives
/// reading it (and a constant, made in the shape before it): one transpose,
/// last (XLA's `ReshapeMover`); not without it.
#[test]
fn rearranges_move_past_elementwise_primitives() {
    let mut g = Graph::new();
    let x = g.input(ty(&[2, 3]));
    let y = g.input(ty(&[2, 3]));
    let two = apply(&mut g, full(&[3, 2], 2.0), &[]);
    let t = Transpose {
        permutation: vec![1, 0],
    };
    let tx = apply(&mut g, t.clone(), &[x]);
    let ty_ = apply(&mut g, t, &[y]);
    let a = apply(&mut g, Add, &[tx, ty_]);
    let a = apply(&mut g, Mul, &[a, two]);
    let a = apply(&mut g, Exp, &[a]);
    g.set_outputs(&[a]).unwrap();
    let moved = super::simplify::simplify_with(&g, true);
    assert_eq!(
        names(&moved),
        ["add", "full", "mul", "exp", "transpose"],
        "{moved}"
    );
    same_values(&g, &moved, &[data(&[2, 3], 1), data(&[2, 3], 2)]);
    assert_eq!(
        names(&super::simplify::simplify(&g)),
        ["full", "transpose", "transpose", "add", "mul", "exp"]
    );
}

/// With `matched`, a widening cast moves after the layout primitives
/// reading it (a concatenate's constant cast too, if exact), and a
/// narrowing one before them (XLA's `ConvertMover`): they move the bfloat16
/// values.
#[test]
fn casts_move_past_layout_primitives() {
    let cast = |dtype| Cast { new_dtype: dtype };
    let widened = |g: &mut Graph| {
        let x = g.input(ty(&[2, 3]));
        let b = apply(g, cast(DType::BF16), &[x]);
        apply(g, cast(DType::F32), &[b])
    };
    let inputs = [data(&[2, 3], 1)];
    // Widening, then a reshape: the reshape first.
    let mut g = Graph::new();
    let w = widened(&mut g);
    let r = apply(&mut g, Reshape { new_sizes: vec![6] }, &[w]);
    g.set_outputs(&[r]).unwrap();
    let moved = super::simplify::simplify_with(&g, true);
    assert_eq!(names(&moved), ["cast", "reshape", "cast"], "{moved}");
    same_values(&g, &moved, &inputs);
    // A transpose and a slice, then narrowing: the cast first.
    let mut g = Graph::new();
    let y = g.input(ty(&[2, 3]));
    let t = Transpose {
        permutation: vec![1, 0],
    };
    let t = apply(&mut g, t, &[y]);
    let slice = Slice {
        start_indices: vec![0, 0],
        limit_indices: vec![2, 2],
    };
    let s = apply(&mut g, slice, &[t]);
    let n = apply(&mut g, cast(DType::BF16), &[s]);
    let o = apply(&mut g, cast(DType::F32), &[n]);
    g.set_outputs(&[o]).unwrap();
    let moved = super::simplify::simplify_with(&g, true);
    assert_eq!(
        names(&moved),
        ["cast", "transpose", "slice", "cast"],
        "{moved}"
    );
    same_values(&g, &moved, &inputs);
    // A concatenate with a constant: moved if it is exact in bfloat16.
    for (value, want) in [
        (1.0, vec!["cast", "full", "concatenate", "cast"]),
        (0.1, vec!["cast", "cast", "full", "concatenate"]),
    ] {
        let mut g = Graph::new();
        let w = widened(&mut g);
        let c = apply(&mut g, full(&[2, 3], value), &[]);
        let cat = apply(&mut g, Concatenate { dimension: 0 }, &[w, c]);
        g.set_outputs(&[cat]).unwrap();
        let moved = super::simplify::simplify_with(&g, true);
        assert_eq!(names(&moved), want, "{moved}");
        same_values(&g, &moved, &inputs);
    }
}
