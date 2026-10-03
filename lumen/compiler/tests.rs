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
