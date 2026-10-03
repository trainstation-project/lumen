use super::merge_dots::merge_dots;
use super::{codegen, fusion};
use crate::graph::Primitive::*;
use crate::graph::tests::mps::{available, check, check_within, values};
use crate::graph::tests::{data, mlp};
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::ops::reference;
use crate::{DType, Device, Scalar, Tensor};

fn ty(dtype: DType, shape: &[usize]) -> TensorType {
    TensorType::new(dtype, shape)
}

fn fuse(g: &Graph) -> Graph {
    fusion::fuse(g, |body| codegen::kernel(body).0)
}

/// The fused graph's primitives, after checking it computes exactly what
/// `g` does (on the reference executor) on `inputs`.
fn fused_primitives(g: &Graph, inputs: &[Tensor]) -> Vec<&'static str> {
    let fused = fuse(g);
    let expected = reference::run(g, inputs).unwrap();
    let actual = reference::run(&fused, inputs).unwrap();
    for (e, a) in expected.iter().zip(&actual) {
        assert_eq!(e.shape(), a.shape());
        assert_eq!(e.to_vec::<f32>(), a.to_vec::<f32>(), "{fused}");
    }
    names(&fused)
}

/// The graph's primitives' names, `fusion` for each fusion.
fn names(g: &Graph) -> Vec<&'static str> {
    let name = |p: &Primitive| match p {
        Fusion { .. } => "fusion",
        p => p.name(),
    };
    g.nodes().iter().map(|n| name(&n.primitive)).collect()
}

fn apply(g: &mut Graph, p: Primitive, inputs: &[Var]) -> Var {
    g.apply(p, inputs).unwrap()
}

#[test]
fn fuses_elementwise_and_layout_chains() {
    // tanh(x * 0.5) + transpose(x), as one kernel.
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[4, 4]));
    let half = Full {
        shape: vec![],
        fill_value: Scalar::Float(0.5),
        dtype: DType::F32,
    };
    let half = apply(&mut g, half, &[]);
    let bcast = BroadcastInDim {
        shape: vec![4, 4],
        broadcast_dimensions: vec![],
    };
    let half = apply(&mut g, bcast, &[half]);
    let h = apply(&mut g, Mul, &[x, half]);
    let t = apply(&mut g, Tanh, &[h]);
    let xt = apply(
        &mut g,
        Transpose {
            permutation: vec![1, 0],
        },
        &[x],
    );
    let y = apply(&mut g, Add, &[t, xt]);
    g.set_outputs(&[y]).unwrap();
    assert_eq!(fused_primitives(&g, &[data(&[4, 4], 1)]), ["fusion"]);
    // Profiled by its primitives, in graph order.
    assert_eq!(
        fuse(&g).nodes()[0].primitive.name(),
        "full -> broadcast_in_dim -> mul -> tanh -> transpose -> add"
    );
}

#[test]
fn reductions_and_contractions_are_boundaries() {
    let inputs = [data(&[4, 8], 1), data(&[8, 16], 2), data(&[16, 3], 3)];
    // relu's max and its zero; softmax's broadcast max, sub and exp; then
    // its broadcast sum and divide.
    assert_eq!(
        fused_primitives(&mlp(), &inputs),
        [
            "dot_general",
            "fusion",
            "dot_general",
            "reduce_max",
            "fusion",
            "reduce_sum",
            "fusion"
        ]
    );
}

#[test]
fn only_cheap_values_are_recomputed() {
    // exp has two users: computed once, not in each.
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[8]));
    let e = apply(&mut g, Exp, &[x]);
    let a = apply(&mut g, Add, &[e, x]);
    let b = apply(&mut g, Mul, &[e, x]);
    g.set_outputs(&[a, b]).unwrap();
    assert_eq!(
        fused_primitives(&g, &[data(&[8], 1)]),
        ["exp", "add", "mul"]
    );

    // add has two users: copied into each.
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[8]));
    let c = apply(&mut g, Add, &[x, x]);
    let e = apply(&mut g, Exp, &[c]);
    let t = apply(&mut g, Tanh, &[c]);
    g.set_outputs(&[e, t]).unwrap();
    assert_eq!(fused_primitives(&g, &[data(&[8], 1)]), ["fusion", "fusion"]);
}

#[test]
fn fusions_fit_in_a_kernel() {
    // A sum of 40 inputs reads more buffers than a kernel binds.
    let mut g = Graph::new();
    let xs: Vec<Var> = (0..40).map(|_| g.input(ty(DType::F32, &[4]))).collect();
    let mut sum = xs[0];
    for &x in &xs[1..] {
        sum = apply(&mut g, Add, &[sum, x]);
    }
    g.set_outputs(&[sum]).unwrap();
    let inputs: Vec<Tensor> = (0..40).map(|i| data(&[4], i)).collect();
    fused_primitives(&g, &inputs);
    for node in fuse(&g).nodes() {
        assert!(node.inputs.len() <= 29, "{}", node.primitive);
    }
}

/// Broadcasts (with size-1 dimensions), transposes, iota, compares,
/// selects and converts, fused, on MPS.
#[test]
fn fused_kernels_match_reference() {
    if !available() {
        return;
    }
    let shape = [2, 3, 4, 5];
    for dtype in [
        DType::U8,
        DType::U32,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
    ] {
        let mut g = Graph::new();
        let a = g.input(ty(dtype, &shape));
        let b = g.input(ty(dtype, &[3, 5]));
        let c = g.input(ty(dtype, &[5, 4, 3, 2]));
        let d = g.input(ty(dtype, &[1, 4, 1]));
        let bcast = |dims: Vec<usize>| BroadcastInDim {
            shape: shape.to_vec(),
            broadcast_dimensions: dims,
        };
        let b = apply(&mut g, bcast(vec![1, 3]), &[b]);
        let c = apply(
            &mut g,
            Transpose {
                permutation: vec![3, 2, 1, 0],
            },
            &[c],
        );
        let d = apply(&mut g, bcast(vec![1, 2, 3]), &[d]);
        let s = apply(&mut g, Mul, &[a, b]);
        let s = apply(&mut g, Add, &[s, c]);
        let s = apply(&mut g, Sub, &[s, d]);
        let iota = Iota {
            dtype,
            shape: shape.to_vec(),
            dimension: 2,
        };
        let iota = apply(&mut g, iota, &[]);
        let m = apply(&mut g, Max, &[s, iota]);
        let p = apply(&mut g, Lt, &[s, iota]);
        let n = apply(&mut g, Neg, &[s]);
        let mut y = apply(&mut g, Select, &[p, m, n]);
        if dtype.is_float() {
            y = apply(&mut g, Logistic, &[y]);
        }
        let r = apply(
            &mut g,
            Reshape {
                new_sizes: vec![6, 20],
            },
            &[y],
        );
        let t = apply(
            &mut g,
            Transpose {
                permutation: vec![1, 0],
            },
            &[r],
        );
        let i = apply(
            &mut g,
            ConvertElementType {
                new_dtype: DType::I32,
            },
            &[y],
        );
        g.set_outputs(&[t, i]).unwrap();
        // logistic, expensive and read twice, is stored: its convert runs alone.
        let expected: &[&str] = if dtype.is_float() {
            &["fusion", "fusion", "convert_element_type"]
        } else {
            &["fusion", "fusion"]
        };
        let fused = fuse(&g);
        assert_eq!(names(&fused), expected, "{fused}");
        let inputs: Vec<Tensor> = g
            .inputs()
            .iter()
            .enumerate()
            .map(|(k, &v)| values(dtype, &g.type_of(v).shape, k as u64 + 1))
            .collect();
        check(&g, &inputs);
    }
}

#[test]
fn dot_operands_are_put_in_matmul_form_by_a_transpose_step() {
    // Contracting dims (0, 2) of lhs do not collapse: a transpose first.
    let mut g = Graph::new();
    let a = g.input(ty(DType::F32, &[3, 2, 4]));
    let b = g.input(ty(DType::F32, &[4, 5, 3]));
    let dot = DotGeneral {
        lhs_contracting: vec![0, 2],
        rhs_contracting: vec![2, 0],
        lhs_batch: vec![],
        rhs_batch: vec![],
    };
    let y = apply(&mut g, dot, &[a, b]);
    g.set_outputs(&[y]).unwrap();
    let canonical = super::canonicalize_dots(&g);
    assert_eq!(
        names(&canonical),
        ["transpose", "transpose", "dot_general"],
        "{canonical}"
    );
    let inputs = [data(&[3, 2, 4], 1), data(&[4, 5, 3], 2)];
    let expected = reference::run(&g, &inputs).unwrap();
    let actual = reference::run(&canonical, &inputs).unwrap();
    assert_eq!(expected[0].to_vec::<f32>(), actual[0].to_vec::<f32>());
    // A matmul already in form is left alone.
    let mut g = Graph::new();
    let a = g.input(ty(DType::F32, &[4, 8]));
    let b = g.input(ty(DType::F32, &[8, 3]));
    let mm = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
    };
    let y = apply(&mut g, mm, &[a, b]);
    g.set_outputs(&[y]).unwrap();
    assert_eq!(names(&super::canonicalize_dots(&g)), ["dot_general"]);
}

/// Split reductions and dots out of matmul form need what only the MPS
/// compiler arranges: their scratch and transposes are planned (no kernel
/// allocates), and a generic plan run on MPS says so.
#[test]
fn mps_plans_hold_kernel_scratch() {
    if !available() {
        return;
    }
    let mut g = Graph::new();
    let x = g.input(ty(DType::F16, &[4, 50_000]));
    let s = apply(&mut g, ReduceSum { axes: vec![1] }, &[x]);
    g.set_outputs(&[s]).unwrap();
    let plan = crate::compiler::compile(&g, crate::Device::Mps).unwrap();
    let (_, bytes) = plan.steps()[0].scratch.expect("partials in the workspace");
    assert!(bytes > 0 && plan.workspace_bytes() >= bytes, "{plan}");
    let input = values(DType::F16, &[4, 50_000], 1).to(crate::Device::Mps);
    let generic = crate::graph::Plan::compile(&g).run(std::slice::from_ref(&input));
    assert!(generic.unwrap_err().contains("compile the graph for MPS"));
}

/// Slices fuse into the kernel that reads them: a packed weight's matmul,
/// then its halves gated, read in place.
#[test]
fn slices_fuse_into_their_readers() {
    let mut g = Graph::new();
    let h = g.input(ty(DType::F32, &[16, 128]));
    let half = |g: &mut Graph, start: usize| {
        let s = Slice {
            start_indices: vec![0, start],
            limit_indices: vec![16, start + 64],
        };
        apply(g, s, &[h])
    };
    let (a, b) = (half(&mut g, 0), half(&mut g, 64));
    let e = apply(&mut g, Exp, &[a]);
    let y = apply(&mut g, Mul, &[e, b]);
    g.set_outputs(&[y]).unwrap();
    assert_eq!(fused_primitives(&g, &[data(&[16, 128], 1)]), ["fusion"]);
    if available() {
        check(&g, &[values(DType::F32, &[16, 128], 1)]);
    }
}

/// `parts` (host tensors) side by side along `dimension`, by the reference
/// executor: the block [`Tensor::pack`] places them in.
fn block(parts: &[&Tensor], dimension: usize) -> Tensor {
    let mut g = Graph::new();
    let vars: Vec<Var> = parts
        .iter()
        .map(|t| g.input(ty(t.dtype(), t.shape())))
        .collect();
    let out = apply(&mut g, Concatenate { dimension }, &vars);
    g.set_outputs(&[out]).unwrap();
    let parts: Vec<Tensor> = parts.iter().map(|&t| t.clone()).collect();
    reference::run(&g, &parts).unwrap().remove(0)
}

/// Dots sharing `x` [16, 32] whose other operands are packable parameters
/// become one dot of `x` and a block of them side by side (a new input),
/// its result sliced back into each: no concatenation.
#[test]
fn dots_sharing_an_operand_merge() {
    let matmul = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
    };
    // The gated MLP: relu(x @ w1) * (x @ w3) @ w2.
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[16, 32]));
    let w1 = g.input(ty(DType::F32, &[32, 64]));
    let w3 = g.input(ty(DType::F32, &[32, 64]));
    let w2 = g.input(ty(DType::F32, &[64, 32]));
    let gate = apply(&mut g, matmul.clone(), &[x, w1]);
    let up = apply(&mut g, matmul.clone(), &[x, w3]);
    let zero = Full {
        shape: vec![16, 64],
        fill_value: Scalar::Float(0.0),
        dtype: DType::F32,
    };
    let zero = apply(&mut g, zero, &[]);
    let relu = apply(&mut g, Max, &[gate, zero]);
    let h = apply(&mut g, Mul, &[relu, up]);
    let y = apply(&mut g, matmul.clone(), &[h, w2]);
    g.set_outputs(&[y]).unwrap();
    let dots = |g: &Graph| names(g).iter().filter(|&&n| n == "dot_general").count();
    // Only parameters merge.
    assert_eq!(dots(&merge_dots(&g, &[false; 4]).0), 3);
    let (merged, packs) = merge_dots(&g, &[false, true, true, true]);
    assert_eq!(packs, [(vec![1, 2], 1)]);
    assert_eq!(dots(&merged), 2, "{merged}");
    // The slices fuse into the gate.
    assert_eq!(
        names(&fuse(&merged)),
        ["dot_general", "fusion", "dot_general"]
    );
    let mut inputs = vec![
        data(&[16, 32], 1),
        data(&[32, 64], 2),
        data(&[32, 64], 3),
        data(&[64, 32], 4),
    ];
    inputs.push(block(&[&inputs[1], &inputs[2]], 1));
    let expected = reference::run(&g, &inputs[..4]).unwrap();
    assert_eq!(
        reference::run(&merged, &inputs).unwrap()[0].to_vec::<f32>(),
        expected[0].to_vec::<f32>()
    );

    // Shared rhs (lhs side by side, rows); a dot reading another's result,
    // or of a parameter something else reads too, does not merge.
    let mut g = Graph::new();
    let w = g.input(ty(DType::F32, &[32, 32]));
    let xs: Vec<Var> = [4, 6, 4]
        .iter()
        .map(|&m| g.input(ty(DType::F32, &[m, 32])))
        .collect();
    let ys: Vec<Var> = xs
        .iter()
        .map(|&x| apply(&mut g, matmul.clone(), &[x, w]))
        .collect();
    let mut outs: Vec<Var> = ys.iter().map(|&y| apply(&mut g, Neg, &[y])).collect();
    let dependent = apply(&mut g, matmul.clone(), &[outs[0], w]);
    let twice = apply(&mut g, matmul.clone(), &[xs[1], w]);
    outs.extend([apply(&mut g, Neg, &[dependent]), twice]);
    g.set_outputs(&outs).unwrap();
    let (merged, packs) = merge_dots(&g, &[true; 4]);
    assert_eq!(packs, [(vec![1, 3], 0)]);
    assert_eq!(dots(&merged), 4, "{merged}");
    let mut inputs: Vec<Tensor> = [vec![32, 32], vec![4, 32], vec![6, 32], vec![4, 32]]
        .iter()
        .enumerate()
        .map(|(i, shape)| data(shape, i as u64 + 1))
        .collect();
    inputs.push(block(&[&inputs[1], &inputs[3]], 0));
    let expected = reference::run(&g, &inputs[..4]).unwrap();
    for (e, a) in expected
        .iter()
        .zip(reference::run(&merged, &inputs).unwrap())
    {
        assert_eq!(
            (e.shape(), e.to_vec::<f32>()),
            (a.shape(), a.to_vec::<f32>())
        );
    }
}

/// The gated MLP compiled to own its memory on MPS: its weights meta
/// parameters, `w1` and `w3` placed side by side in one block the merged
/// dot reads; the data copied into their places; `x` copied in each run.
#[test]
fn owned_plans_read_packed_parameters() {
    if !available() {
        return;
    }
    let matmul = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
    };
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[16, 32]));
    let w1 = g.input(ty(DType::F32, &[32, 64]));
    let w3 = g.input(ty(DType::F32, &[32, 64]));
    let gate = apply(&mut g, matmul.clone(), &[x, w1]);
    let up = apply(&mut g, matmul, &[x, w3]);
    let y = apply(&mut g, Mul, &[gate, up]);
    g.set_outputs(&[y]).unwrap();
    let params = vec![false, true, true];
    let options = crate::compiler::Options {
        parameters: Some(params.clone()),
        packable: params,
        ..Default::default()
    };
    let plan = crate::compiler::compile_with(&g, Device::Mps, &options).unwrap();
    assert_eq!(plan.packed, [(vec![1, 2], 1)]);
    let meta = || Tensor::zeros(&[32, 64], Device::Meta);
    let (m1, m3) = (meta(), meta());
    let block = Tensor::pack(&[m1.clone(), m3.clone()], 1, Device::Mps)
        .unwrap()
        .unwrap();
    // Placed again the same way: the same block; alone, not.
    let again = Tensor::pack(&[m1.clone(), m3.clone()], 1, Device::Mps).unwrap();
    assert!(again.is_some_and(|b| b.shares_storage_with(&block)));
    assert!(
        Tensor::pack(&[m3.clone(), m1.clone()], 1, Device::Mps)
            .unwrap()
            .is_none()
    );
    let inputs = [data(&[16, 32], 1), data(&[32, 64], 2), data(&[32, 64], 3)];
    let (p1, p3) = (
        m1.placed(Device::Mps).unwrap(),
        m3.placed(Device::Mps).unwrap(),
    );
    assert!(p1.shares_storage_with(&block) && !p1.is_contiguous());
    p1.copy_(&inputs[1]).unwrap();
    p3.copy_(&inputs[2]).unwrap();
    let workspace = Tensor::zeros(
        &[plan.workspace_bytes()],
        crate::TensorOptions::new()
            .dtype(DType::U8)
            .device(Device::Mps),
    );
    let expected = reference::run(&g, &inputs).unwrap()[0].to_vec::<f32>();
    for _ in 0..2 {
        let out = plan
            .run_in(
                &workspace,
                &[inputs[0].clone(), p1.clone(), p3.clone(), block.clone()],
            )
            .unwrap();
        assert!(out[0].shares_storage_with(&workspace));
        for (e, a) in expected.iter().zip(out[0].to(Device::Cpu).to_vec::<f32>()) {
            assert!((e - a).abs() <= 1e-4 * (1.0 + e.abs()), "{e} vs {a}");
        }
    }
}

/// Slices read only by dots are views of the sliced value, read in place
/// at its strides; one the matmul cannot read in its form stays a copy.
#[test]
fn dots_read_slices_in_place() {
    let dot = |lc: usize, rc: usize| DotGeneral {
        lhs_contracting: vec![lc],
        rhs_contracting: vec![rc],
        lhs_batch: vec![],
        rhs_batch: vec![],
    };
    let slice = |start: Vec<usize>, limit: Vec<usize>| Slice {
        start_indices: start,
        limit_indices: limit,
    };
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[16, 32]));
    let w = g.input(ty(DType::F32, &[32, 96]));
    let b = g.input(ty(DType::F32, &[32, 8]));
    let a = g.input(ty(DType::F32, &[4, 16]));
    let x3 = g.input(ty(DType::F32, &[2, 3, 8]));
    let c = g.input(ty(DType::F32, &[8, 4]));
    let h = apply(&mut g, dot(1, 0), &[x, w]);
    let q = apply(&mut g, slice(vec![0, 0], vec![16, 32]), &[h]);
    let v = apply(&mut g, slice(vec![0, 64], vec![16, 96]), &[h]);
    let y = apply(&mut g, dot(1, 0), &[q, b]); // lhs at row stride 96
    let z = apply(&mut g, dot(1, 0), &[a, v]); // rhs at row stride 96
    // Free dimensions [2, 2] of [2, 3]: not one strided dimension.
    let s = apply(&mut g, slice(vec![0, 0, 0], vec![2, 2, 8]), &[x3]);
    let u = apply(&mut g, dot(2, 0), &[s, c]);
    g.set_outputs(&[y, z, u]).unwrap();
    if !available() {
        return;
    }
    let plan = crate::compiler::compile(&g, crate::Device::Mps).unwrap();
    let names: Vec<_> = plan.steps().iter().map(|s| s.primitive.name()).collect();
    assert_eq!(names.iter().filter(|&&n| n == "slice").count(), 1, "{plan}");
    let views: Vec<_> = plan
        .steps()
        .iter()
        .flat_map(|s| s.views.iter().flatten())
        .collect();
    assert_eq!(views.len(), 2, "{plan}");
    assert_eq!((views[0].offset, &views[0].strides), (0, &vec![96, 1]));
    assert_eq!((views[1].offset, &views[1].strides), (64, &vec![96, 1]));
    let shapes = [
        vec![16, 32],
        vec![32, 96],
        vec![32, 8],
        vec![4, 16],
        vec![2, 3, 8],
        vec![8, 4],
    ];
    let inputs: Vec<Tensor> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| values(DType::F32, s, i as u64 + 1))
        .collect();
    let expected = reference::run(&g, &inputs).unwrap();
    let on_mps: Vec<Tensor> = inputs.iter().map(|t| t.to(crate::Device::Mps)).collect();
    for (e, a) in expected.iter().zip(plan.run(&on_mps).unwrap()) {
        for (e, a) in e
            .to_vec::<f32>()
            .iter()
            .zip(a.to(crate::Device::Cpu).to_vec::<f32>())
        {
            assert!((e - a).abs() <= 1e-3 * (1.0 + e.abs()), "{e} vs {a}");
        }
    }
}

/// A reduction fuses the primitives computing its input, in each of its
/// layouts (rows, split rows, columns, other axes), for sums and maxima of
/// several dtypes; its consumers read its output.
#[test]
fn reductions_fuse_their_inputs() {
    for (shape, axes) in [
        (vec![8, 300], vec![1]),
        (vec![2, 70_000], vec![1]),
        (vec![300, 5], vec![0]),
        (vec![2, 40_000, 3], vec![1]),
        (vec![4, 5, 6], vec![0, 2]),
    ] {
        for dtype in [DType::F32, DType::F16, DType::I32] {
            for sum in [true, false] {
                let mut g = Graph::new();
                let x = g.input(ty(dtype, &shape));
                let y = g.input(ty(dtype, &shape));
                // A product, so the reduction reads two inputs, then `neg`.
                let p = apply(&mut g, Mul, &[x, y]);
                let n = apply(&mut g, Neg, &[p]);
                let r = match sum {
                    true => ReduceSum { axes: axes.clone() },
                    false => ReduceMax { axes: axes.clone() },
                };
                let r = apply(&mut g, r, &[n]);
                let out = apply(&mut g, Neg, &[r]);
                g.set_outputs(&[out]).unwrap();
                let fused = fuse(&g);
                assert_eq!(names(&fused), ["fusion", "neg"], "{fused}");
                if available() {
                    let inputs = [values(dtype, &shape, 1), values(dtype, &shape, 2)];
                    // Sums of many floats: their accumulation error.
                    let slack = if dtype.is_float() && sum { 1.0 } else { 0.0 };
                    check_within(&g, &inputs, slack);
                }
            }
        }
    }
}

/// A value read by a reduction and by another fusion is computed in each
/// when cheap; an expensive one (exp) with two readers is stored once.
#[test]
fn reduction_inputs_shared_with_other_readers() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[4, 16]));
    let half = Full {
        shape: vec![4, 16],
        fill_value: Scalar::Float(0.5),
        dtype: DType::F32,
    };
    let half = apply(&mut g, half, &[]);
    let s = apply(&mut g, Mul, &[x, half]);
    let m = apply(&mut g, ReduceMax { axes: vec![1] }, &[s]);
    let m = apply(
        &mut g,
        BroadcastInDim {
            shape: vec![4, 16],
            broadcast_dimensions: vec![0],
        },
        &[m],
    );
    let d = apply(&mut g, Sub, &[s, m]);
    let e = apply(&mut g, Exp, &[d]);
    let z = apply(&mut g, ReduceSum { axes: vec![1] }, &[e]);
    let z = apply(
        &mut g,
        BroadcastInDim {
            shape: vec![4, 16],
            broadcast_dimensions: vec![0],
        },
        &[z],
    );
    let y = apply(&mut g, Div, &[e, z]);
    g.set_outputs(&[y]).unwrap();
    // The scale fuses into the max and into the exp; the sum reads the
    // stored exp.
    let fused = fused_primitives(&g, &[data(&[4, 16], 1)]);
    assert_eq!(fused, ["fusion", "fusion", "reduce_sum", "fusion"]);
    if available() {
        check(&g, &[values(DType::F32, &[4, 16], 1)]);
    }
}
