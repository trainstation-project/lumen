//! Tests of the MPS passes: fusion, dot merging and canonicalization, reduction and GEMM codegen,
//! checked against the reference interpreter and, when a GPU is present, on device.

use super::kernel_codegen;
use crate::codegen::dot_general::mps as gemm;
use crate::codegen::kinds::{gemm_dot, row_reductions};
use crate::codegen::layout::transpose_tiling;
use crate::codegen::reduction::mps::has_epilogue;
use crate::codegen::utils::mps::metal_type;
use crate::graph_compiler::common::merge_dots::{Target, merge_dots};

/// Dot merging target matching the MPS pipeline's.
const MPS: Target = Target {
    reads: super::fusion::fusible,
    views: true,
    lanes: 1,
    columns: crate::codegen::regions::ROW_COLUMNS,
};
use super::fusion;
use crate::graph::Primitive::*;
use crate::graph::reference;
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::graph_compiler::MpsConfig;
use crate::graph_compiler::plan_tests::mps::{
    available, check, check_within, exact_values, values,
};
use crate::graph_compiler::plan_tests::{data, mlp};
use crate::{DType, Device, Scalar, Tensor};

/// Shorthand for a tensor type.
fn ty(dtype: DType, shape: &[usize]) -> TensorType {
    TensorType::new(dtype, shape)
}

/// Runs MPS fusion with default config and normalization rows on `g`.
fn fuse(g: &Graph) -> Graph {
    let config = MpsConfig::default();

    fusion::fuse(
        g,
        &super::diamonds::diamonds(g),
        &[],
        &config,
        |body, by_value| {
            kernel_codegen::kernel(body, by_value, &config, &Default::default(), None).0
        },
    )
}

/// Fuses `g`, asserts the fused graph computes the same values on `inputs`, and returns its node names.
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

/// Node primitive names of `g`, with every fusion named `fusion`.
fn names(g: &Graph) -> Vec<&'static str> {
    let name = |p: &Primitive| match p {
        Fusion { .. } => "fusion",
        p => p.name(),
    };

    g.nodes().iter().map(|n| name(&n.primitive)).collect()
}

/// Applies `p` to `inputs`, panicking on a type error.
fn apply(g: &mut Graph, p: Primitive, inputs: &[Var]) -> Var {
    g.apply(p, inputs).unwrap()
}

/// Gathers index with `uint` for small tables and `ulong` once byte offsets pass 2^32.
#[test]
fn gathers_from_tables_past_u32_index_in_ulong() {
    let source = |rows: usize| {
        let mut g = Graph::new();
        let table = g.input(ty(DType::U8, &[rows, 4]));
        let index = g.input(ty(DType::I32, &[8]));
        let out = apply(&mut g, Gather { axis: 0 }, &[table, index]);
        g.set_outputs(&[out]).unwrap();
        kernel_codegen::kernel(&g, &[], &MpsConfig::default(), &Default::default(), None).1
    };

    let (narrow, wide) = (source(1000), source(1 << 31));

    assert!(
        !narrow.contains("ulong") && narrow.contains("uint i"),
        "{narrow}"
    );

    assert!(
        wide.contains("ulong i") && !wide.contains("uint i"),
        "{wide}"
    );
}

/// Elementwise, broadcast and transpose chains become one fusion labeled by its ops.
#[test]
fn fuses_elementwise_and_layout_chains() {
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

    assert_eq!(
        fuse(&g).nodes()[0].primitive.name(),
        "mul → tanh → transpose → add"
    );
}

/// An MLP fuses into one kernel per layer, split at its dots and reductions.
#[test]
fn reductions_and_contractions_are_boundaries() {
    let inputs = [data(&[4, 8], 1), data(&[8, 16], 2), data(&[16, 3], 3)];
    assert_eq!(fused_primitives(&mlp(), &inputs), ["fusion", "fusion"]);
}

/// A shared expensive value is hosted as an extra output; a shared cheap one is recomputed in each reader.
#[test]
fn only_cheap_values_are_recomputed() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[8]));
    let e = apply(&mut g, Exp, &[x]);
    let a = apply(&mut g, Add, &[e, x]);
    let b = apply(&mut g, Mul, &[e, x]);
    g.set_outputs(&[a, b]).unwrap();

    assert_eq!(
        fused_primitives(&g, &[data(&[8], 1)]),
        ["fusion", "fusion_output", "mul"]
    );

    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[8]));
    let c = apply(&mut g, Add, &[x, x]);
    let e = apply(&mut g, Exp, &[c]);
    let t = apply(&mut g, Tanh, &[c]);
    g.set_outputs(&[e, t]).unwrap();
    assert_eq!(fused_primitives(&g, &[data(&[8], 1)]), ["fusion", "fusion"]);
}

/// A chain reading 40 inputs is split into fusions binding at most 29 buffers.
#[test]
fn fusions_fit_in_a_kernel() {
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

/// Broadcasts, transposes, iota, select and casts fuse and match the reference for every dtype on device.
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

        if dtype.is_float() && dtype != DType::BF16 {
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
            Cast {
                new_dtype: DType::I32,
            },
            &[y],
        );

        g.set_outputs(&[t, i]).unwrap();

        let expected: &[&str] = if dtype.is_float() && dtype != DType::BF16 {
            &["fusion", "fusion", "cast"]
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

/// Dots whose operand layout isn't collapsible get explicit transposes; plain matmuls stay untouched.
#[test]
fn dot_operands_are_put_in_matmul_form_by_a_transpose_step() {
    let mut g = Graph::new();
    let a = g.input(ty(DType::F32, &[3, 2, 4]));
    let b = g.input(ty(DType::F32, &[4, 5, 3]));

    let dot = DotGeneral {
        lhs_contracting: vec![0, 2],
        rhs_contracting: vec![2, 0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
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
    let mut g = Graph::new();
    let a = g.input(ty(DType::F32, &[4, 8]));
    let b = g.input(ty(DType::F32, &[8, 3]));

    let mm = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };

    let y = apply(&mut g, mm, &[a, b]);
    g.set_outputs(&[y]).unwrap();
    assert_eq!(names(&super::canonicalize_dots(&g)), ["dot_general"]);
}

/// Large reductions get workspace scratch in the plan, and the generic plan refuses MPS inputs.
#[test]
fn mps_plans_hold_kernel_scratch() {
    if !available() {
        return;
    }

    let mut g = Graph::new();
    let x = g.input(ty(DType::F16, &[4, 50_000]));

    let s = apply(
        &mut g,
        ReduceSum {
            axes: vec![1],
            accum_dtype: DType::F16,
        },
        &[x],
    );

    g.set_outputs(&[s]).unwrap();
    let plan = crate::graph_compiler::compile(&g, crate::Device::Mps(0)).unwrap();
    let (_, bytes) = plan.steps()[0].scratch.expect("partials in the workspace");
    assert!(bytes > 0 && plan.workspace_bytes() >= bytes, "{plan}");
    let input = values(DType::F16, &[4, 50_000], 1).to(crate::Device::Mps(0));
    let generic = crate::graph_compiler::Plan::compile(&g).run(std::slice::from_ref(&input));
    assert!(generic.unwrap_err().contains("compile the graph for MPS"));
}

/// Two slices of one input fuse into the kernel reading them.
#[test]
fn slices_fuse_into_their_readers() {
    let mut g = Graph::new();
    let h = g.input(ty(DType::F32, &[16, 128]));

    let half = |g: &mut Graph, start: usize| {
        let s = Slice {
            start_indices: vec![0, start],
            limit_indices: vec![16, start + 64],
            view: false,
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

/// Reference concatenation of `parts` along `dimension`, the layout of a packed block.
fn block(parts: &[&Tensor], dimension: usize) -> Tensor {
    let mut g = Graph::new();

    let vars: Vec<Var> = parts
        .iter()
        .map(|t| g.input(ty(t.dtype(), t.shape())))
        .collect();

    let out = apply(
        &mut g,
        Concatenate {
            dimension,
            in_place: false,
        },
        &vars,
    );

    g.set_outputs(&[out]).unwrap();
    let parts: Vec<Tensor> = parts.iter().map(|&t| t.clone()).collect();
    reference::run(&g, &parts).unwrap().remove(0)
}

/// Reference interleaving of `parts` element by element along `dimension`.
fn interleaved(parts: &[&Tensor], dimension: usize) -> Tensor {
    let mut g = Graph::new();
    let shape = parts[0].shape().to_vec();
    let mut one = shape.clone();
    one.insert(dimension + 1, 1);

    let ones: Vec<Var> = parts
        .iter()
        .map(|t| {
            let v = g.input(ty(t.dtype(), t.shape()));

            apply(
                &mut g,
                Reshape {
                    new_sizes: one.clone(),
                },
                &[v],
            )
        })
        .collect();

    let tuples = apply(
        &mut g,
        Concatenate {
            dimension: dimension + 1,
            in_place: false,
        },
        &ones,
    );

    let mut flat = shape;
    flat[dimension] *= parts.len();
    let out = apply(&mut g, Reshape { new_sizes: flat }, &[tuples]);
    g.set_outputs(&[out]).unwrap();
    let parts: Vec<Tensor> = parts.iter().map(|&t| t.clone()).collect();
    reference::run(&g, &parts).unwrap().remove(0)
}

/// Transposes in fusions use threadgroup tiles exactly for shapes where the tiled kernel pays off.
#[test]
fn transposes_in_fusions_are_tiled() {
    let cases: [(&[usize], &[usize], bool); 6] = [
        (&[50, 7, 70], &[2, 1, 0], true),
        (&[40, 36], &[1, 0], true),
        (&[4, 40, 1, 36], &[0, 3, 2, 1], true),
        (&[33, 5, 3, 40], &[3, 1, 2, 0], true),
        (&[300, 8], &[1, 0], false),
        (&[6, 40, 32], &[1, 0, 2], false),
    ];

    for dtype in [DType::F32, DType::BF16, DType::I32] {
        for (shape, permutation, tiled) in cases {
            let out: Vec<usize> = permutation.iter().map(|&d| shape[d]).collect();
            let mut g = Graph::new();
            let x = g.input(ty(dtype, shape));
            let y = g.input(ty(dtype, &out));

            let transpose = || Transpose {
                permutation: permutation.to_vec(),
            };

            let sq = apply(&mut g, Mul, &[x, x]);
            let t = apply(&mut g, transpose(), &[sq]);
            let m = apply(&mut g, Mul, &[t, y]);
            let t2 = apply(&mut g, transpose(), &[x]);
            let mut r = apply(&mut g, Add, &[m, t2]);

            if dtype.is_float() {
                let f = Cast {
                    new_dtype: DType::F32,
                };

                r = apply(&mut g, f, &[r]);
                r = apply(&mut g, Exp, &[r]);
            }

            g.set_outputs(&[r]).unwrap();
            let fused = fuse(&g);

            let Fusion { body, .. } = &fused.nodes()[0].primitive else {
                panic!("{fused}")
            };

            assert_eq!(fused.nodes().len(), 1, "{fused}");

            let source =
                kernel_codegen::kernel(body, &[], &MpsConfig::default(), &Default::default(), None)
                    .1;

            assert_eq!(transpose_tiling(body).is_some(), tiled, "{shape:?}");

            assert_eq!(source.contains("threadgroup_barrier"), tiled, "{source}");

            if tiled {
                assert!(source.contains("tile1["), "{source}");
            }

            if available() {
                let inputs = [values(dtype, shape, 1), values(dtype, &out, 2)];
                check(&g, &inputs);
            }
        }
    }
}

/// Reductions pick row, column or grouped kernels after squeezing unit axes, with matching scratch.
#[test]
fn reductions_are_normalized() {
    use crate::codegen::reduction::mps::scratch_bytes;

    for (shape, axes, kernel) in [
        (vec![400, 1, 300], vec![0, 2], "reduce_rows"),
        (vec![4, 1, 64, 1, 300], vec![1, 2, 3, 4], "reduce_rows"),
        (vec![300, 1, 7, 1], vec![0, 1], "reduce_cols"),
        (vec![300, 1, 7], vec![1], "reduce_cols"),
        (vec![1, 1024, 1024], vec![0], "reduce_cols"),
        (vec![40, 1, 7, 1, 300], vec![0, 1, 4], "reduce_grouped"),
    ] {
        let mut g = Graph::new();
        let x = g.input(ty(DType::F32, &shape));
        let e = apply(&mut g, Exp, &[x]);

        let sum = ReduceSum {
            axes: axes.clone(),
            accum_dtype: DType::F32,
        };

        let r = apply(&mut g, sum, &[e]);
        g.set_outputs(&[r]).unwrap();
        let fused = fuse(&g);

        let Fusion { body, .. } = &fused.nodes()[0].primitive else {
            panic!("{fused}")
        };

        let source =
            kernel_codegen::kernel(body, &[], &MpsConfig::default(), &Default::default(), None).1;

        assert!(
            source.contains(&format!("{kernel}<")),
            "{shape:?} {axes:?}: {source}"
        );
    }

    let squeezed = scratch_bytes(&ty(DType::F32, &[4, 50_000]), &[1], DType::F32);

    let spread = scratch_bytes(
        &ty(DType::F32, &[4, 1, 500, 1, 100]),
        &[1, 2, 4],
        DType::F32,
    );

    assert!(squeezed > 0);
    assert_eq!(spread, squeezed);
}

/// Half-precision reductions accumulate in float only when asked, including inside norm and softmax rows.
#[test]
fn hierarchical_reductions_keep_their_dtype() {
    use crate::codegen::reduction::mps::scratch_bytes;

    let mentions = |source: &str, word: &str| {
        source
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .any(|w| w == word)
    };

    for dtype in [DType::F16, DType::BF16] {
        for (shape, axes) in [
            (vec![4, 200_000], vec![1]),
            (vec![200_000, 4], vec![0]),
            (vec![40, 7, 300], vec![0, 2]),
            (vec![8, 300], vec![1]),
        ] {
            let wide = scratch_bytes(&ty(dtype, &shape), &axes, DType::F32);
            assert_eq!(scratch_bytes(&ty(dtype, &shape), &axes, dtype) * 2, wide);

            for accum_dtype in [dtype, DType::F32] {
                let mut g = Graph::new();
                let x = g.input(ty(dtype, &shape));
                let p = apply(&mut g, Mul, &[x, x]);

                let sum = ReduceSum {
                    axes: axes.clone(),
                    accum_dtype,
                };

                let r = apply(&mut g, sum, &[p]);
                g.set_outputs(&[r]).unwrap();
                let fused = fuse(&g);

                let Fusion { body, .. } = &fused.nodes()[0].primitive else {
                    panic!("{fused}")
                };

                let source = kernel_codegen::kernel(
                    body,
                    &[],
                    &MpsConfig::default(),
                    &Default::default(),
                    None,
                )
                .1;

                match accum_dtype {
                    DType::F32 => {
                        let (t, a) = (metal_type(dtype), "float");

                        for part in [
                            format!("<Add, {a}>("),
                            format!("device {a} *out"),
                            format!("inline {t} operator[]"),
                        ] {
                            assert!(source.contains(&part), "{part}: {source}");
                        }
                    }
                    _ => assert!(!mentions(&source, "float"), "{source}"),
                }
            }
        }
    }

    let mut g = Graph::new();
    let x = g.input(ty(DType::F16, &[8, 300]));
    let sq = apply(&mut g, Mul, &[x, x]);

    let sum = apply(
        &mut g,
        ReduceSum {
            axes: vec![1],
            accum_dtype: DType::F16,
        },
        &[sq],
    );

    let b = BroadcastInDim {
        shape: vec![8, 300],
        broadcast_dimensions: vec![0],
    };

    let sum = apply(&mut g, b.clone(), &[sum]);
    let norm = apply(&mut g, Sqrt, &[sum]);
    let y = apply(&mut g, Div, &[x, norm]);
    let m = apply(&mut g, ReduceMax { axes: vec![1] }, &[y]);
    let m = apply(&mut g, b.clone(), &[m]);
    let d = apply(&mut g, Sub, &[y, m]);
    let e = apply(&mut g, Exp, &[d]);

    let sum = ReduceSum {
        axes: vec![1],
        accum_dtype: DType::F16,
    };

    let z = apply(&mut g, sum, &[e]);
    let z = apply(&mut g, b, &[z]);
    let s = apply(&mut g, Div, &[e, z]);
    g.set_outputs(&[y, s]).unwrap();
    let fused = fuse(&g);

    for node in fused.nodes() {
        if let Fusion { body, .. } = &node.primitive {
            let source =
                kernel_codegen::kernel(body, &[], &MpsConfig::default(), &Default::default(), None)
                    .1;

            assert!(!mentions(&source, "float"), "{source}");
        }
    }
}

/// Dots of weights scaled and cast elementwise merge when the elementwise ops match, sharing one cast.
#[test]
fn dots_of_weights_computed_elementwise_merge() {
    let dot = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![1],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::BF16,
    };

    let build = |scales: [f64; 3]| {
        let mut g = Graph::new();
        let x = g.input(ty(DType::BF16, &[16, 32]));
        let ws: Vec<Var> = (0..3).map(|_| g.input(ty(DType::F32, &[64, 32]))).collect();
        let mut outs = Vec::new();
        let mut operands = Vec::new();

        for (&w, scale) in ws.iter().zip(scales) {
            let fill = Full {
                shape: vec![],
                fill_value: Scalar::Float(scale),
                dtype: DType::F32,
            };

            let s = apply(&mut g, fill, &[]);

            let splat = BroadcastInDim {
                shape: vec![64, 32],
                broadcast_dimensions: vec![],
            };

            let s = apply(&mut g, splat, &[s]);
            let scaled = apply(&mut g, Mul, &[w, s]);

            let cast = apply(
                &mut g,
                Cast {
                    new_dtype: DType::BF16,
                },
                &[scaled],
            );

            operands.push(cast);
            let y = apply(&mut g, dot.clone(), &[x, cast]);
            outs.push(apply(&mut g, Neg, &[y]));
        }

        let xt = apply(
            &mut g,
            Transpose {
                permutation: vec![1, 0],
            },
            &[x],
        );

        let other = DotGeneral {
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
            accum_dtype: DType::F32,
            output_dtype: DType::F32,
        };

        outs.push(apply(&mut g, other, &[xt, x]));

        for k in [0, 2] {
            let again = apply(&mut g, dot.clone(), &[x, operands[k]]);
            outs.push(apply(&mut g, Neg, &[again]));
        }

        g.set_outputs(&outs).unwrap();
        g
    };

    let packable = [false, true, true, true];
    let g = build([0.5; 3]);
    let (merged, packs, _) = merge_dots(&g, &packable, &[], &MPS);
    assert_eq!(packs, [(vec![1, 2, 3], 0)], "{merged}");

    let casts = names(&merged.prune().0)
        .iter()
        .filter(|&&n| n == "cast")
        .count();

    assert_eq!(casts, 1, "{merged}");

    let mut inputs = vec![
        data(&[16, 32], 1).to_dtype(DType::BF16).unwrap(),
        data(&[64, 32], 2),
        data(&[64, 32], 3),
        data(&[64, 32], 4),
    ];

    inputs.push(block(&[&inputs[1], &inputs[2], &inputs[3]], 0));
    let expected = reference::run(&g, &inputs[..4]).unwrap();
    let f32s = |t: &Tensor| t.to_dtype(DType::F32).unwrap().to_vec::<f32>();

    for (e, a) in expected
        .iter()
        .zip(reference::run(&merged, &inputs).unwrap())
    {
        assert_eq!(f32s(e), f32s(&a));
    }

    let (merged, packs, _) = merge_dots(&build([0.5, 0.25, 0.5]), &packable, &[], &MPS);
    assert_eq!(packs, [(vec![1, 3], 0)], "{merged}");
}

/// Donated weights only merge when they pack along their leading axis, so stay contiguous.
#[test]
fn donated_weights_merge_only_contiguously() {
    let merged = |rhs_contracting: usize, shape: [usize; 2]| {
        let dot = DotGeneral {
            lhs_contracting: vec![1],
            rhs_contracting: vec![rhs_contracting],
            lhs_batch: vec![],
            rhs_batch: vec![],
            accum_dtype: DType::F32,
            output_dtype: DType::F32,
        };

        let mut g = Graph::new();
        let x = g.input(ty(DType::F32, &[16, 32]));
        let ws: Vec<Var> = (0..3).map(|_| g.input(ty(DType::F32, &shape))).collect();

        let ys: Vec<Var> = ws
            .iter()
            .map(|&w| {
                let y = apply(&mut g, dot.clone(), &[x, w]);
                apply(&mut g, Neg, &[y])
            })
            .collect();

        g.set_outputs(&ys).unwrap();
        let packable = [false, true, true, true];

        (
            merge_dots(&g, &packable, &[], &MPS).1,
            merge_dots(&g, &packable, &packable, &MPS).1,
        )
    };

    assert_eq!(merged(0, [32, 64]), (vec![(vec![1, 2, 3], 1)], vec![]));
    let along_first = vec![(vec![1, 2, 3], 0)];
    assert_eq!(merged(1, [64, 32]), (along_first.clone(), along_first));
}

/// Only dots with the same accumulation dtype merge.
#[test]
fn dots_accumulating_in_other_dtypes_do_not_merge() {
    let dot = |accum_dtype| DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype,
        output_dtype: accum_dtype,
    };

    let dots = |accums: [DType; 3]| {
        let mut g = Graph::new();
        let x = g.input(ty(DType::BF16, &[16, 32]));

        let ws: Vec<Var> = (0..3)
            .map(|_| g.input(ty(DType::BF16, &[32, 64])))
            .collect();

        let ys: Vec<Var> = ws
            .iter()
            .zip(accums)
            .map(|(&w, a)| {
                let y = apply(&mut g, dot(a), &[x, w]);
                apply(&mut g, Neg, &[y])
            })
            .collect();

        g.set_outputs(&ys).unwrap();
        merge_dots(&g, &[false, true, true, true], &[], &MPS)
    };

    let (merged, packs, _) = dots([DType::F32; 3]);
    assert_eq!(packs, [(vec![1, 2, 3], 1)], "{merged}");
    let (merged, packs, _) = dots([DType::F32, DType::BF16, DType::F32]);
    assert_eq!(packs, [(vec![1, 3], 1)], "{merged}");

    let accums: Vec<DType> = merged
        .nodes()
        .iter()
        .filter_map(|n| match n.primitive {
            DotGeneral { accum_dtype, .. } => Some(accum_dtype),
            _ => None,
        })
        .collect();

    assert_eq!(accums, [DType::F32, DType::BF16], "{merged}");
}

/// Dots sharing an operand merge into one GEMM, interleaving gated pairs, and match the reference.
#[test]
fn dots_sharing_an_operand_merge() {
    let matmul = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };

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
    assert_eq!(dots(&merge_dots(&g, &[false; 4], &[], &MPS).0), 3);
    let (merged, packs, interleave) = merge_dots(&g, &[false, true, true, true], &[], &MPS);
    assert_eq!(packs, [(vec![1, 2], 1)]);
    assert_eq!(interleave, [vec![2]]);
    assert_eq!(dots(&merged), 2, "{merged}");
    assert_eq!(names(&fuse(&merged)), ["fusion", "dot_general"]);
    let mut read_twice = g.clone();
    let w3_again = apply(&mut read_twice, Neg, &[w3]);
    read_twice.set_outputs(&[y, w3_again]).unwrap();

    assert!(
        merge_dots(&read_twice, &[false, true, true, true], &[], &MPS)
            .1
            .is_empty()
    );

    let mut inputs = vec![
        data(&[16, 32], 1),
        data(&[32, 64], 2),
        data(&[32, 64], 3),
        data(&[64, 32], 4),
    ];

    inputs.push(interleaved(&[&inputs[1], &inputs[2]], 1));
    let expected = reference::run(&g, &inputs[..4]).unwrap();

    assert_eq!(
        reference::run(&merged, &inputs).unwrap()[0].to_vec::<f32>(),
        expected[0].to_vec::<f32>()
    );

    let mut alone = Graph::new();
    let x = alone.input(ty(DType::F32, &[16, 32]));

    let ws: Vec<Var> = (0..2)
        .map(|_| alone.input(ty(DType::F32, &[32, 64])))
        .collect();

    let outs: Vec<Var> = ws
        .iter()
        .map(|&w| {
            let y = apply(&mut alone, matmul.clone(), &[x, w]);
            apply(&mut alone, Neg, &[y])
        })
        .collect();

    alone.set_outputs(&outs).unwrap();
    let packed = merge_dots(&alone, &[false, true, true], &[], &MPS);

    assert_eq!(
        (packed.1, packed.2),
        (vec![(vec![1, 2], 1)], vec![vec![1, 1]])
    );

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
    let (merged, packs, _) = merge_dots(&g, &[true; 4], &[], &MPS);
    assert_eq!(packs, [(vec![1, 2, 3], 0)]);
    assert_eq!(dots(&merged), 3, "{merged}");

    let mut inputs: Vec<Tensor> = [vec![32, 32], vec![4, 32], vec![6, 32], vec![4, 32]]
        .iter()
        .enumerate()
        .map(|(i, shape)| data(shape, i as u64 + 1))
        .collect();

    inputs.push(block(&[&inputs[1], &inputs[2], &inputs[3]], 0));
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

/// Gate and up projections interleave, unless their concatenated block is read, which packs them side by side.
#[test]
fn gated_pairs_interleave_unless_their_block_is_read_side_by_side() {
    let linear = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![1],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };

    let build = |extra: &dyn Fn(&mut Graph, Var, Var) -> Option<Var>| {
        let mut g = Graph::new();
        let x = g.input(ty(DType::F32, &[16, 32]));
        let w1 = g.input(ty(DType::F32, &[64, 32]));
        let w3 = g.input(ty(DType::F32, &[64, 32]));
        let gate = apply(&mut g, linear.clone(), &[x, w1]);
        let up = apply(&mut g, linear.clone(), &[x, w3]);
        let y = apply(&mut g, Mul, &[gate, up]);
        let mut outs = vec![y];
        outs.extend(extra(&mut g, w1, w3));
        g.set_outputs(&outs).unwrap();
        g
    };

    let packable = [false, true, true];

    let pack = |g: &Graph, donated: &[bool]| {
        let (_, packs, interleave) = merge_dots(g, &packable, donated, &MPS);
        (packs, interleave)
    };

    let rows = vec![(vec![1, 2], 0)];

    assert_eq!(
        pack(&build(&|_, _, _| None), &[]),
        (rows.clone(), vec![vec![2]])
    );

    let read = build(&|g, _, w3| Some(apply(g, Neg, &[w3])));
    assert_eq!(pack(&read, &[]), (rows.clone(), vec![vec![2]]));

    let concatenated = build(&|g, w1, w3| {
        let block = Concatenate {
            dimension: 0,
            in_place: false,
        };

        Some(apply(g, block, &[w1, w3]))
    });

    let (merged, packs, interleave) = merge_dots(&concatenated, &packable, &[], &MPS);
    assert_eq!((packs, interleave), (rows.clone(), vec![vec![1, 1]]));
    assert!(!names(&merged).contains(&"concatenate"), "{merged}");

    assert_eq!(
        pack(&build(&|_, _, _| None), &[false, true, true]),
        (rows, vec![vec![2]])
    );
}

/// A compiled plan with packed parameters runs on placed views of one shared block.
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
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
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

    let options = crate::graph_compiler::Options {
        parameters: Some(params.clone()),
        packable: params,
        ..Default::default()
    };

    let plan = crate::graph_compiler::compile_with(&g, Device::Mps(0), &options).unwrap();
    assert_eq!(plan.packed, [(vec![1, 2], 1)]);
    assert_eq!(plan.interleaved, [vec![2]]);
    let meta = || Tensor::zeros(&[32, 64], Device::Meta(0));
    let (m1, m3) = (meta(), meta());

    let block = Tensor::pack(&[m1.clone(), m3.clone()], 1, &[2], Device::Mps(0))
        .unwrap()
        .unwrap();

    let again = Tensor::pack(&[m1.clone(), m3.clone()], 1, &[2], Device::Mps(0)).unwrap();
    assert!(again.is_some_and(|b| b.shares_storage_with(&block)));

    for (members, regions) in [([&m3, &m1], vec![2]), ([&m1, &m3], vec![1, 1])] {
        let members: Vec<Tensor> = members.iter().map(|&m| m.clone()).collect();
        let packed = Tensor::pack(&members, 1, &regions, Device::Mps(0));
        assert!(packed.unwrap().is_none());
    }

    let inputs = [data(&[16, 32], 1), data(&[32, 64], 2), data(&[32, 64], 3)];

    let (p1, p3) = (
        m1.placed(Device::Mps(0)).unwrap(),
        m3.placed(Device::Mps(0)).unwrap(),
    );

    assert!(p1.shares_storage_with(&block) && !p1.is_contiguous());
    p1.copy_(&inputs[1]).unwrap();
    p3.copy_(&inputs[2]).unwrap();

    let workspace = Tensor::zeros(
        &[plan.workspace_bytes()],
        crate::TensorOptions::new()
            .dtype(DType::U8)
            .device(Device::Mps(0)),
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

        for (e, a) in expected
            .iter()
            .zip(out[0].to(Device::Cpu(0)).to_vec::<f32>())
        {
            assert!((e - a).abs() <= 1e-4 * (1.0 + e.abs()), "{e} vs {a}");
        }
    }
}

/// Bias add and ReLU after a dot fuse into its GEMM epilogue.
#[test]
fn dots_fuse_their_epilogue() {
    let build = |read_twice: bool| {
        let mut g = Graph::new();
        let x = g.input(ty(DType::F32, &[40, 24]));
        let w = g.input(ty(DType::F32, &[24, 72]));
        let bias = g.input(ty(DType::F32, &[72]));

        let dot = DotGeneral {
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
            accum_dtype: DType::F32,
            output_dtype: DType::F32,
        };

        let y = apply(&mut g, dot, &[x, w]);

        let b = BroadcastInDim {
            shape: vec![40, 72],
            broadcast_dimensions: vec![1],
        };

        let b = apply(&mut g, b, &[bias]);
        let s = apply(&mut g, Add, &[y, b]);

        let zero = Full {
            shape: vec![40, 72],
            fill_value: Scalar::Float(0.0),
            dtype: DType::F32,
        };

        let zero = apply(&mut g, zero, &[]);
        let r = apply(&mut g, Max, &[s, zero]);
        let outputs = if read_twice { vec![r, y] } else { vec![r] };
        g.set_outputs(&outputs).unwrap();
        g
    };

    let inputs = [data(&[40, 24], 1), data(&[24, 72], 2), data(&[72], 3)];
    let g = build(false);
    assert_eq!(fused_primitives(&g, &inputs), ["fusion"]);
    let fused = fuse(&g);

    let Fusion { body, label, .. } = &fused.nodes()[0].primitive else {
        panic!("{fused}")
    };

    assert!(label.starts_with("dot_general"), "{label}");
    let source =
        kernel_codegen::kernel(body, &[], &MpsConfig::default(), &Default::default(), None).1;

    assert!(
        source.contains("matmul_sg_impl") && source.contains("_small("),
        "{source}"
    );

    assert_eq!(
        fused_primitives(&build(true), &inputs),
        ["fusion", "fusion_output"]
    );

    let config = MpsConfig {
        contraction_epilogues: false,
        ..MpsConfig::default()
    };

    let unfused = fusion::fuse(
        &g,
        &super::diamonds::diamonds(&g),
        &[],
        &config,
        |body, by_value| {
            kernel_codegen::kernel(body, by_value, &config, &Default::default(), None).0
        },
    );

    assert_eq!(names(&unfused), ["dot_general", "fusion"]);

    if available() {
        check(&g, &inputs);
        check(&build(true), &inputs);
    }
}

/// Slices of a dot's output read by later dots become strided views instead of copies.
#[test]
fn dots_read_slices_in_place() {
    let dot = |lc: usize, rc: usize| DotGeneral {
        lhs_contracting: vec![lc],
        rhs_contracting: vec![rc],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };

    let slice = |start: Vec<usize>, limit: Vec<usize>| Slice {
        start_indices: start,
        limit_indices: limit,
        view: false,
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
    let y = apply(&mut g, dot(1, 0), &[q, b]);
    let z = apply(&mut g, dot(1, 0), &[a, v]);
    let s = apply(&mut g, slice(vec![0, 0, 0], vec![2, 2, 8]), &[x3]);
    let u = apply(&mut g, dot(2, 0), &[s, c]);
    g.set_outputs(&[y, z, u]).unwrap();

    if !available() {
        return;
    }

    let plan = crate::graph_compiler::compile(&g, crate::Device::Mps(0)).unwrap();
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
    let on_mps: Vec<Tensor> = inputs.iter().map(|t| t.to(crate::Device::Mps(0))).collect();

    for (e, a) in expected.iter().zip(plan.run(&on_mps).unwrap()) {
        for (e, a) in e
            .to_vec::<f32>()
            .iter()
            .zip(a.to(crate::Device::Cpu(0)).to_vec::<f32>())
        {
            assert!((e - a).abs() <= 1e-3 * (1.0 + e.abs()), "{e} vs {a}");
        }
    }
}

/// Elementwise producers and consumers fuse into a sum or max reduction for every dtype and layout.
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
                let p = apply(&mut g, Mul, &[x, y]);
                let n = apply(&mut g, Neg, &[p]);

                let r = match sum {
                    true => ReduceSum {
                        axes: axes.clone(),
                        accum_dtype: dtype,
                    },
                    false => ReduceMax { axes: axes.clone() },
                };

                let r = apply(&mut g, r, &[n]);
                let out = apply(&mut g, Neg, &[r]);
                g.set_outputs(&[out]).unwrap();
                let fused = fuse(&g);
                assert_eq!(names(&fused), ["fusion"], "{fused}");

                if available() {
                    let count: usize = axes.iter().map(|&d| shape[d]).product();

                    let inputs = match dtype {
                        DType::F16 if sum => {
                            let period = count.div_ceil(128);
                            [1, 2].map(|seed| exact_values(dtype, &shape, seed, period))
                        }
                        _ => [values(dtype, &shape, 1), values(dtype, &shape, 2)],
                    };

                    let slack = if dtype == DType::F32 && sum { 1.0 } else { 0.0 };
                    check_within(&g, &inputs, slack);
                }
            }
        }
    }
}

/// A cast, sum, mean and cast back run as one reduction with an epilogue, split into a final launch when large.
#[test]
fn reductions_apply_their_epilogue() {
    for (shape, axes, split) in [
        (vec![64, 300], vec![1], false),
        (vec![4, 200_000], vec![1], true),
        (vec![100_000, 8], vec![0], true),
        (vec![40, 7, 30], vec![0, 2], false),
    ] {
        for mean in [false, true] {
            let mut g = Graph::new();
            let x = g.input(ty(DType::BF16, &shape));

            let w = apply(
                &mut g,
                Cast {
                    new_dtype: DType::F32,
                },
                &[x],
            );

            let sum = ReduceSum {
                axes: axes.clone(),
                accum_dtype: DType::F32,
            };

            let mut s = apply(&mut g, sum, &[w]);

            if mean {
                let count: usize = axes.iter().map(|&d| shape[d]).product();

                let n = Full {
                    shape: vec![],
                    fill_value: Scalar::Float(count as f64),
                    dtype: DType::F32,
                };

                let n = apply(&mut g, n, &[]);

                let b = BroadcastInDim {
                    shape: g.type_of(s).shape.clone(),
                    broadcast_dimensions: vec![],
                };

                let n = apply(&mut g, b, &[n]);
                s = apply(&mut g, Div, &[s, n]);
            }

            let y = apply(
                &mut g,
                Cast {
                    new_dtype: DType::BF16,
                },
                &[s],
            );

            g.set_outputs(&[y]).unwrap();
            let fused = fuse(&g);
            assert_eq!(names(&fused), ["fusion"], "{fused}");

            let Fusion { body, .. } = &fused.nodes()[0].primitive else {
                panic!("{fused}")
            };

            assert!(has_epilogue(body));

            let source =
                kernel_codegen::kernel(body, &[], &MpsConfig::default(), &Default::default(), None)
                    .1;

            assert!(source.contains("operator()(float r)"), "{source}");
            assert_eq!(source.contains("_final("), split, "{source}");

            if available() {
                check(&g, &[values(DType::BF16, &shape, 1)]);
            }
        }
    }
}

/// A reduction epilogue may read the reduced value more than once.
#[test]
fn reductions_apply_an_epilogue_reading_them_twice() {
    for (shape, split) in [(vec![64, 300], false), (vec![4, 200_000], true)] {
        let mut g = Graph::new();
        let x = g.input(ty(DType::F32, &shape));

        let sum = ReduceSum {
            axes: vec![1],
            accum_dtype: DType::F32,
        };

        let s = apply(&mut g, sum, &[x]);
        let sigmoid = apply(&mut g, Logistic, &[s]);
        let y = apply(&mut g, Mul, &[s, sigmoid]);
        g.set_outputs(&[y]).unwrap();
        let fused = fuse(&g);
        assert_eq!(names(&fused), ["fusion"], "{fused}");

        let Fusion { body, .. } = &fused.nodes()[0].primitive else {
            panic!("{fused}")
        };

        assert!(has_epilogue(body));
        let source =
            kernel_codegen::kernel(body, &[], &MpsConfig::default(), &Default::default(), None).1;
        assert_eq!(source.contains("_final("), split, "{source}");

        if available() {
            let slack = shape[1] as f64 * 2.0 * f64::from(f32::EPSILON);
            check_within(&g, &[values(DType::F32, &shape, 1)], slack);
        }
    }
}

/// A softmax built from primitives becomes one row kernel with two row reductions.
#[test]
fn softmax_written_out_is_one_row_kernel() {
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

    let z = apply(
        &mut g,
        ReduceSum {
            axes: vec![1],
            accum_dtype: DType::F32,
        },
        &[e],
    );

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
    let fused = fused_primitives(&g, &[data(&[4, 16], 1)]);
    assert_eq!(fused, ["fusion"]);
    let fused = fuse(&g);

    let Fusion { body, .. } = &fused.nodes()[0].primitive else {
        unreachable!("a fusion")
    };

    assert_eq!(row_reductions(body).len(), 2);

    if available() {
        check(&g, &[values(DType::F32, &[4, 16], 1)]);
    }
}

/// A fusion writes values other kernels also read as extra outputs instead of recomputing them.
#[test]
fn multi_output_fusion() {
    for (shape, axes) in [
        (vec![8, 300], vec![1]),
        (vec![2, 70_000], vec![1]),
        (vec![300, 5], vec![0]),
        (vec![4, 5, 6], vec![0, 2]),
    ] {
        let mut g = Graph::new();
        let x = g.input(ty(DType::F32, &shape));
        let y = g.input(ty(DType::F32, &shape));
        let p = apply(&mut g, Mul, &[x, y]);
        let e = apply(&mut g, Exp, &[p]);

        let s = apply(
            &mut g,
            ReduceSum {
                axes: axes.clone(),
                accum_dtype: DType::F32,
            },
            &[e],
        );

        let n = apply(&mut g, Neg, &[e]);
        g.set_outputs(&[s, n]).unwrap();
        let inputs = [data(&shape, 1), data(&shape, 2)];
        let fused = fused_primitives(&g, &inputs);
        assert_eq!(fused, ["fusion", "fusion_output", "neg"], "{shape:?}");

        if available() {
            check_within(&g, &inputs, 1.0);
        }
    }

    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[64]));
    let e = apply(&mut g, Exp, &[x]);
    let a = apply(&mut g, Neg, &[e]);
    let b = apply(&mut g, Tanh, &[e]);
    g.set_outputs(&[a, b]).unwrap();

    assert_eq!(
        fused_primitives(&g, &[data(&[64], 1)]),
        ["fusion", "fusion_output", "tanh"]
    );

    if available() {
        check(&g, &[values(DType::F32, &[64], 1)]);
    }

    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[4, 8]));
    let e = apply(&mut g, Exp, &[x]);
    let first = apply(&mut g, ReduceMax { axes: vec![1] }, &[e]);

    let t = apply(
        &mut g,
        Transpose {
            permutation: vec![1, 0],
        },
        &[e],
    );

    let second = apply(
        &mut g,
        ReduceSum {
            axes: vec![0],
            accum_dtype: DType::F32,
        },
        &[t],
    );

    g.set_outputs(&[first, second]).unwrap();
    let fused = fused_primitives(&g, &[data(&[4, 8], 1)]);
    assert_eq!(fused, ["fusion", "fusion_output", "fusion"]);
}

/// RMS norms, weighted or not and in either operand order, fuse into one row kernel unless an intermediate is read
/// or the reduction isn't over the last axis.
#[test]
fn rms_norms_are_one_row_kernel() {
    let build = |weighted: bool, flip: bool, n: f64, axis: usize, leak: bool| {
        let mut g = Graph::new();
        let x = g.input(ty(DType::F32, &[8, 300]));
        let w = g.input(ty(DType::F32, &[300]));
        let pair = |a, b| if flip { [b, a] } else { [a, b] };
        let sq = apply(&mut g, Mul, &[x, x]);

        let sum = apply(
            &mut g,
            ReduceSum {
                axes: vec![axis],
                accum_dtype: DType::F32,
            },
            &[sq],
        );

        let mut kept = vec![8, 300];
        kept[axis] = 1;

        let sum = apply(
            &mut g,
            Reshape {
                new_sizes: kept.clone(),
            },
            &[sum],
        );

        let scalar = |g: &mut Graph, v: f64| {
            let c = Full {
                shape: vec![],
                fill_value: Scalar::Float(v),
                dtype: DType::F32,
            };

            let c = apply(g, c, &[]);

            let b = BroadcastInDim {
                shape: kept.clone(),
                broadcast_dimensions: vec![],
            };

            apply(g, b, &[c])
        };

        let n = scalar(&mut g, n);
        let mean = apply(&mut g, Div, &[sum, n]);
        let eps = scalar(&mut g, 1e-6);
        let v = apply(&mut g, Add, &pair(mean, eps));
        let s = apply(&mut g, Sqrt, &[v]);
        let one = scalar(&mut g, 1.0);
        let r = apply(&mut g, Div, &[one, s]);

        let b = BroadcastInDim {
            shape: vec![8, 300],
            broadcast_dimensions: vec![0, 1],
        };

        let r = apply(&mut g, b, &[r]);
        let mut y = apply(&mut g, Mul, &pair(x, r));

        if weighted {
            let b = BroadcastInDim {
                shape: vec![8, 300],
                broadcast_dimensions: vec![1],
            };

            let w = apply(&mut g, b, &[w]);
            y = apply(&mut g, Mul, &pair(y, w));
        }

        let outputs = if leak { vec![y, mean] } else { vec![y] };
        g.set_outputs(&outputs).unwrap();
        g
    };

    let inputs = [data(&[8, 300], 1), data(&[300], 2)];
    let rewritten = |g: &Graph| fused_primitives(g, &inputs) == ["fusion"];

    for weighted in [false, true] {
        for flip in [false, true] {
            let g = build(weighted, flip, 300.0, 1, false);
            assert!(rewritten(&g), "weighted {weighted}, flipped {flip}");

            if available() {
                let inputs = [
                    values(DType::F32, &[8, 300], 1),
                    values(DType::F32, &[300], 2),
                ];

                check(&g, &inputs);
            }
        }
    }

    assert!(
        !rewritten(&build(true, false, 300.0, 1, true)),
        "a read intermediate"
    );

    assert!(
        rewritten(&build(true, false, 299.0, 1, false)),
        "not a mean"
    );

    assert!(
        !rewritten(&build(false, false, 8.0, 0, false)),
        "not the last dimension"
    );
}

/// An RMS norm dividing by the square root directly, without epsilon, is still one row kernel.
#[test]
fn rms_norms_without_epsilon_are_one_row_kernel() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[8, 300]));
    let sq = apply(&mut g, Mul, &[x, x]);

    let sum = apply(
        &mut g,
        ReduceSum {
            axes: vec![1],
            accum_dtype: DType::F32,
        },
        &[sq],
    );

    let sum = apply(
        &mut g,
        Reshape {
            new_sizes: vec![8, 1],
        },
        &[sum],
    );

    let n = Full {
        shape: vec![],
        fill_value: Scalar::Float(300.0),
        dtype: DType::F32,
    };

    let n = apply(&mut g, n, &[]);

    let b = BroadcastInDim {
        shape: vec![8, 1],
        broadcast_dimensions: vec![],
    };

    let n = apply(&mut g, b, &[n]);
    let mean = apply(&mut g, Div, &[sum, n]);
    let s = apply(&mut g, Sqrt, &[mean]);

    let b = BroadcastInDim {
        shape: vec![8, 300],
        broadcast_dimensions: vec![0, 1],
    };

    let s = apply(&mut g, b, &[s]);
    let y = apply(&mut g, Div, &[x, s]);
    g.set_outputs(&[y]).unwrap();
    assert_eq!(fused_primitives(&g, &[data(&[8, 300], 1)]), ["fusion"]);

    if available() {
        check(&g, &[values(DType::F32, &[8, 300], 1)]);
    }
}

/// Division by a square root is generated as traced, never rewritten into `rsqrt`.
#[test]
fn division_by_sqrt_is_as_traced() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[4, 8]));
    let y = g.input(ty(DType::F32, &[4, 8]));
    let s = apply(&mut g, Sqrt, &[y]);
    let q = apply(&mut g, Div, &[x, s]);
    g.set_outputs(&[q]).unwrap();
    let fused = fuse(&g);

    let Fusion { body, .. } = &fused.nodes()[0].primitive else {
        panic!("{fused}")
    };

    let source =
        kernel_codegen::kernel(body, &[], &MpsConfig::default(), &Default::default(), None).1;

    assert!(
        !source.contains("rsqrt(") && source.contains("Div::") && source.contains("Sqrt::"),
        "{source}"
    );

    if available() {
        check(
            &g,
            &[data(&[4, 8], 1), Tensor::full(&[4, 8], 2.0, DType::F32)],
        );
    }

    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[4, 8]));
    let m = g.input(ty(DType::F32, &[4, 1]));
    let s = apply(&mut g, Sqrt, &[m]);

    let b = BroadcastInDim {
        shape: vec![4, 8],
        broadcast_dimensions: vec![0, 1],
    };

    let s = apply(&mut g, b, &[s]);
    let q = apply(&mut g, Div, &[x, s]);
    g.set_outputs(&[q]).unwrap();
    let fused = fuse(&g);

    let Fusion { body, .. } = &fused.nodes()[0].primitive else {
        panic!("{fused}")
    };

    let source =
        kernel_codegen::kernel(body, &[], &MpsConfig::default(), &Default::default(), None).1;

    assert!(
        !source.contains("rsqrt(") && source.contains("Div::") && source.contains("Sqrt::"),
        "{source}"
    );

    if available() {
        let m = Tensor::from_slice(&[1.0f32, 4.0, 9.0, 16.0], DType::F32).reshape(&[4, 1]);
        check(&g, &[data(&[4, 8], 1), m]);
    }
}

/// The elementwise ops feeding an RMS norm and the weight transform after it fuse into its row kernel.
#[test]
fn rms_norm_row_kernels_fuse_their_input() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[8, 300]));
    let y = g.input(ty(DType::F32, &[8, 300]));
    let w = g.input(ty(DType::F32, &[300]));
    let h = apply(&mut g, Add, &[x, y]);
    let sq = apply(&mut g, Mul, &[h, h]);

    let sum = apply(
        &mut g,
        ReduceSum {
            axes: vec![1],
            accum_dtype: DType::F32,
        },
        &[sq],
    );

    let sum = apply(
        &mut g,
        Reshape {
            new_sizes: vec![8, 1],
        },
        &[sum],
    );

    let scalar = |g: &mut Graph, v: f64| {
        let c = Full {
            shape: vec![],
            fill_value: Scalar::Float(v),
            dtype: DType::F32,
        };

        let c = apply(g, c, &[]);

        let b = BroadcastInDim {
            shape: vec![8, 1],
            broadcast_dimensions: vec![],
        };

        apply(g, b, &[c])
    };

    let n = scalar(&mut g, 300.0);
    let mean = apply(&mut g, Div, &[sum, n]);
    let s = apply(&mut g, Sqrt, &[mean]);

    let b = BroadcastInDim {
        shape: vec![8, 300],
        broadcast_dimensions: vec![0, 1],
    };

    let s = apply(&mut g, b, &[s]);
    let normed = apply(&mut g, Div, &[h, s]);
    let w2 = apply(&mut g, Exp, &[w]);

    let b = BroadcastInDim {
        shape: vec![8, 300],
        broadcast_dimensions: vec![1],
    };

    let w2 = apply(&mut g, b, &[w2]);
    let out = apply(&mut g, Mul, &[normed, w2]);
    g.set_outputs(&[out]).unwrap();
    let inputs = [data(&[8, 300], 1), data(&[8, 300], 2), data(&[300], 3)];
    let fused = fuse(&g);
    let labels: Vec<_> = fused.nodes().iter().map(|n| n.primitive.name()).collect();
    assert_eq!(labels.len(), 1, "{fused}");

    assert!(
        labels[0].starts_with("add → mul → reduce_sum"),
        "{labels:?}"
    );

    assert_eq!(fused_primitives(&g, &inputs), ["fusion"]);

    if available() {
        let inputs: Vec<Tensor> = [vec![8, 300], vec![8, 300], vec![300]]
            .iter()
            .enumerate()
            .map(|(i, s)| values(DType::F32, s, i as u64 + 1))
            .collect();

        check(&g, &inputs);
    }
}

/// Three dots read by one elementwise chain each get their own kernel rather than sharing an epilogue.
#[test]
fn dots_epilogues_do_not_overlap() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[8, 16]));
    let ws: Vec<Var> = (0..3).map(|_| g.input(ty(DType::F32, &[16, 12]))).collect();

    let dot = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };

    let ys: Vec<Var> = ws
        .iter()
        .map(|&w| apply(&mut g, dot.clone(), &[x, w]))
        .collect();

    let zero = Full {
        shape: vec![8, 12],
        fill_value: Scalar::Float(0.0),
        dtype: DType::F32,
    };

    let zero = apply(&mut g, zero, &[]);
    let r = apply(&mut g, Max, &[ys[0], zero]);
    let m = apply(&mut g, Mul, &[r, ys[1]]);
    let out = apply(&mut g, Add, &[m, ys[2]]);
    g.set_outputs(&[out]).unwrap();

    let inputs = [
        data(&[8, 16], 1),
        data(&[16, 12], 2),
        data(&[16, 12], 3),
        data(&[16, 12], 4),
    ];

    let names = fused_primitives(&g, &inputs);
    assert_eq!(names, ["fusion", "fusion", "fusion"], "{names:?}");
}

/// Batched vector dots are strength-reduced to a multiply and a sum.
#[test]
fn vector_dots_are_reduced() {
    let mut g = Graph::new();
    let a = g.input(ty(DType::F32, &[4, 6, 16]));
    let b = g.input(ty(DType::F32, &[4, 6, 16]));

    let dot = DotGeneral {
        lhs_contracting: vec![2],
        rhs_contracting: vec![2],
        lhs_batch: vec![0, 1],
        rhs_batch: vec![0, 1],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };

    let d = apply(&mut g, dot, &[a, b]);
    g.set_outputs(&[d]).unwrap();
    let reduced = crate::graph_compiler::common::dot_strength::reduce_vector_dots(&g);
    let names: Vec<&str> = reduced.nodes().iter().map(|n| n.primitive.name()).collect();
    assert_eq!(names, ["mul", "reduce_sum"], "{reduced}");
    let inputs = [data(&[4, 6, 16], 1), data(&[4, 6, 16], 2)];
    let want = reference::run(&g, &inputs).unwrap();
    let got = reference::run(&reduced, &inputs).unwrap();
    let (want, got) = (want[0].to_vec::<f32>(), got[0].to_vec::<f32>());

    for (w, g) in want.iter().zip(&got) {
        assert!((w - g).abs() <= 1e-5 * w.abs().max(1.0), "{w} {g}");
    }
}

/// Skinny long-K matmuls split K into chunks, atomically in f32 when allowed, keeping the accumulation dtype;
/// large, short-K or unevenly divisible ones stay whole.
#[test]
fn skinny_dots_split_their_contraction() {
    let build_with = |m: usize, n: usize, k: usize, dtype: DType, accum: DType| {
        let mut g = Graph::new();
        let a = g.input(ty(dtype, &[m, k]));
        let b = g.input(ty(dtype, &[k, n]));

        let dot = DotGeneral {
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
            accum_dtype: accum,
            output_dtype: dtype,
        };

        let d = apply(&mut g, dot, &[a, b]);
        g.set_outputs(&[d]).unwrap();
        g
    };

    let build = |m: usize, n: usize, k: usize, dtype: DType| build_with(m, n, k, dtype, DType::F32);
    let g = build(4, 256, 4096, DType::F32);
    let split = super::split_k::split_k(&g, false, false);
    let names: Vec<&str> = split.nodes().iter().map(|n| n.primitive.name()).collect();

    assert_eq!(
        names,
        ["reshape", "reshape", "dot_general", "reduce_sum"],
        "{split}"
    );

    assert_eq!(split.type_of(split.nodes()[2].output).shape, [8, 4, 256]);
    let inputs = [data(&[4, 4096], 1), data(&[4096, 256], 2)];
    let want = reference::run(&g, &inputs).unwrap();
    let got = reference::run(&split, &inputs).unwrap();
    let (want, got) = (want[0].to_vec::<f32>(), got[0].to_vec::<f32>());

    for (w, g) in want.iter().zip(&got) {
        assert!((w - g).abs() <= 4096.0 * 4.0 * f32::EPSILON, "{w} {g}");
    }

    let split = super::split_k::split_k(&build(4, 256, 4096, DType::BF16), false, false);
    let names: Vec<&str> = split.nodes().iter().map(|n| n.primitive.name()).collect();

    assert_eq!(
        names,
        ["reshape", "reshape", "dot_general", "reduce_sum", "cast"]
    );

    let split = super::split_k::split_k(&g, true, false);
    assert_eq!(split.nodes().len(), 4, "{split}");
    let g = build(352, 352, 2048, DType::BF16);
    let split = super::split_k::split_k(&g, true, false);
    let names: Vec<&str> = split.nodes().iter().map(|n| n.primitive.name()).collect();

    assert_eq!(
        names,
        ["reshape", "reshape", "dot_general (split-K)", "cast"],
        "{split}"
    );

    let Primitive::Fusion { body, .. } = &split.nodes()[2].primitive else {
        unreachable!("a fusion")
    };

    assert!(super::split_k::atomic_dot(body).is_some(), "{body}");
    let g = build_with(352, 352, 2048, DType::BF16, DType::BF16);
    let split = super::split_k::split_k(&g, true, false);
    let names: Vec<&str> = split.nodes().iter().map(|n| n.primitive.name()).collect();

    assert_eq!(
        names,
        ["reshape", "reshape", "dot_general", "reduce_sum"],
        "{split}"
    );

    for node in &split.nodes()[2..] {
        let accum = match &node.primitive {
            DotGeneral { accum_dtype, .. } | ReduceSum { accum_dtype, .. } => *accum_dtype,
            p => unreachable!("{p}"),
        };

        assert_eq!(accum, DType::BF16, "{split}");
        assert_eq!(split.type_of(node.output).dtype, DType::BF16, "{split}");
    }

    let inputs = [
        data(&[352, 2048], 1).to_dtype(DType::BF16).unwrap(),
        data(&[2048, 352], 2).to_dtype(DType::BF16).unwrap(),
    ];

    let run = |g: &Graph| {
        reference::run(g, &inputs).unwrap()[0]
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec::<f32>()
    };

    let exact = run(&build_with(352, 352, 2048, DType::BF16, DType::F32));

    let error = |got: &[f32]| {
        exact
            .iter()
            .zip(got)
            .fold(0f32, |m, (e, v)| m.max((e - v).abs()))
    };

    let (unsplit, split) = (error(&run(&g)), error(&run(&split)));
    assert!(split > 0.0 && split <= unsplit, "{split} {unsplit}");

    for g in [
        build(1024, 1024, 4096, DType::F32),
        build(4, 256, 2049, DType::F32),
        build(4, 256, 1024, DType::F32),
    ] {
        let split = super::split_k::split_k(&g, false, false);
        assert_eq!(split.nodes().len(), 1, "{split}");
    }
}

/// Every tiled, MPP and skinny GEMM config computes plain and bias-fused matmuls like the reference.
#[test]
fn every_gemm_candidate_matches_the_reference() {
    use super::autotune::{Gemms, matmul_of};
    use crate::codegen::dot_general::mps as gemm;
    use crate::graph_compiler::common::autotune::Kernels;

    if !available() {
        return;
    }

    let dot = |lc: usize, rc: usize, out: DType| DotGeneral {
        lhs_contracting: vec![lc],
        rhs_contracting: vec![rc],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: out,
    };

    type Case = (&'static [usize], &'static [usize], usize, usize);

    let cases: [Case; 4] = [
        (&[256, 192], &[192, 320], 1, 0),
        (&[100, 99], &[99, 70], 1, 0),
        (&[128, 96], &[80, 128], 0, 1),
        (&[5, 1000], &[1000, 300], 1, 0),
    ];

    let mut options = crate::graph_compiler::Options::default();
    options.config.mps.autotune = false;

    for dtype in [DType::F32, DType::BF16] {
        for (lhs, rhs, lc, rc) in cases {
            for fused in [false, true] {
                let out = [lhs[1 - lc], rhs[1 - rc]];
                let mut g = Graph::new();
                let a = g.input(ty(dtype, lhs));
                let b = g.input(ty(dtype, rhs));
                let mut d = apply(&mut g, dot(lc, rc, dtype), &[a, b]);
                let mut inputs = vec![values(dtype, lhs, 1), values(dtype, rhs, 2)];

                if fused {
                    let c = g.input(ty(dtype, &out));
                    d = apply(&mut g, Add, &[d, c]);
                    inputs.push(values(dtype, &out, 3));
                }

                g.set_outputs(&[d]).unwrap();

                let expected = reference::run(&g, &inputs).unwrap()[0]
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec::<f32>();

                let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
                let on_mps: Vec<Tensor> = inputs.iter().map(|t| t.to(Device::Mps(0))).collect();

                let mut plan =
                    crate::graph_compiler::compile_with(&g, Device::Mps(0), &options).unwrap();

                let at = plan
                    .steps()
                    .iter()
                    .position(|s| matmul_of(s).is_some())
                    .expect("a matmul step");

                let step = plan.steps()[at].clone();

                let Fusion { body, .. } = &step.primitive else {
                    panic!("every float matmul runs as a GEMM fusion: {plan}");
                };

                assert_eq!(body.nodes().len(), 1 + usize::from(fused), "{body}");
                let (p, l, r, o) = matmul_of(&step).unwrap();
                let layout = gemm::mpp_transposes(p, l, r, o);

                let mpp = match layout.is_some() && super::mpp_available() {
                    true => gemm::mpp_configs(),
                    false => &[],
                };

                let skinny = match layout {
                    Some((_, false)) => gemm::skinny_configs(),
                    _ => &[],
                };

                let mut gemms = Gemms::default();

                let built: Vec<_> = gemm::configs()
                    .iter()
                    .chain(mpp)
                    .chain(skinny)
                    .map(|&config| (config, gemms.build(&step, config)))
                    .collect();

                gemms.compile().unwrap();

                for (config, tuned) in built {
                    plan.steps_mut()[at].tuned = Some(tuned);

                    let got = plan.run(&on_mps).unwrap()[0]
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec::<f32>();

                    let tol = scale * if dtype == DType::F32 { 1e-5 } else { 1e-2 };

                    for (k, (w, v)) in expected.iter().zip(&got).enumerate() {
                        assert!(
                            (w - v).abs() <= tol,
                            "{config:?} {dtype} {lhs:?}·{rhs:?} fused={fused} [{k}]: {v} != {w}"
                        );
                    }
                }
            }
        }
    }
}

/// Every tiled GEMM config stores column regions, row halves and concatenated expansions correctly.
#[test]
fn every_tiled_gemm_stores_regions_halves_and_expansions_along_rows_and_columns() {
    use super::autotune::{Gemms, matmul_of};
    use crate::codegen::dot_general::mps as gemm;
    use crate::graph_compiler::common::autotune::Kernels;

    if !available() {
        return;
    }

    let matmul = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };

    let (pairs, plain, other, k) = (40, 24, 69, 96);
    let extent = 2 * pairs + plain;
    let mut options = crate::graph_compiler::Options::default();
    options.config.mps.autotune = false;

    let along = |g: &mut Graph, v: Var, shape: &[usize], at: usize, start: usize, end: usize| {
        let mut limit = shape.to_vec();
        let mut first = vec![0; shape.len()];
        (first[at], limit[at]) = (start, end);

        let s = Slice {
            start_indices: first,
            limit_indices: limit,
            view: false,
        };

        apply(g, s, &[v])
    };

    let reshape = |g: &mut Graph, v: Var, sizes: &[usize]| {
        let r = Reshape {
            new_sizes: sizes.to_vec(),
        };

        apply(g, r, &[v])
    };

    let regions = |axis: usize| {
        let sized = move |along: usize, rest: usize| match axis {
            0 => vec![along, rest],
            _ => vec![rest, along],
        };

        let out = sized(extent, other);
        let mut g = Graph::new();
        let a = g.input(ty(DType::F32, &[out[0], k]));
        let b = g.input(ty(DType::F32, &[k, out[1]]));
        let d = apply(&mut g, matmul.clone(), &[a, b]);
        let group = along(&mut g, d, &out, axis, 0, 2 * pairs);

        let tuples: Vec<usize> = match axis {
            0 => vec![pairs, 2, other],
            _ => vec![other, pairs, 2],
        };

        let tupled = reshape(&mut g, group, &tuples);

        let mut member = |t: usize| {
            let s = along(&mut g, tupled, &tuples, axis + 1, t, t + 1);
            reshape(&mut g, s, &sized(pairs, other))
        };

        let (first, second) = (member(0), member(1));
        let gated = apply(&mut g, Mul, &[first, second]);
        let rest = along(&mut g, d, &out, axis, 2 * pairs, extent);
        let negated = apply(&mut g, Neg, &[rest]);
        g.set_outputs(&[gated, negated]).unwrap();
        g
    };

    let (half, rows) = (45, 50);

    let halves = {
        let mut g = Graph::new();
        let a = g.input(ty(DType::F32, &[2 * half, k]));
        let b = g.input(ty(DType::F32, &[k, other]));
        let d = apply(&mut g, matmul.clone(), &[a, b]);
        let shape = [2 * half, other];
        let first = along(&mut g, d, &shape, 0, 0, half);
        let second = along(&mut g, d, &shape, 0, half, 2 * half);
        let gated = apply(&mut g, Mul, &[first, second]);
        g.set_outputs(&[gated]).unwrap();
        g
    };

    let expanded = |interleaved: bool| {
        let mut g = Graph::new();
        let a = g.input(ty(DType::F32, &[rows, k]));
        let b = g.input(ty(DType::F32, &[k, other]));
        let d = apply(&mut g, matmul.clone(), &[a, b]);
        let squared = apply(&mut g, Mul, &[d, d]);
        let negated = apply(&mut g, Neg, &[d]);

        let both = match interleaved {
            true => {
                let parts = [squared, negated].map(|v| reshape(&mut g, v, &[rows, 1, other]));

                let concat = Concatenate {
                    dimension: 1,
                    in_place: false,
                };

                let pairs = apply(&mut g, concat, &parts);
                reshape(&mut g, pairs, &[2 * rows, other])
            }
            false => {
                let concat = Concatenate {
                    dimension: 0,
                    in_place: false,
                };

                apply(&mut g, concat, &[squared, negated])
            }
        };

        g.set_outputs(&[both]).unwrap();
        g
    };

    let cases = [
        ("regions along rows", regions(0)),
        ("regions along columns", regions(1)),
        ("row halves", halves),
        ("rows expanded side by side", expanded(false)),
        ("rows expanded interleaved", expanded(true)),
    ];

    for (case, g) in cases {
        let inputs: Vec<Tensor> = g
            .inputs()
            .iter()
            .enumerate()
            .map(|(seed, &v)| values(DType::F32, &g.type_of(v).shape, seed as u64 + 1))
            .collect();

        let expected: Vec<Vec<f32>> = reference::run(&g, &inputs)
            .unwrap()
            .iter()
            .map(|t| t.to_vec::<f32>())
            .collect();

        let on_mps: Vec<Tensor> = inputs.iter().map(|t| t.to(Device::Mps(0))).collect();
        let mut plan = crate::graph_compiler::compile_with(&g, Device::Mps(0), &options).unwrap();
        assert_eq!(plan.steps().len(), 1, "{case}: {plan}");
        let step = plan.steps()[0].clone();
        assert!(matmul_of(&step).is_some(), "{case}: {plan}");
        let mut gemms = Gemms::default();

        let built: Vec<_> = gemm::configs()
            .iter()
            .map(|&config| (config, gemms.build(&step, config)))
            .collect();

        gemms.compile().unwrap();

        for (config, tuned) in built {
            plan.steps_mut()[0].tuned = Some(tuned);
            let got = plan.run(&on_mps).unwrap();

            for (want, have) in expected.iter().zip(&got) {
                let have = have.to_vec::<f32>();
                let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));

                for (e, (w, v)) in want.iter().zip(&have).enumerate() {
                    assert!(
                        (w - v).abs() <= scale * 1e-5,
                        "{case} {config:?} [{e}]: {v} != {w}"
                    );
                }
            }
        }
    }
}

/// Every MPP config handles gated pairs (halves or interleaved) and an RMS norm wider than one tile.
#[test]
fn every_tuned_paired_and_wide_norm_gemm_matches_the_reference() {
    use super::autotune::{Gemms, matmul_of};
    use crate::codegen::dot_general::mps::Algorithm;
    use crate::graph_compiler::common::autotune::Kernels;

    if !available() || !super::mpp_available() {
        return;
    }

    let (m, h) = (100, 768);

    let projection = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![1],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };

    let slice = |g: &mut Graph, v: Var, start_indices: Vec<usize>, limit_indices: Vec<usize>| {
        let s = Slice {
            start_indices,
            limit_indices,
            view: false,
        };

        apply(g, s, &[v])
    };

    let reshape = |g: &mut Graph, v: Var, sizes: &[usize]| {
        let r = Reshape {
            new_sizes: sizes.to_vec(),
        };

        apply(g, r, &[v])
    };

    let full = |g: &mut Graph, value: f64| {
        let f = Full {
            shape: vec![m],
            fill_value: Scalar::Float(value),
            dtype: DType::F32,
        };

        apply(g, f, &[])
    };

    let pair = |k: usize, interleaved: bool| {
        let mut g = Graph::new();
        let a = g.input(ty(DType::BF16, &[m, k]));
        let w = g.input(ty(DType::BF16, &[2 * h, k]));
        let d = apply(&mut g, projection.clone(), &[a, w]);

        let (first, second) = match interleaved {
            true => {
                let pairs = reshape(&mut g, d, &[m, h, 2]);

                let mut member = |t: usize| {
                    let s = slice(&mut g, pairs, vec![0, 0, t], vec![m, h, t + 1]);
                    reshape(&mut g, s, &[m, h])
                };

                (member(0), member(1))
            }
            false => (
                slice(&mut g, d, vec![0, 0], vec![m, h]),
                slice(&mut g, d, vec![0, h], vec![m, 2 * h]),
            ),
        };

        let gate = apply(&mut g, Logistic, &[first]);
        let silu = apply(&mut g, Mul, &[first, gate]);
        let z = apply(&mut g, Mul, &[silu, second]);

        let out = apply(
            &mut g,
            Cast {
                new_dtype: DType::BF16,
            },
            &[z],
        );

        g.set_outputs(&[out]).unwrap();
        g
    };

    let norm = |k: usize| {
        let mut g = Graph::new();
        let a = g.input(ty(DType::BF16, &[m, k]));
        let w = g.input(ty(DType::BF16, &[h, k]));
        let r = g.input(ty(DType::F32, &[m, h]));
        let gamma = g.input(ty(DType::F32, &[h]));
        let d = apply(&mut g, projection.clone(), &[a, w]);
        let x = apply(&mut g, Add, &[r, d]);
        let squares = apply(&mut g, Mul, &[x, x]);

        let sum = ReduceSum {
            axes: vec![1],
            accum_dtype: DType::F32,
        };

        let s = apply(&mut g, sum, &[squares]);
        let inverse = full(&mut g, 1.0 / h as f64);
        let mean = apply(&mut g, Mul, &[s, inverse]);
        let eps = full(&mut g, 1e-5);
        let variance = apply(&mut g, Add, &[mean, eps]);
        let rms = apply(&mut g, Sqrt, &[variance]);
        let column = reshape(&mut g, rms, &[m, 1]);

        let rows = BroadcastInDim {
            shape: vec![m, h],
            broadcast_dimensions: vec![0, 1],
        };

        let rms = apply(&mut g, rows, &[column]);
        let y = apply(&mut g, Div, &[x, rms]);

        let columns = BroadcastInDim {
            shape: vec![m, h],
            broadcast_dimensions: vec![1],
        };

        let gamma = apply(&mut g, columns, &[gamma]);
        let out = apply(&mut g, Mul, &[y, gamma]);
        g.set_outputs(&[out, x]).unwrap();
        g
    };

    let mut options = crate::graph_compiler::Options::default();
    options.config.mps.autotune = false;

    for k in [96, 256] {
        let cases = [
            ("interleaved pair", pair(k, true)),
            ("halves pair", pair(k, false)),
            ("norm wider than a tile", norm(k)),
        ];

        for (case, g) in cases {
            let inputs: Vec<Tensor> = g
                .inputs()
                .iter()
                .enumerate()
                .map(|(seed, &v)| {
                    let t = g.type_of(v);
                    values(t.dtype, &t.shape, seed as u64 + 1)
                })
                .collect();

            let expected: Vec<Vec<f32>> = reference::run(&g, &inputs)
                .unwrap()
                .iter()
                .map(|t| t.to_dtype(DType::F32).unwrap().to_vec::<f32>())
                .collect();

            let on_mps: Vec<Tensor> = inputs.iter().map(|t| t.to(Device::Mps(0))).collect();

            let mut plan =
                crate::graph_compiler::compile_with(&g, Device::Mps(0), &options).unwrap();

            let at = plan
                .steps()
                .iter()
                .position(|s| matmul_of(s).is_some())
                .expect("a matmul step");

            let step = plan.steps()[at].clone();
            let mut gemms = Gemms::new(&MpsConfig::default());

            let configs: Vec<_> = gemms
                .configs(&step)
                .into_iter()
                .filter(|c| c.algorithm == Algorithm::Mpp)
                .collect();

            assert!(!configs.is_empty(), "{case} K={k}: no MPP candidate");

            let built: Vec<_> = configs
                .into_iter()
                .map(|config| (config, gemms.build(&step, config)))
                .collect();

            gemms.compile().unwrap();

            for (config, tuned) in built {
                plan.steps_mut()[at].tuned = Some(tuned);
                let got = plan.run(&on_mps).unwrap();

                for (want, have) in expected.iter().zip(&got) {
                    let have = have.to_dtype(DType::F32).unwrap().to_vec::<f32>();
                    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));

                    for (e, (w, v)) in want.iter().zip(&have).enumerate() {
                        assert!(
                            (w - v).abs() <= scale * 1e-2,
                            "{case} K={k} {config:?} [{e}]: {v} != {w}"
                        );
                    }
                }
            }
        }
    }
}

/// MPP and skinny GEMM candidates pick the expected epilogue layout flags and store every epilogue correctly.
#[test]
fn every_tuned_gemm_kernel_stores_every_epilogue() {
    use super::autotune::{Gemms, matmul_of};
    use crate::codegen::dot_general::mps::{Algorithm, GemmConfig};
    use crate::graph_compiler::common::autotune::Kernels;

    if !available() || !super::mpp_available() {
        return;
    }

    let matmul = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };

    let slice = |g: &mut Graph, v: Var, shape: &[usize], at: usize, start: usize, end: usize| {
        let mut limit = shape.to_vec();
        let mut first = vec![0; shape.len()];
        (first[at], limit[at]) = (start, end);

        let s = Slice {
            start_indices: first,
            limit_indices: limit,
            view: false,
        };

        apply(g, s, &[v])
    };

    let reshape = |g: &mut Graph, v: Var, sizes: &[usize]| {
        let r = Reshape {
            new_sizes: sizes.to_vec(),
        };

        apply(g, r, &[v])
    };

    let dot = |g: &mut Graph, rows: usize, k: usize, columns: usize| {
        let a = g.input(ty(DType::F32, &[rows, k]));
        let b = g.input(ty(DType::F32, &[k, columns]));
        apply(g, matmul.clone(), &[a, b])
    };

    let regions = |axis: usize, pairs: usize, plain: usize, other: usize, k: usize| {
        let extent = 2 * pairs + plain;

        let sized = |along: usize, rest: usize| match axis {
            0 => vec![along, rest],
            _ => vec![rest, along],
        };

        let out = sized(extent, other);
        let mut g = Graph::new();
        let d = dot(&mut g, out[0], k, out[1]);
        let group = slice(&mut g, d, &out, axis, 0, 2 * pairs);

        let tuples: Vec<usize> = match axis {
            0 => vec![pairs, 2, other],
            _ => vec![other, pairs, 2],
        };

        let tupled = reshape(&mut g, group, &tuples);

        let mut member = |t: usize| {
            let s = slice(&mut g, tupled, &tuples, axis + 1, t, t + 1);
            reshape(&mut g, s, &sized(pairs, other))
        };

        let (first, second) = (member(0), member(1));
        let gated = apply(&mut g, Mul, &[first, second]);
        let rest = slice(&mut g, d, &out, axis, 2 * pairs, extent);
        let negated = apply(&mut g, Neg, &[rest]);
        g.set_outputs(&[gated, negated]).unwrap();
        g
    };

    let row_halves = |half: usize, other: usize, k: usize| {
        let mut g = Graph::new();
        let d = dot(&mut g, 2 * half, k, other);
        let shape = [2 * half, other];
        let first = slice(&mut g, d, &shape, 0, 0, half);
        let second = slice(&mut g, d, &shape, 0, half, 2 * half);
        let gated = apply(&mut g, Mul, &[first, second]);
        g.set_outputs(&[gated]).unwrap();
        g
    };

    let pair = |rows: usize, half: usize, k: usize, interleaved: bool| {
        let mut g = Graph::new();
        let d = dot(&mut g, rows, k, 2 * half);

        let (first, second) = match interleaved {
            true => {
                let tuples = [rows, half, 2];
                let pairs = reshape(&mut g, d, &tuples);

                let mut member = |t: usize| {
                    let s = slice(&mut g, pairs, &tuples, 2, t, t + 1);
                    reshape(&mut g, s, &[rows, half])
                };

                (member(0), member(1))
            }
            false => {
                let shape = [rows, 2 * half];

                (
                    slice(&mut g, d, &shape, 1, 0, half),
                    slice(&mut g, d, &shape, 1, half, 2 * half),
                )
            }
        };

        let gate = apply(&mut g, Logistic, &[first]);
        let gated = apply(&mut g, Mul, &[gate, second]);
        g.set_outputs(&[gated]).unwrap();
        g
    };

    let expanded = |rows: usize, other: usize, k: usize, along: usize| {
        let mut g = Graph::new();
        let d = dot(&mut g, rows, k, other);
        let squared = apply(&mut g, Mul, &[d, d]);
        let negated = apply(&mut g, Neg, &[d]);

        let concat = |dimension: usize| Concatenate {
            dimension,
            in_place: false,
        };

        let both = match along {
            0 => apply(&mut g, concat(0), &[squared, negated]),
            1 => {
                let parts = [squared, negated].map(|v| reshape(&mut g, v, &[rows, 1, other]));
                let pairs = apply(&mut g, concat(1), &parts);
                reshape(&mut g, pairs, &[2 * rows, other])
            }
            _ => {
                let parts = [squared, negated].map(|v| reshape(&mut g, v, &[rows, other, 1]));
                let pairs = apply(&mut g, concat(2), &parts);
                reshape(&mut g, pairs, &[rows, 2 * other])
            }
        };

        g.set_outputs(&[both]).unwrap();
        g
    };

    let flags = |expanded, interleaved, group, rows| gemm::EpilogueFlags {
        expanded,
        interleaved,
        group,
        rows,
    };

    let large = [
        (
            "regions along rows",
            regions(0, 320, 64, 256, 96),
            flags(false, true, 8, true),
        ),
        (
            "regions along columns",
            regions(1, 320, 64, 256, 96),
            flags(false, true, 8, false),
        ),
        (
            "row halves",
            row_halves(256, 300, 96),
            flags(false, false, 8, true),
        ),
        (
            "column halves",
            pair(256, 320, 96, false),
            flags(false, false, 2, false),
        ),
        (
            "interleaved pairs",
            pair(256, 320, 96, true),
            flags(false, true, 2, false),
        ),
        (
            "expanded side by side",
            expanded(200, 384, 96, 0),
            flags(true, false, 1, true),
        ),
        (
            "expanded row pairs",
            expanded(200, 384, 96, 1),
            flags(true, false, 1, false),
        ),
        (
            "expanded column pairs",
            expanded(200, 384, 96, 2),
            flags(true, true, 1, false),
        ),
    ];

    let skinny = [
        (
            "regions along rows",
            regions(0, 8, 8, 1024, 1024),
            flags(false, true, 8, true),
        ),
        (
            "regions along columns",
            regions(1, 256, 512, 24, 1024),
            flags(false, true, 8, false),
        ),
        (
            "row halves",
            row_halves(12, 1024, 1024),
            flags(false, false, 8, true),
        ),
        (
            "column halves",
            pair(24, 512, 1024, false),
            flags(false, false, 2, false),
        ),
        (
            "interleaved pairs",
            pair(24, 512, 1024, true),
            flags(false, true, 2, false),
        ),
        (
            "expanded side by side",
            expanded(24, 1024, 1024, 0),
            flags(true, false, 1, true),
        ),
        (
            "expanded row pairs",
            expanded(24, 1024, 1024, 1),
            flags(true, false, 1, false),
        ),
        (
            "expanded column pairs",
            expanded(24, 1024, 1024, 2),
            flags(true, true, 1, false),
        ),
    ];

    let mut options = crate::graph_compiler::Options::default();
    options.config.mps.autotune = false;
    options.config.mps.split_k = false;

    for (algorithm, cases) in [(Algorithm::Mpp, large), (Algorithm::Skinny, skinny)] {
        let tried = |c: &GemmConfig| c.algorithm == algorithm;

        for (case, g, want) in cases {
            let inputs: Vec<Tensor> = g
                .inputs()
                .iter()
                .enumerate()
                .map(|(seed, &v)| values(DType::F32, &g.type_of(v).shape, seed as u64 + 1))
                .collect();

            let expected: Vec<Vec<f32>> = reference::run(&g, &inputs)
                .unwrap()
                .iter()
                .map(|t| t.to_vec::<f32>())
                .collect();

            let on_mps: Vec<Tensor> = inputs.iter().map(|t| t.to(Device::Mps(0))).collect();

            let mut plan =
                crate::graph_compiler::compile_with(&g, Device::Mps(0), &options).unwrap();

            let at = plan
                .steps()
                .iter()
                .position(|s| matmul_of(s).is_some())
                .unwrap_or_else(|| panic!("{case}: a matmul step in {plan}"));

            let step = plan.steps()[at].clone();

            let Fusion { body, .. } = &step.primitive else {
                panic!("{case}: a GEMM fusion: {plan}");
            };

            let got = gemm::gemm_flags(body, gemm_dot(body).unwrap());
            assert_eq!(got, want, "{case} {algorithm:?}: {body}");
            let mut gemms = Gemms::new(&MpsConfig::default());

            let configs: Vec<_> = gemms
                .configs(&step)
                .into_iter()
                .filter(|c| c.algorithm != Algorithm::Tiled)
                .fold(Vec::<GemmConfig>::new(), |mut kept, c| {
                    let shape = |k: &GemmConfig| (k.algorithm, k.bm, k.bn);

                    if !kept.iter().any(|k| shape(k) == shape(&c)) {
                        kept.push(c);
                    }

                    kept
                });

            assert!(
                configs.iter().any(tried),
                "{case}: no {algorithm:?} candidate: {} configs, output {:?}, views {:?}",
                gemms.configs(&step).len(),
                step.output.1,
                step.views
            );

            let built: Vec<_> = configs
                .into_iter()
                .map(|config| (config, gemms.build(&step, config)))
                .collect();

            gemms.compile().unwrap();

            for (config, tuned) in built {
                plan.steps_mut()[at].tuned = Some(tuned);
                let got = plan.run(&on_mps).unwrap();

                for (want, have) in expected.iter().zip(&got) {
                    let have = have.to_vec::<f32>();
                    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));

                    for (e, (w, v)) in want.iter().zip(&have).enumerate() {
                        assert!(
                            (w - v).abs() <= scale * 1e-4,
                            "{case} {config:?} [{e}]: {v} != {w}"
                        );
                    }
                }
            }
        }
    }
}

/// Every GEMM candidate accumulates in the dot's accumulation dtype, checked by summing 2^16 ones.
#[test]
fn every_gemm_candidate_accumulates_in_the_accum_dtype() {
    use super::autotune::{Gemms, matmul_of};
    use crate::graph_compiler::common::autotune::Kernels;

    if !available() {
        return;
    }

    let mut options = crate::graph_compiler::Options::default();
    options.config.mps.autotune = false;
    options.config.mps.split_k = false;
    let k = 1 << 16;

    for accum in [DType::F32, DType::BF16] {
        for (m, n) in [(4, 1024), (256, 256)] {
            let mut g = Graph::new();
            let a = g.input(ty(DType::BF16, &[m, k]));
            let b = g.input(ty(DType::BF16, &[k, n]));

            let dot = DotGeneral {
                lhs_contracting: vec![1],
                rhs_contracting: vec![0],
                lhs_batch: vec![],
                rhs_batch: vec![],
                accum_dtype: accum,
                output_dtype: accum,
            };

            let d = apply(&mut g, dot, &[a, b]);
            g.set_outputs(&[d]).unwrap();

            let ones = |s: &[usize]| {
                let options = crate::TensorOptions::new()
                    .dtype(DType::BF16)
                    .device(Device::Mps(0));

                Tensor::ones(s, options)
            };

            let on_mps = [ones(&[m, k]), ones(&[k, n])];

            let mut plan =
                crate::graph_compiler::compile_with(&g, Device::Mps(0), &options).unwrap();

            let at = plan
                .steps()
                .iter()
                .position(|s| matmul_of(s).is_some())
                .expect("a matmul step");

            let step = plan.steps()[at].clone();
            let mut gemms = Gemms::new(&MpsConfig::default());

            let built: Vec<_> = gemms
                .configs(&step)
                .into_iter()
                .map(|config| (config, gemms.build(&step, config)))
                .collect();

            assert!(built.len() > 1, "{accum} {m}x{n}: tuned candidates");
            gemms.compile().unwrap();

            for (config, tuned) in built {
                plan.steps_mut()[at].tuned = Some(tuned);
                let got = &plan.run(&on_mps).unwrap()[0];
                assert_eq!(got.dtype(), accum, "{config:?}");
                let sum = got.to_dtype(DType::F32).unwrap().to_vec::<f32>()[0];

                match accum {
                    DType::F32 => assert_eq!(sum, k as f32, "{config:?}"),
                    _ => assert!(sum < k as f32, "{config:?}: {sum} of {k} ones in {accum}"),
                }
            }
        }
    }
}

/// A dot written as multiply and sum, feeding an RMS norm over other rows, stays its own kernel: no fusion holds two sums.
#[test]
fn a_reduction_feeding_a_norm_of_other_rows_is_its_own_kernel() {
    if !available() {
        return;
    }

    let (n, k) = (64, 128);
    let mut g = Graph::new();
    let w = g.input(ty(DType::BF16, &[n, k]));
    let x = g.input(ty(DType::F32, &[k]));
    let r = g.input(ty(DType::BF16, &[1, n]));
    let gamma = g.input(ty(DType::BF16, &[n]));
    let cast = |g: &mut Graph, v: Var, new_dtype: DType| apply(g, Cast { new_dtype }, &[v]);

    let broadcast = |g: &mut Graph, v: Var, shape: &[usize], dims: Vec<usize>| {
        let b = BroadcastInDim {
            shape: shape.to_vec(),
            broadcast_dimensions: dims,
        };

        apply(g, b, &[v])
    };

    let sum = |g: &mut Graph, v: Var| {
        let s = ReduceSum {
            axes: vec![1],
            accum_dtype: DType::F32,
        };

        apply(g, s, &[v])
    };

    let xs = broadcast(&mut g, x, &[n, k], vec![1]);
    let wf = cast(&mut g, w, DType::F32);
    let products = apply(&mut g, Mul, &[xs, wf]);
    let dot = sum(&mut g, products);

    let row = apply(
        &mut g,
        Reshape {
            new_sizes: vec![1, n],
        },
        &[dot],
    );

    let rounded = cast(&mut g, row, DType::BF16);
    let y = apply(&mut g, Add, &[r, rounded]);
    let yf = cast(&mut g, y, DType::F32);
    let squares = apply(&mut g, Mul, &[yf, yf]);
    let total = sum(&mut g, squares);

    let inverse = apply(
        &mut g,
        Full {
            shape: vec![],
            fill_value: Scalar::Float(1.0 / n as f64),
            dtype: DType::F32,
        },
        &[],
    );

    let inverse = broadcast(&mut g, inverse, &[1], vec![]);
    let mean = apply(&mut g, Mul, &[total, inverse]);
    let rms = apply(&mut g, Sqrt, &[mean]);

    let rms = apply(
        &mut g,
        Reshape {
            new_sizes: vec![1, 1],
        },
        &[rms],
    );

    let rms = broadcast(&mut g, rms, &[1, n], vec![0, 1]);
    let normed = apply(&mut g, Div, &[yf, rms]);
    let normed = cast(&mut g, normed, DType::BF16);
    let gamma = broadcast(&mut g, gamma, &[1, n], vec![1]);
    let out = apply(&mut g, Mul, &[normed, gamma]);
    g.set_outputs(&[out]).unwrap();

    let inputs = [
        values(DType::BF16, &[n, k], 1),
        values(DType::F32, &[k], 2),
        values(DType::BF16, &[1, n], 3),
        values(DType::BF16, &[n], 4),
    ];

    let expected = reference::run(&g, &inputs).unwrap()[0]
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec::<f32>();

    let plan = crate::graph_compiler::compile(&g, Device::Mps(0)).unwrap();

    for step in plan.steps() {
        if let Fusion { body, .. } = &step.primitive {
            let reductions = body
                .nodes()
                .iter()
                .filter(|n| matches!(n.primitive, ReduceSum { .. }))
                .count();

            assert!(reductions <= 1, "{plan}");
        }
    }

    let on_mps: Vec<Tensor> = inputs.iter().map(|t| t.to(Device::Mps(0))).collect();

    let got = plan.run(&on_mps).unwrap()[0]
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec::<f32>();

    for (w, v) in expected.iter().zip(&got) {
        assert!((w - v).abs() <= 2e-2 * w.abs().max(1.0), "{v} != {w}");
    }
}

/// Every GEMV config stores plain, row-half and row-region epilogues for either vector side, bit-identical to the default.
#[test]
fn every_gemv_candidate_stores_every_epilogue() {
    use super::autotune::{Gemms, matmul_of};
    use crate::codegen::dot_general::mps::Algorithm;
    use crate::graph_compiler::common::autotune::Kernels;

    if !available() {
        return;
    }

    let slice = |g: &mut Graph, v: Var, shape: &[usize], at: usize, start: usize, end: usize| {
        let mut limit = shape.to_vec();
        let mut first = vec![0; shape.len()];
        (first[at], limit[at]) = (start, end);

        let s = Slice {
            start_indices: first,
            limit_indices: limit,
            view: false,
        };

        apply(g, s, &[v])
    };

    let reshape = |g: &mut Graph, v: Var, sizes: &[usize]| {
        let r = Reshape {
            new_sizes: sizes.to_vec(),
        };

        apply(g, r, &[v])
    };

    let dot = |g: &mut Graph, dtype: DType, m: usize, k: usize, n: usize| {
        let vector_lhs = m <= 8;
        let a = g.input(ty(dtype, &[m, k]));

        let b = match vector_lhs {
            true => g.input(ty(dtype, &[n, k])),
            false => g.input(ty(dtype, &[k, n])),
        };

        let product = DotGeneral {
            lhs_contracting: vec![1],
            rhs_contracting: vec![usize::from(vector_lhs)],
            lhs_batch: vec![],
            rhs_batch: vec![],
            accum_dtype: DType::F32,
            output_dtype: DType::F32,
        };

        apply(g, product, &[a, b])
    };

    let plain = |dtype: DType, m: usize, k: usize, n: usize| {
        let mut g = Graph::new();
        let d = dot(&mut g, dtype, m, k, n);
        let r = g.input(ty(DType::F32, &[m, n]));
        let out = apply(&mut g, Add, &[d, r]);
        g.set_outputs(&[out]).unwrap();
        g
    };

    let pair = |dtype: DType, m: usize, k: usize, n: usize, interleaved: bool| {
        let mut g = Graph::new();
        let d = dot(&mut g, dtype, m, k, n);
        let half = n / 2;

        let (first, second) = match interleaved {
            true => {
                let tuples = [m, half, 2];
                let pairs = reshape(&mut g, d, &tuples);

                let mut member = |t: usize| {
                    let s = slice(&mut g, pairs, &tuples, 2, t, t + 1);
                    reshape(&mut g, s, &[m, half])
                };

                (member(0), member(1))
            }
            false => (
                slice(&mut g, d, &[m, n], 1, 0, half),
                slice(&mut g, d, &[m, n], 1, half, n),
            ),
        };

        let gate = apply(&mut g, Logistic, &[first]);
        let gated = apply(&mut g, Mul, &[gate, second]);
        g.set_outputs(&[gated]).unwrap();
        g
    };

    let row_halves = |m: usize, k: usize, n: usize| {
        let mut g = Graph::new();
        let d = dot(&mut g, DType::F32, m, k, n);
        let first = slice(&mut g, d, &[m, n], 0, 0, m / 2);
        let second = slice(&mut g, d, &[m, n], 0, m / 2, m);
        let gated = apply(&mut g, Mul, &[first, second]);
        g.set_outputs(&[gated]).unwrap();
        g
    };

    let regions = |axis: usize, m: usize, k: usize, n: usize, pairs: usize| {
        let mut g = Graph::new();
        let d = dot(&mut g, DType::F32, m, k, n);
        let (out, extent) = ([m, n], [m, n][axis]);
        let other = [m, n][1 - axis];

        let sized = |along: usize| match axis {
            0 => vec![along, other],
            _ => vec![other, along],
        };

        let group = slice(&mut g, d, &out, axis, 0, 2 * pairs);

        let tuples: Vec<usize> = match axis {
            0 => vec![pairs, 2, other],
            _ => vec![other, pairs, 2],
        };

        let tupled = reshape(&mut g, group, &tuples);

        let mut member = |t: usize| {
            let s = slice(&mut g, tupled, &tuples, axis + 1, t, t + 1);
            reshape(&mut g, s, &sized(pairs))
        };

        let (first, second) = (member(0), member(1));
        let gated = apply(&mut g, Mul, &[first, second]);
        let mut outputs = vec![gated];

        if 2 * pairs < extent {
            let rest = slice(&mut g, d, &out, axis, 2 * pairs, extent);
            outputs.push(apply(&mut g, Neg, &[rest]));
        }

        g.set_outputs(&outputs).unwrap();
        g
    };

    let expanded = |m: usize, k: usize, n: usize, along: usize| {
        let mut g = Graph::new();
        let d = dot(&mut g, DType::F32, m, k, n);
        let squared = apply(&mut g, Mul, &[d, d]);
        let negated = apply(&mut g, Neg, &[d]);

        let concat = |dimension: usize| Concatenate {
            dimension,
            in_place: false,
        };

        let both = match along {
            0 => apply(&mut g, concat(0), &[squared, negated]),
            1 => {
                let parts = [squared, negated].map(|v| reshape(&mut g, v, &[m, 1, n]));
                let pairs = apply(&mut g, concat(1), &parts);
                reshape(&mut g, pairs, &[2 * m, n])
            }
            _ => {
                let parts = [squared, negated].map(|v| reshape(&mut g, v, &[m, n, 1]));
                let pairs = apply(&mut g, concat(2), &parts);
                reshape(&mut g, pairs, &[m, 2 * n])
            }
        };

        g.set_outputs(&[both]).unwrap();
        g
    };

    let cases: Vec<(&str, Graph, f32)> = vec![
        ("vector lhs, plain", plain(DType::F32, 3, 520, 72), 1e-4),
        (
            "vector lhs, plain bf16",
            plain(DType::BF16, 1, 2048, 256),
            1e-2,
        ),
        (
            "vector lhs, interleaved pairs",
            pair(DType::F32, 2, 256, 128, true),
            1e-4,
        ),
        (
            "vector lhs, column halves",
            pair(DType::F32, 1, 256, 128, false),
            1e-4,
        ),
        (
            "vector lhs, column halves bf16",
            pair(DType::BF16, 4, 512, 256, false),
            1e-2,
        ),
        ("vector lhs, row halves", row_halves(8, 256, 64), 1e-4),
        (
            "vector lhs, column regions",
            regions(1, 2, 256, 96, 32),
            1e-4,
        ),
        ("vector lhs, row regions", regions(0, 8, 256, 64, 4), 1e-4),
        (
            "vector lhs, expanded side by side",
            expanded(2, 256, 64, 0),
            1e-4,
        ),
        (
            "vector lhs, expanded row pairs",
            expanded(2, 256, 64, 1),
            1e-4,
        ),
        (
            "vector lhs, expanded column pairs",
            expanded(2, 256, 64, 2),
            1e-4,
        ),
        ("matrix lhs, plain", plain(DType::F32, 72, 520, 3), 1e-4),
        (
            "matrix lhs, column halves",
            pair(DType::F32, 64, 256, 8, false),
            1e-4,
        ),
        (
            "matrix lhs, interleaved pairs",
            pair(DType::F32, 64, 256, 8, true),
            1e-4,
        ),
        ("matrix lhs, row halves", row_halves(96, 256, 4), 1e-4),
        ("matrix lhs, row regions", regions(0, 96, 256, 4, 32), 1e-4),
        (
            "matrix lhs, expanded side by side",
            expanded(64, 256, 4, 0),
            1e-4,
        ),
        (
            "matrix lhs, expanded column pairs",
            expanded(64, 256, 4, 2),
            1e-4,
        ),
    ];

    let mut options = crate::graph_compiler::Options::default();
    options.config.mps.autotune = false;

    for (case, g, tol) in cases {
        let inputs: Vec<Tensor> = g
            .inputs()
            .iter()
            .enumerate()
            .map(|(seed, &v)| {
                let t = g.type_of(v);
                values(t.dtype, &t.shape, seed as u64 + 1)
            })
            .collect();

        let expected: Vec<Vec<f32>> = reference::run(&g, &inputs)
            .unwrap()
            .iter()
            .map(|t| t.to_dtype(DType::F32).unwrap().to_vec::<f32>())
            .collect();

        let on_mps: Vec<Tensor> = inputs.iter().map(|t| t.to(Device::Mps(0))).collect();
        let mut plan = crate::graph_compiler::compile_with(&g, Device::Mps(0), &options).unwrap();

        let at = plan
            .steps()
            .iter()
            .position(|s| matmul_of(s).is_some())
            .unwrap_or_else(|| panic!("{case}: a matmul step in {plan}"));

        let untuned = plan.run(&on_mps).unwrap();
        let step = plan.steps()[at].clone();
        let mut gemms = Gemms::new(&MpsConfig::default());
        let configs = gemms.configs(&step);

        assert!(
            !configs.is_empty() && configs.iter().all(|c| c.algorithm == Algorithm::Gemv),
            "{case}: {configs:?}"
        );

        let built: Vec<_> = configs
            .into_iter()
            .map(|config| (Some(config), Some(gemms.build(&step, config))))
            .collect();

        gemms.compile().unwrap();

        let check = |config: &Option<_>, got: &[Tensor]| {
            for (want, have) in expected.iter().zip(got) {
                let have = have.to_dtype(DType::F32).unwrap().to_vec::<f32>();
                let scale = want.iter().fold(1f32, |m, v| m.max(v.abs()));

                for (e, (w, v)) in want.iter().zip(&have).enumerate() {
                    assert!(
                        (w - v).abs() <= scale * tol,
                        "{case} {config:?} [{e}]: {v} != {w}"
                    );
                }
            }
        };

        check(&None, &untuned);

        let bits = |got: &[Tensor]| -> Vec<Vec<u32>> {
            got.iter()
                .map(|t| t.to_dtype(DType::F32).unwrap().to_vec::<f32>())
                .map(|v| v.iter().map(|x| x.to_bits()).collect())
                .collect()
        };

        for (config, tuned) in built {
            plan.steps_mut()[at].tuned = tuned;
            let got = plan.run(&on_mps).unwrap();
            check(&config, &got);

            assert_eq!(
                bits(&got),
                bits(&untuned),
                "{case} {config:?}: not the default's sums"
            );
        }
    }
}
