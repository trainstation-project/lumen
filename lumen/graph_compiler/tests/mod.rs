//! Tests for the shared graph compiler passes: CSE, simplification, host stages and the pass pipeline.

#[cfg(lumen_mps_linked)]
mod mps;
mod tpu;
use super::passes::cse::cse;
use crate::graph::Primitive::*;
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::plan::tests::data;
use crate::{DType, Scalar, reference};

/// `g` as the `Simplify` pass leaves it, with `matched` as given.
fn simplified(g: &Graph, matched: bool) -> Graph {
    super::pipeline::rewritten(g, super::passes::simplify::Simplify { matched })
}

/// An `f32` tensor type of `shape`.
fn ty(shape: &[usize]) -> TensorType {
    TensorType::new(DType::F32, shape)
}

/// Applies `p` to `inputs`, panicking on a type error.
fn apply(g: &mut Graph, p: Primitive, inputs: &[Var]) -> Var {
    g.apply(p, inputs).unwrap()
}

/// Primitive names of `g`'s nodes, in order.
fn names(g: &Graph) -> Vec<&'static str> {
    g.nodes().iter().map(|n| n.primitive.name()).collect()
}

/// A plain `[m, k] x [k, n]` matmul accumulating and returning `f32`.
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

/// CSE merges the repeated matmul and zero broadcast and keeps the results unchanged.
#[test]
fn repeated_computations_run_once() {
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

/// CSE keeps nodes that differ in parameters or operand order.
#[test]
fn different_parameters_or_order_are_kept() {
    let mut g = Graph::new();
    let x = g.input(ty(&[4, 4]));
    let t = |p: Vec<usize>| Transpose { permutation: p };
    let a = apply(&mut g, t(vec![1, 0]), &[x]);
    let b = apply(&mut g, t(vec![0, 1]), &[x]);
    let c = apply(&mut g, Sub, &[a, b]);
    let d = apply(&mut g, Sub, &[b, a]);
    g.set_outputs(&[c, d]).unwrap();
    assert_eq!(names(&cse(&g)), names(&g));
}

/// Asserts `g` and `simplified` produce identical `f32` outputs on `inputs`.
fn same_values(g: &Graph, simplified: &Graph, inputs: &[crate::Tensor]) {
    let want = reference::run(g, inputs).unwrap();
    let got = reference::run(simplified, inputs).unwrap();

    for (w, g) in want.iter().zip(&got) {
        assert_eq!(w.to_vec::<f32>(), g.to_vec::<f32>());
    }
}

/// A filled `f32` constant of `shape`.
fn full(shape: &[usize], value: f64) -> Primitive {
    Full {
        shape: shape.to_vec(),
        fill_value: Scalar::Float(value),
        dtype: DType::F32,
    }
}

/// Identities, double negation, same-branch selects, no-op casts and reshape, transpose and broadcast
/// chains all fold away.
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
    let simplified = simplified(&g, false);

    assert_eq!(
        names(&simplified),
        ["reshape", "broadcast_in_dim"],
        "{simplified}"
    );

    same_values(&g, &simplified, &[data(&[2, 3], 1)]);
}

/// A transposed matmul becomes a matmul with swapped operands when the untransposed result isn't read.
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
    let swapped = simplified(&g, true);
    assert_eq!(names(&swapped), ["dot_general"], "{swapped}");
    same_values(&g, &swapped, &inputs);

    assert_eq!(names(&simplified(&g, false)), ["dot_general", "transpose"]);

    let twice = simplified(&build(true), true);
    assert_eq!(names(&twice), ["dot_general", "transpose"], "{twice}");
}

/// The late fold: a bfloat16 dot's float32 accumulator, cast to bfloat16 (after a reshape too), becomes
/// one dot rounding to bfloat16 once pattern matching has run; read twice, or before, it stays a cast.
#[test]
fn rounding_casts_fold_into_their_dots() {
    let build = |reshaped: bool, read_twice: bool| {
        let mut g = Graph::new();
        let a = g.input(TensorType::new(DType::BF16, &[4, 6]));
        let b = g.input(TensorType::new(DType::BF16, &[6, 5]));
        let mut d = apply(&mut g, matmul(), &[a, b]);
        let product = d;

        if reshaped {
            d = apply(
                &mut g,
                Reshape {
                    new_sizes: vec![20],
                },
                &[d],
            );
        }

        let new_dtype = DType::BF16;
        let c = apply(&mut g, Cast { new_dtype }, &[d]);
        let outputs = if read_twice {
            vec![c, product]
        } else {
            vec![c]
        };
        g.set_outputs(&outputs).unwrap();
        g
    };

    for reshaped in [false, true] {
        let g = build(reshaped, false);
        let folded = simplified(&g, true);
        let dot = &folded.nodes()[0];
        assert_eq!(folded.type_of(dot.output).dtype, DType::BF16, "{folded}");
        assert_eq!(names(&folded).len(), 1 + usize::from(reshaped), "{folded}");

        let unfolded = simplified(&g, false);
        assert!(names(&unfolded).contains(&"cast"), "{unfolded}");
    }

    let twice = simplified(&build(false, true), true);
    assert_eq!(names(&twice), ["dot_general", "cast"], "{twice}");
}

/// With rearranging enabled, operand transposes move after the elementwise ops into one transpose.
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
    let moved = simplified(&g, true);

    assert_eq!(
        names(&moved),
        ["add", "full", "mul", "exp", "transpose"],
        "{moved}"
    );

    same_values(&g, &moved, &[data(&[2, 3], 1), data(&[2, 3], 2)]);

    assert_eq!(
        names(&simplified(&g, false)),
        ["full", "transpose", "transpose", "add", "mul", "exp"]
    );
}

/// A widening cast moves after reshapes, transposes and slices, and after a concat only when the
/// constant it joins survives the round trip.
#[test]
fn casts_move_past_layout_primitives() {
    let cast = |dtype| Cast { new_dtype: dtype };

    let widened = |g: &mut Graph| {
        let x = g.input(ty(&[2, 3]));
        let b = apply(g, cast(DType::BF16), &[x]);
        apply(g, cast(DType::F32), &[b])
    };

    let inputs = [data(&[2, 3], 1)];
    let mut g = Graph::new();
    let w = widened(&mut g);
    let r = apply(&mut g, Reshape { new_sizes: vec![6] }, &[w]);
    g.set_outputs(&[r]).unwrap();
    let moved = simplified(&g, true);
    assert_eq!(names(&moved), ["cast", "reshape", "cast"], "{moved}");
    same_values(&g, &moved, &inputs);
    let mut g = Graph::new();
    let y = g.input(ty(&[2, 3]));

    let t = Transpose {
        permutation: vec![1, 0],
    };

    let t = apply(&mut g, t, &[y]);

    let slice = Slice {
        start_indices: vec![0, 0],
        limit_indices: vec![2, 2],
        view: false,
    };

    let s = apply(&mut g, slice, &[t]);
    let n = apply(&mut g, cast(DType::BF16), &[s]);
    let o = apply(&mut g, cast(DType::F32), &[n]);
    g.set_outputs(&[o]).unwrap();
    let moved = simplified(&g, true);

    assert_eq!(
        names(&moved),
        ["cast", "transpose", "slice", "cast"],
        "{moved}"
    );

    same_values(&g, &moved, &inputs);

    // 1.0 is exact in bf16, so the widening cast can move past the concat; 0.1 isn't.
    for (value, want) in [
        (1.0, vec!["cast", "full", "concatenate", "cast"]),
        (0.1, vec!["cast", "cast", "full", "concatenate"]),
    ] {
        let mut g = Graph::new();
        let w = widened(&mut g);
        let c = apply(&mut g, full(&[2, 3], value), &[]);

        let cat = apply(
            &mut g,
            Concatenate {
                dimension: 0,
                in_place: false,
            },
            &[w, c],
        );

        g.set_outputs(&[cat]).unwrap();
        let moved = simplified(&g, true);
        assert_eq!(names(&moved), want, "{moved}");
        same_values(&g, &moved, &inputs);
    }
}

/// A device multiply copied to the host and multiplied there.
fn to_host_graph() -> Graph {
    let mut g = Graph::new();
    let x = g.input(ty(&[2]));
    let three = apply(&mut g, full(&[2], 3.0), &[]);
    let y = apply(&mut g, Mul, &[x, three]);
    let h = apply(&mut g, ToHost, &[y]);
    let two = apply(&mut g, full(&[2], 2.0), &[]);
    let z = apply(&mut g, Mul, &[h, two]);
    g.set_outputs(&[z]).unwrap();
    g
}

/// Work after a `ToHost` runs in a separate host stage that reads the copied value.
#[test]
fn host_values_run_in_a_host_stage() {
    let device = match crate::device::mps::MPS.is_available() {
        true => crate::Device::Mps(0),
        false => crate::Device::Meta(0),
    };

    let plan = super::compile(&to_host_graph(), device).unwrap();
    let staged = plan.staged().expect("a staged plan");
    let hosts: Vec<bool> = staged.stages.iter().map(|s| s.host).collect();
    assert_eq!(hosts, [false, true]);

    assert_eq!(
        staged.stages[1].inputs,
        [crate::graph_compiler::Source::ToHost(0, 0)]
    );

    if device == crate::Device::Mps(0) {
        let x = crate::Tensor::from_slice(&[1.0f32, 2.0], DType::F32).to(device);
        let out = plan.run(&[x]).unwrap();
        assert_eq!(out[0].device(), crate::Device::Cpu(0));
        assert_eq!(out[0].to_vec::<f32>(), [6.0, 12.0]);
    }
}

/// A value copied to the host and straight back splits the plan into device, host and device stages.
#[test]
fn values_copied_back_at_once_pass_through_a_host_stage() {
    let device = match crate::device::mps::MPS.is_available() {
        true => crate::Device::Mps(0),
        false => crate::Device::Meta(0),
    };

    let mut g = Graph::new();
    let x = g.input(ty(&[2]));
    let three = apply(&mut g, full(&[2], 3.0), &[]);
    let y = apply(&mut g, Mul, &[x, three]);
    let h = apply(&mut g, ToHost, &[y]);
    let d = apply(&mut g, ToDevice, &[h]);
    let two = apply(&mut g, full(&[2], 2.0), &[]);
    let z = apply(&mut g, Mul, &[d, two]);
    g.set_outputs(&[z]).unwrap();
    let plan = super::compile(&g, device).unwrap();
    let staged = plan.staged().expect("a staged plan");
    let hosts: Vec<bool> = staged.stages.iter().map(|s| s.host).collect();
    assert_eq!(hosts, [false, true, false]);

    assert_eq!(
        staged.stages[2].inputs,
        [crate::graph_compiler::Source::ToDevice(1, 0)]
    );

    if device == crate::Device::Mps(0) {
        let x = crate::Tensor::from_slice(&[1.0f32, 2.0], DType::F32).to(device);
        let out = plan.run(&[x]).unwrap();
        assert_eq!(out[0].device(), crate::Device::Mps(0));
        assert_eq!(out[0].to_vec::<f32>(), [6.0, 12.0]);
    }
}

/// A `ToDevice` of a device value is a no-op and leaves one unstaged plan.
#[test]
fn programs_without_host_values_are_one_plan() {
    let mut g = Graph::new();
    let x = g.input(ty(&[2]));
    let d = apply(&mut g, ToDevice, &[x]);
    let y = apply(&mut g, Mul, &[d, x]);
    g.set_outputs(&[y]).unwrap();
    let plan = super::compile(&g, crate::Device::Meta(0)).unwrap();
    assert!(plan.staged().is_none());
    let steps: Vec<_> = plan.steps().iter().map(|s| s.primitive.name()).collect();
    assert_eq!(steps, ["mul"]);
}

/// Mixing a host value and a device value in one op is a compile error.
#[test]
fn ops_on_host_and_device_values_raise() {
    let mut g = Graph::new();
    let x = g.input(ty(&[2]));
    let h = apply(&mut g, ToHost, &[x]);
    let y = apply(&mut g, Add, &[h, x]);
    g.set_outputs(&[y]).unwrap();
    let error = super::compile(&g, crate::Device::Meta(0)).unwrap_err();

    assert!(
        error.contains("Expected all tensors to be on the same device"),
        "{error}"
    );
}

/// A pipeline runs its passes in order and dumps the graph before and after each to `dump_to`.
#[test]
#[cfg_attr(miri, ignore = "file I/O, which Miri isolates")]
fn a_pipeline_runs_its_passes_in_order_and_dumps_each() {
    use super::passes::CompilerPass;
    use super::passes::cse::Cse;
    use super::pipeline::{GraphModule, Pipeline};
    use super::{CompilerConfig, Options};

    /// Test pass that negates the graph's first output.
    struct Negate;

    impl CompilerPass<GraphModule<'_>> for Negate {
        fn name(&self) -> &'static str {
            "negate"
        }
        fn apply(&mut self, m: &mut GraphModule<'_>) -> Result<(), String> {
            let mut g = m.graph.clone();
            let out = apply(&mut g, Neg, &[m.graph.outputs()[0]]);
            g.set_outputs(&[out]).unwrap();
            m.graph = g;
            Ok(())
        }
    }

    let mut g = Graph::new();
    let x = g.input(ty(&[4]));
    let y = apply(&mut g, Exp, &[x]);
    let z = apply(&mut g, Exp, &[x]);
    let s = apply(&mut g, Add, &[y, z]);
    g.set_outputs(&[s]).unwrap();
    let dir = std::env::temp_dir().join(format!("lumen-passes-{}", std::process::id()));

    let options = Options {
        config: CompilerConfig {
            dump_to: Some(dir.to_string_lossy().into_owned()),
            ..CompilerConfig::default()
        },
        ..Options::default()
    };

    let mut module = GraphModule::new(&g, &options);

    Pipeline::new("test")
        .add(Cse)
        .add(Negate)
        .run(&mut module)
        .unwrap();

    assert_eq!(names(&module.graph), ["exp", "add", "neg"]);

    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();

    files.sort();
    let steps: Vec<&str> = files.iter().map(|f| f.split_once('.').unwrap().1).collect();

    assert_eq!(
        steps,
        ["test.00.input.txt", "test.01.cse.txt", "test.02.negate.txt"]
    );

    let last = std::fs::read_to_string(dir.join(&files[2])).unwrap();
    assert_eq!(last, module.graph.to_string());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `Strides::of` keeps only the operands not laid out contiguously in their own buffers.
#[test]
fn attention_strides_record_only_strided_operands() {
    use super::passes::attention::Strides;
    let mut g = Graph::new();
    let a = g.input(ty(&[2, 3]));
    let b = g.input(ty(&[2, 3]));
    let sum = apply(&mut g, Add, &[a, b]);
    g.set_outputs(&[sum, a]).unwrap();

    let strides = Strides::of(
        &g,
        &[vec![3, 1], vec![1, 2]],
        &[(0, vec![3, 1]), (4, vec![3, 1])],
    );

    assert_eq!(strides.inputs, [(1, vec![1, 2])].into_iter().collect());

    assert_eq!(
        strides.outputs,
        [(1, (4, vec![3, 1]))].into_iter().collect()
    );

    assert_eq!(
        Strides::of(
            &g,
            &[vec![3, 1], vec![3, 1]],
            &[(0, vec![3, 1]), (0, vec![3, 1])]
        ),
        Strides::default()
    );
}

/// `x` (`[8, 32]`) scaled per row by `r` (`[8]`), rounded to bfloat16, times a bfloat16 `w` (`[32, 16]`)
/// accumulated in float32 and rounded to bfloat16: a norm's `1 / rms` before the next matmul.
fn row_scaled_matmul(extra_output: bool) -> Graph {
    let mut g = Graph::new();
    let x = g.input(ty(&[8, 32]));
    let r = g.input(ty(&[8]));
    let w = g.input(TensorType::new(DType::BF16, &[32, 16]));

    let spread = BroadcastInDim {
        shape: vec![8, 32],
        broadcast_dimensions: vec![0],
    };

    let r = apply(&mut g, spread, &[r]);
    let scaled = apply(&mut g, Mul, &[x, r]);
    let rounded = apply(
        &mut g,
        Cast {
            new_dtype: DType::BF16,
        },
        &[scaled],
    );

    let dot = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::BF16,
    };

    let y = apply(&mut g, dot, &[rounded, w]);
    let outputs = if extra_output {
        vec![y, rounded]
    } else {
        vec![y]
    };
    g.set_outputs(&outputs).unwrap();
    g
}

/// A per-row scale of a matmul's lhs, through a rounding cast, moves to its output: the matmul of the
/// rounded unscaled rows in float32, times the scale, rounded once to bfloat16; the values the traced
/// program's to bfloat16 rounding.
#[test]
fn row_scale_moves_past_its_matmul() {
    use super::passes::linear_reordering::LinearReordering;
    use super::pipeline::rewritten;

    let g = row_scaled_matmul(false);
    let out = rewritten(&g, LinearReordering { tile: None });
    let nodes = out.nodes();
    let dot = nodes
        .iter()
        .position(|n| n.primitive.name() == "dot_general")
        .unwrap();

    assert_eq!(out.type_of(nodes[dot].output).dtype, DType::F32, "{out}");
    assert!(!nodes[..dot].iter().any(|n| n.primitive == Mul), "{out}");
    assert_eq!(
        names(&out)[dot + 1..],
        ["broadcast_in_dim", "mul", "cast"],
        "{out}"
    );

    let inputs = [
        data(&[8, 32], 1),
        data(&[8], 2),
        data(&[32, 16], 3).to_dtype(DType::BF16).unwrap(),
    ];

    let values = |g: &Graph| {
        reference::run(g, &inputs).unwrap()[0]
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec::<f32>()
    };

    let (want, got) = (values(&g), values(&out));
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));

    for (w, v) in want.iter().zip(&got) {
        assert!((w - v).abs() <= scale * 2f32.powi(-6), "{w} {v}");
    }
}

/// A per-row division of a matmul's lhs (`x / rms`, the rms kept `[rows, 1]`) moves to its output as a
/// division; the values the traced program's to float32 rounding.
#[test]
fn row_division_moves_past_its_matmul() {
    use super::passes::linear_reordering::LinearReordering;
    use super::pipeline::rewritten;

    let mut g = Graph::new();
    let x = g.input(ty(&[8, 32]));
    let rms = g.input(ty(&[8, 1]));
    let w = g.input(ty(&[32, 16]));

    let spread = BroadcastInDim {
        shape: vec![8, 32],
        broadcast_dimensions: vec![0, 1],
    };

    let rms = apply(&mut g, spread, &[rms]);
    let normalized = apply(&mut g, Div, &[x, rms]);
    let y = apply(&mut g, matmul(), &[normalized, w]);
    g.set_outputs(&[y]).unwrap();
    let out = rewritten(&g, LinearReordering { tile: None });

    assert_eq!(
        names(&out),
        ["dot_general", "broadcast_in_dim", "div"],
        "{out}"
    );

    let inputs = [data(&[8, 32], 1), data(&[8, 1], 2), data(&[32, 16], 3)];
    let want = reference::run(&g, &inputs).unwrap()[0].to_vec::<f32>();
    let got = reference::run(&out, &inputs).unwrap()[0].to_vec::<f32>();

    for (w, v) in want.iter().zip(&got) {
        assert!((w - v).abs() <= 1e-4 * w.abs().max(1.0), "{w} {v}");
    }
}

/// A matmul's lhs read elsewhere too (the rounded rows returned) keeps its scale: removing it would
/// compute the rows twice. A per-column scale (over the contracted axis) does not commute with the matmul
/// and stays.
#[test]
fn shared_lhs_and_column_scales_stay() {
    use super::passes::linear_reordering::LinearReordering;
    use super::pipeline::rewritten;

    let g = row_scaled_matmul(true);
    assert_eq!(
        names(&rewritten(&g, LinearReordering { tile: None })),
        names(&g)
    );

    let mut g = Graph::new();
    let x = g.input(ty(&[8, 32]));
    let gamma = g.input(ty(&[32]));
    let w = g.input(ty(&[32, 16]));

    let spread = BroadcastInDim {
        shape: vec![8, 32],
        broadcast_dimensions: vec![1],
    };

    let gamma = apply(&mut g, spread, &[gamma]);
    let scaled = apply(&mut g, Mul, &[x, gamma]);
    let y = apply(&mut g, matmul(), &[scaled, w]);
    g.set_outputs(&[y]).unwrap();

    assert_eq!(
        names(&rewritten(&g, LinearReordering { tile: None })),
        names(&g)
    );
}

/// A weight's per-output-channel scale (`x · (q · s)`, a dequantized weight's) moves to the matmul's
/// output columns: the scaled weight is never computed; the values the traced program's to float32
/// rounding.
#[test]
fn column_scale_of_the_weight_moves_past_its_matmul() {
    use super::passes::linear_reordering::LinearReordering;
    use super::pipeline::rewritten;

    let mut g = Graph::new();
    let x = g.input(ty(&[8, 32]));
    let q = g.input(ty(&[32, 16]));
    let s = g.input(ty(&[16]));

    let spread = BroadcastInDim {
        shape: vec![32, 16],
        broadcast_dimensions: vec![1],
    };

    let s = apply(&mut g, spread, &[s]);
    let w = apply(&mut g, Mul, &[q, s]);
    let y = apply(&mut g, matmul(), &[x, w]);
    g.set_outputs(&[y]).unwrap();
    let out = rewritten(&g, LinearReordering { tile: None });

    assert_eq!(
        names(&out),
        ["dot_general", "broadcast_in_dim", "mul"],
        "{out}"
    );

    let inputs = [data(&[8, 32], 1), data(&[32, 16], 2), data(&[16], 3)];
    let want = reference::run(&g, &inputs).unwrap()[0].to_vec::<f32>();
    let got = reference::run(&out, &inputs).unwrap()[0].to_vec::<f32>();

    for (w, v) in want.iter().zip(&got) {
        assert!((w - v).abs() <= 1e-4 * w.abs().max(1.0), "{w} {v}");
    }
}

/// `factor` (an input of the kept axes' sizes, or `full`'s rank with 1s on the others when `keepdim`)
/// broadcast to `full` along `kept`.
fn broadcast_factor(g: &mut Graph, full: &[usize], kept: &[usize], keepdim: bool) -> Var {
    let sizes: Vec<usize> = match keepdim {
        true => (0..full.len())
            .map(|a| if kept.contains(&a) { full[a] } else { 1 })
            .collect(),
        false => kept.iter().map(|&a| full[a]).collect(),
    };

    let f = g.input(ty(&sizes));

    let broadcast_dimensions = match keepdim {
        true => (0..full.len()).collect(),
        false => kept.to_vec(),
    };

    let spread = BroadcastInDim {
        shape: full.to_vec(),
        broadcast_dimensions,
    };

    apply(g, spread, &[f])
}

/// An `[8, 32] x [32, 16]` matmul of `x` stored `[32, 8]` when `xt` and `w` stored `[16, 32]` when `wt`,
/// its lhs scaled per output row when `row` and its rhs per output column when `col` (each factor of its
/// kept axis's size, or the operand's rank with 1s when `keepdim`); the graph and its inputs' shapes.
fn factored_matmul(
    xt: bool,
    wt: bool,
    keepdim: bool,
    row: bool,
    col: bool,
) -> (Graph, Vec<Vec<usize>>) {
    let mut g = Graph::new();
    let (xs, ws) = (
        if xt { [32, 8] } else { [8, 32] },
        if wt { [16, 32] } else { [32, 16] },
    );
    let (lc, rc) = (usize::from(!xt), usize::from(wt));
    let x = g.input(ty(&xs));
    let w = g.input(ty(&ws));

    let lhs = match row {
        true => {
            let r = broadcast_factor(&mut g, &xs, &[1 - lc], keepdim);
            apply(&mut g, Mul, &[x, r])
        }
        false => x,
    };

    let rhs = match col {
        true => {
            let s = broadcast_factor(&mut g, &ws, &[1 - rc], keepdim);
            apply(&mut g, Mul, &[s, w])
        }
        false => w,
    };

    let dot = DotGeneral {
        lhs_contracting: vec![lc],
        rhs_contracting: vec![rc],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };

    let y = apply(&mut g, dot, &[lhs, rhs]);
    g.set_outputs(&[y]).unwrap();
    let shapes = g
        .inputs()
        .iter()
        .map(|&v| g.type_of(v).shape.clone())
        .collect();
    (g, shapes)
}

/// Asserts `out`, `g` reordered, multiplies only after its matmul, `moved` times (one a factor), and
/// computes `g`'s values to float32 rounding; `case` names the graph.
fn assert_reordered(g: &Graph, out: &Graph, shapes: &[Vec<usize>], moved: usize, case: &str) {
    let nodes = out.nodes();
    let dot = nodes
        .iter()
        .position(|n| n.primitive.name() == "dot_general")
        .unwrap();

    assert!(
        !nodes[..dot].iter().any(|n| n.primitive == Mul),
        "{case}: {out}"
    );
    let after = nodes[dot..].iter().filter(|n| n.primitive == Mul).count();
    assert_eq!(after, moved, "{case}: {out}");

    let inputs: Vec<_> = shapes
        .iter()
        .zip(1..)
        .map(|(s, seed)| data(s, seed))
        .collect();

    let want = reference::run(g, &inputs).unwrap()[0].to_vec::<f32>();
    let got = reference::run(out, &inputs).unwrap()[0].to_vec::<f32>();

    for (w, v) in want.iter().zip(&got) {
        assert!((w - v).abs() <= 1e-4 * w.abs().max(1.0), "{case}: {w} {v}");
    }
}

/// A per-row factor of the lhs and a per-column factor of the rhs, alone or together, move to the
/// matmul's output in every layout (`x W`, `x W.T`, `x.T W`, `x.T W.T`), each factor of its kept axis's
/// size or kept with its operand's rank; the values the traced program's to float32 rounding.
#[test]
fn factors_move_past_their_matmul_in_every_layout() {
    use super::passes::linear_reordering::LinearReordering;
    use super::pipeline::rewritten;

    for (xt, wt) in [(false, false), (false, true), (true, false), (true, true)] {
        for keepdim in [false, true] {
            for (row, col) in [(true, false), (false, true), (true, true)] {
                let case = format!("xt {xt} wt {wt} keepdim {keepdim} row {row} col {col}");
                let (g, shapes) = factored_matmul(xt, wt, keepdim, row, col);
                let out = rewritten(&g, LinearReordering { tile: None });
                let moved = usize::from(row) + usize::from(col);
                assert_reordered(&g, &out, &shapes, moved, &case);
            }
        }
    }
}

/// A factor along the contracted axis (the lhs's per-column `x · γ` of `x W`, the rhs's per-row
/// `W · t` of `x W`) does not commute with the matmul and stays, on either side in every layout.
#[test]
fn contracted_axis_factors_stay_in_every_layout() {
    use super::passes::linear_reordering::LinearReordering;
    use super::pipeline::rewritten;

    for (xt, wt) in [(false, false), (false, true), (true, false), (true, true)] {
        for side in 0..2 {
            let mut g = Graph::new();
            let (xs, ws) = (
                if xt { [32, 8] } else { [8, 32] },
                if wt { [16, 32] } else { [32, 16] },
            );
            let (lc, rc) = (usize::from(!xt), usize::from(wt));
            let mut operands = [g.input(ty(&xs)), g.input(ty(&ws))];
            let (shape, contracted) = if side == 0 { (xs, lc) } else { (ws, rc) };
            let f = broadcast_factor(&mut g, &shape, &[contracted], false);
            operands[side] = apply(&mut g, Mul, &[operands[side], f]);

            let dot = DotGeneral {
                lhs_contracting: vec![lc],
                rhs_contracting: vec![rc],
                lhs_batch: vec![],
                rhs_batch: vec![],
                accum_dtype: DType::F32,
                output_dtype: DType::F32,
            };

            let y = apply(&mut g, dot, &operands);
            g.set_outputs(&[y]).unwrap();

            let case = format!("xt {xt} wt {wt} side {side}");
            assert_eq!(
                names(&rewritten(&g, LinearReordering { tile: None })),
                names(&g),
                "{case}"
            );
        }
    }
}

/// A batched lhs (`[2, 4, 32] x [32, 16]`, `W` or `W.T`) moves its per-token factor (`[2, 4]` or
/// `[2, 4, 1]`) to the output's rows, with the weight's per-column factor too; the values the traced
/// program's to float32 rounding.
#[test]
fn batched_rows_move_past_their_matmul() {
    use super::passes::linear_reordering::LinearReordering;
    use super::pipeline::rewritten;

    for wt in [false, true] {
        for keepdim in [false, true] {
            let mut g = Graph::new();
            let ws = if wt { [16, 32] } else { [32, 16] };
            let rc = usize::from(wt);
            let x = g.input(ty(&[2, 4, 32]));
            let w = g.input(ty(&ws));
            let r = broadcast_factor(&mut g, &[2, 4, 32], &[0, 1], keepdim);
            let s = broadcast_factor(&mut g, &ws, &[1 - rc], keepdim);
            let lhs = apply(&mut g, Mul, &[x, r]);
            let rhs = apply(&mut g, Mul, &[w, s]);

            let dot = DotGeneral {
                lhs_contracting: vec![2],
                rhs_contracting: vec![rc],
                lhs_batch: vec![],
                rhs_batch: vec![],
                accum_dtype: DType::F32,
                output_dtype: DType::F32,
            };

            let y = apply(&mut g, dot, &[lhs, rhs]);
            g.set_outputs(&[y]).unwrap();
            let shapes: Vec<_> = g
                .inputs()
                .iter()
                .map(|&v| g.type_of(v).shape.clone())
                .collect();
            let out = rewritten(&g, LinearReordering { tile: None });
            assert_reordered(&g, &out, &shapes, 2, &format!("wt {wt} keepdim {keepdim}"));
        }
    }
}

/// `x` (`[8, 128]`) scaled per row by `1 / sqrt(sum(x²))` over its 128-wide rows (a norm's), times `w`
/// (`[128, 16]`).
fn norm_scaled_matmul() -> Graph {
    let mut g = Graph::new();
    let x = g.input(ty(&[8, 128]));
    let w = g.input(ty(&[128, 16]));

    let squares = apply(&mut g, Mul, &[x, x]);

    let sum = ReduceSum {
        axes: vec![1],
        accum_dtype: DType::F32,
    };

    let sum = apply(&mut g, sum, &[squares]);
    let r = apply(&mut g, Unary(crate::graph::UnaryOp::Rsqrt), &[sum]);

    let spread = BroadcastInDim {
        shape: vec![8, 128],
        broadcast_dimensions: vec![0],
    };

    let r = apply(&mut g, spread, &[r]);
    let scaled = apply(&mut g, Mul, &[x, r]);
    let y = apply(&mut g, matmul(), &[scaled, w]);
    g.set_outputs(&[y]).unwrap();
    g
}

/// With a tile width, the pass moves only a factor of a reduction over rows wider than the tile (a
/// norm's `1 / rms` over 128-wide rows past 64-wide tiles), whose rows would otherwise take a pass of
/// their own after the GEMM; not over rows a tile holds whole (256-wide tiles), nor a factor that is an
/// input (a per-row scale, a weight's dequantization scale); without one, any factor moves.
#[test]
fn reordering_moves_only_factors_of_cross_tile_reductions() {
    use super::passes::linear_reordering::LinearReordering;
    use super::pipeline::rewritten;

    let g = norm_scaled_matmul();
    let moved = |out: &Graph| names(out).ends_with(&["dot_general", "broadcast_in_dim", "mul"]);

    assert!(
        moved(&rewritten(&g, LinearReordering { tile: Some(64) })),
        "{g}"
    );
    assert_eq!(
        names(&rewritten(&g, LinearReordering { tile: Some(256) })),
        names(&g)
    );
    assert!(moved(&rewritten(&g, LinearReordering { tile: None })));

    let (g, _) = factored_matmul(false, false, false, true, true);
    assert_eq!(
        names(&rewritten(&g, LinearReordering { tile: Some(64) })),
        names(&g)
    );
    assert_ne!(
        names(&rewritten(&g, LinearReordering { tile: None })),
        names(&g)
    );
}

/// With `preserve_precision`, a forward reader of a round trip through bfloat16 (`f32(bf16(x))`) reads
/// `x`; a gradient's reader keeps the round trip (the activation saved in bfloat16); a round trip
/// through a wider dtype (`bf16(f32(y))`) is dropped for both.
#[test]
fn preserve_precision_drops_forward_round_trips() {
    use super::passes::preserve_precision::PreservePrecision;
    use super::pipeline::rewritten;

    let mut g = Graph::new();
    let x = g.input(ty(&[4]));
    let y = g.input(TensorType::new(DType::BF16, &[4]));
    let bf16 = Cast {
        new_dtype: DType::BF16,
    };
    let f32 = Cast {
        new_dtype: DType::F32,
    };

    let narrowed = apply(&mut g, bf16.clone(), &[x]);
    let trip = apply(&mut g, f32.clone(), &[narrowed]);
    let forward = apply(&mut g, Exp, &[trip]);
    let widened = apply(&mut g, f32, &[y]);
    let exact = apply(&mut g, bf16, &[widened]);

    g.set_backward(true);
    let gradient = apply(&mut g, Mul, &[trip, trip]);
    let other = apply(&mut g, Neg, &[exact]);
    g.set_backward(false);

    g.set_outputs(&[forward, gradient, other]).unwrap();
    let out = rewritten(&g, PreservePrecision);
    let nodes = out.nodes();
    let reads = |name: &str| nodes.iter().find(|n| n.primitive.name() == name).unwrap();

    // The forward exp reads x; the gradient's mul reads the round trip; neg reads y.
    assert_eq!(reads("exp").inputs, [out.inputs()[0]], "{out}");
    let mul = reads("mul");
    assert!(mul.backward, "{out}");
    assert_eq!(out.type_of(mul.inputs[0]).dtype, DType::F32, "{out}");
    assert_ne!(mul.inputs[0], out.inputs()[0], "{out}");
    assert_eq!(reads("neg").inputs, [out.inputs()[1]], "{out}");
    assert_eq!(
        nodes
            .iter()
            .filter(|n| n.primitive.name() == "cast")
            .count(),
        2,
        "{out}"
    );
}
