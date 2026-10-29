//! Tests for the shared graph compiler passes: CSE, simplification, host stages and the pass pipeline.
use super::common::cse::cse;
use crate::graph::Primitive::*;
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::graph_compiler::plan_tests::data;
use crate::kernels::reference;
use crate::{DType, Scalar};

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
    let simplified = super::common::simplify::simplify_with(&g, false);
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
    let swapped = super::common::simplify::simplify_with(&g, true);
    assert_eq!(names(&swapped), ["dot_general"], "{swapped}");
    same_values(&g, &swapped, &inputs);
    assert_eq!(
        names(&super::common::simplify::simplify_with(&g, false)),
        ["dot_general", "transpose"]
    );
    let twice = super::common::simplify::simplify_with(&build(true), true);
    assert_eq!(names(&twice), ["dot_general", "transpose"], "{twice}");
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
    let moved = super::common::simplify::simplify_with(&g, true);
    assert_eq!(
        names(&moved),
        ["add", "full", "mul", "exp", "transpose"],
        "{moved}"
    );
    same_values(&g, &moved, &[data(&[2, 3], 1), data(&[2, 3], 2)]);
    assert_eq!(
        names(&super::common::simplify::simplify_with(&g, false)),
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
    let moved = super::common::simplify::simplify_with(&g, true);
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
    let moved = super::common::simplify::simplify_with(&g, true);
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
        let moved = super::common::simplify::simplify_with(&g, true);
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
    let staged = plan.staged.as_ref().expect("a staged plan");
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
    let staged = plan.staged.as_ref().expect("a staged plan");
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
    assert!(plan.staged.is_none());
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
    use super::common::cse::Cse;
    use super::pass::{CompilerPass, GraphModule, Pipeline};
    use super::{CpuConfig, Options};

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
        config: super::Configs {
            cpu: CpuConfig {
                dump_to: Some(dir.to_string_lossy().into_owned()),
                ..CpuConfig::default()
            },
            ..Default::default()
        },
        ..Options::default()
    };
    let mut module = GraphModule {
        graph: g,
        device: crate::Device::Cpu(0),
        options: &options,
    };
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

/// A `row` fusion over `[4, 8]` rows computing `(x - rowsum(x)) * gamma` from operands `x` and `gamma`.
fn row_fusion() -> Primitive {
    let mut b = Graph::new();
    let x = b.input(ty(&[4, 8]));
    let gamma = b.input(ty(&[4, 8]));
    let sum = ReduceSum {
        axes: vec![1],
        accum_dtype: DType::F32,
    };
    let s = apply(&mut b, sum, &[x]);
    let s = apply(
        &mut b,
        Reshape {
            new_sizes: vec![4, 1],
        },
        &[s],
    );
    let bcast = BroadcastInDim {
        shape: vec![4, 8],
        broadcast_dimensions: vec![0, 1],
    };
    let s = apply(&mut b, bcast, &[s]);
    let y = apply(&mut b, Sub, &[x, s]);
    let y = apply(&mut b, Mul, &[y, gamma]);
    b.set_outputs(&[y]).unwrap();
    Fusion {
        name: "row".into(),
        label: "row",
        body: b,
    }
}

/// `row(a @ w, gamma)`, or with `residual` `row(a @ w + r, gamma)` that also returns the sum.
fn dot_into_row(residual: bool) -> Graph {
    let mut g = Graph::new();
    let a = g.input(ty(&[4, 6]));
    let w = g.input(ty(&[6, 8]));
    let gamma = g.input(ty(&[4, 8]));
    let r = residual.then(|| g.input(ty(&[4, 8])));
    let d = apply(&mut g, matmul(), &[a, w]);
    let v = match r {
        Some(r) => apply(&mut g, Add, &[d, r]),
        None => d,
    };
    let y = apply(&mut g, row_fusion(), &[v, gamma]);
    let outputs = match r {
        Some(_) => vec![y, v],
        None => vec![y],
    };
    g.set_outputs(&outputs).unwrap();
    g
}

/// The `name` and body primitive names of fusion node `node`.
fn fusion_of(node: &crate::graph::Node) -> (&str, Vec<&'static str>) {
    match &node.primitive {
        Fusion { name, body, .. } => (name.as_str(), names(body)),
        p => panic!("not a fusion: {}", p.name()),
    }
}

/// A matmul read whole by a `row` fusion joins it as one `dot_row` fusion with the same values, other
/// nodes (other fusions too) are copied; a backend that doesn't join the operand or fit the body keeps
/// the dot and row apart.
#[test]
fn dots_join_the_row_fusions_reading_them() {
    use super::common::dot_rows::dot_rows;
    let mut g = dot_into_row(false);
    let y = g.outputs()[0];
    let n = apply(&mut g, Neg, &[y]);
    let mut body = Graph::new();
    let v = body.input(ty(&[4, 8]));
    let t = apply(&mut body, Tanh, &[v]);
    body.set_outputs(&[t]).unwrap();
    let loop_ = Fusion {
        name: "loop".into(),
        label: "loop",
        body,
    };
    let l = apply(&mut g, loop_, &[n]);
    g.set_outputs(&[l]).unwrap();
    let fused = dot_rows(&g, |_, _| true, |_| true, false);
    assert_eq!(names(&fused)[1..], ["neg", "loop"], "{fused}");
    assert_eq!(
        fusion_of(&fused.nodes()[0]),
        (
            "dot_row",
            vec![
                "dot_general",
                "reduce_sum",
                "reshape",
                "broadcast_in_dim",
                "sub",
                "mul"
            ]
        )
    );
    same_values(
        &g,
        &fused,
        &[data(&[4, 6], 1), data(&[6, 8], 2), data(&[4, 8], 3)],
    );
    let apart = ["dot_general", "row", "neg", "loop"];
    assert_eq!(names(&dot_rows(&g, |_, _| false, |_| true, false)), apart);
    assert_eq!(names(&dot_rows(&g, |_, _| true, |_| false, false)), apart);
}

/// With `through`, a residual add between the matmul and the row joins too and the sum the program
/// also returns comes out as the fusion's extra output; without it the nodes stay apart.
#[test]
fn dots_join_rows_through_a_residual_add() {
    use super::common::dot_rows::dot_rows;
    let g = dot_into_row(true);
    assert_eq!(
        names(&dot_rows(&g, |_, _| true, |_| true, false)),
        ["dot_general", "add", "row"]
    );
    let fused = dot_rows(&g, |_, _| true, |_| true, true);
    assert_eq!(fused.nodes().len(), 2, "{fused}");
    let (name, body) = fusion_of(&fused.nodes()[0]);
    assert_eq!((name, &body[..2]), ("dot_row", &["dot_general", "add"][..]));
    assert_eq!(names(&fused)[1], "fusion_output");
    same_values(
        &g,
        &fused,
        &[
            data(&[4, 6], 1),
            data(&[6, 8], 2),
            data(&[4, 8], 3),
            data(&[4, 8], 4),
        ],
    );
}

/// Grouped nodes become one fusion at their root that reads the values from outside the group and
/// exports the extra outputs read later; ungrouped nodes are copied as they were.
#[test]
fn loop_groups_become_fusions() {
    use super::common::loops::{fused, members};
    let mut g = Graph::new();
    let x = g.input(ty(&[8]));
    let y = g.input(ty(&[8]));
    let e = apply(&mut g, Exp, &[x]);
    let m = apply(&mut g, Mul, &[e, y]);
    let a = apply(&mut g, Add, &[m, x]);
    let n = apply(&mut g, Neg, &[a]);
    let t = apply(&mut g, Tanh, &[m]);
    g.set_outputs(&[n, t]).unwrap();
    let group = [Some(2), Some(2), Some(2), None, None];
    assert_eq!(members(&group, 2), [0, 1, 2]);
    let out = fused(
        &g,
        &group,
        |root, members| {
            assert_eq!((root, members), (2, &[0, 1, 2][..]));
            ("loop", 0)
        },
        |root| if root == 2 { vec![m] } else { vec![] },
    );
    assert_eq!(
        fusion_of(&out.nodes()[0]),
        ("loop", vec!["exp", "mul", "add"])
    );
    assert_eq!(names(&out)[1..], ["fusion_output", "neg", "tanh"], "{out}");
    let Fusion { body, .. } = &out.nodes()[0].primitive else {
        unreachable!()
    };
    assert_eq!((body.inputs().len(), body.outputs().len()), (2, 2));
    same_values(&g, &out, &[data(&[8], 1), data(&[8], 2)]);
}

/// Runs `pass` on `g` as a one-pass CPU pipeline and returns the rewritten graph.
fn run_pass(
    g: &Graph,
    pass: impl super::pass::CompilerPass<super::pass::GraphModule<'static>> + 'static,
) -> Graph {
    use super::pass::{GraphModule, Module, Pipeline};
    let options: &'static super::Options = Box::leak(Box::default());
    let mut module = GraphModule {
        graph: g.clone(),
        device: crate::Device::Cpu(0),
        options,
    };
    assert!(std::ptr::eq(module.options(), options));
    Pipeline::new("test").add(pass).run(&mut module).unwrap();
    module.graph
}

/// A 2-D dot reading a reshape that flattens leading dims runs on the unflattened lhs and reshapes
/// its result; a dot of a plain matrix is left alone.
#[test]
fn dots_read_through_flattening_reshapes() {
    use super::common::fold_dot_reshapes::FoldDotReshapes;
    let mut g = Graph::new();
    let x = g.input(ty(&[2, 3, 4]));
    let w = g.input(ty(&[4, 5]));
    let z = g.input(ty(&[6, 4]));
    let flat = apply(
        &mut g,
        Reshape {
            new_sizes: vec![6, 4],
        },
        &[x],
    );
    let a = apply(&mut g, matmul(), &[flat, w]);
    let b = apply(&mut g, matmul(), &[z, w]);
    g.set_outputs(&[a, b]).unwrap();
    let folded = run_pass(&g, FoldDotReshapes);
    assert_eq!(
        names(&folded),
        ["reshape", "dot_general", "reshape", "dot_general"],
        "{folded}"
    );
    let dot = &folded.nodes()[1];
    assert_eq!(folded.type_of(dot.output).shape, [2, 3, 5]);
    assert_eq!(dot.inputs[0], folded.inputs()[0]);
    same_values(
        &g,
        &folded,
        &[data(&[2, 3, 4], 1), data(&[4, 5], 2), data(&[6, 4], 3)],
    );
}

/// Dots whose operands have only size-1 free dims become a multiply and reduce-sum, laid out,
/// broadcast, cast and transposed into the dot's output; other dots stay dots.
#[test]
fn vector_dots_become_multiply_and_reduce() {
    use super::common::dot_strength::DotStrength;
    let dot = |lb: usize, lc: usize, rb: usize, rc: usize, output_dtype| DotGeneral {
        lhs_contracting: vec![lc],
        rhs_contracting: vec![rc],
        lhs_batch: vec![lb],
        rhs_batch: vec![rb],
        accum_dtype: DType::F32,
        output_dtype,
    };
    let cast = |d| Cast { new_dtype: d };
    let mut g = Graph::new();
    let x = g.input(ty(&[3, 1, 2]));
    let y = g.input(ty(&[2, 3, 1]));
    let u = g.input(ty(&[2, 3]));
    let v = g.input(ty(&[1, 2, 3]));
    let w = g.input(ty(&[3, 4]));
    // bf16 operands accumulated in f32, written as bf16: batch on lhs dim 2, rhs dim 0.
    let xb = apply(&mut g, cast(DType::BF16), &[x]);
    let yb = apply(&mut g, cast(DType::BF16), &[y]);
    let p = apply(&mut g, dot(2, 0, 0, 1, DType::BF16), &[xb, yb]);
    let p = apply(&mut g, cast(DType::F32), &[p]);
    // The rhs's free dim comes before its batch dim, so the sum is transposed back.
    let q = apply(&mut g, dot(0, 1, 1, 2, DType::F32), &[u, v]);
    let r = apply(&mut g, matmul(), &[u, w]);
    g.set_outputs(&[p, q, r]).unwrap();
    assert_eq!(g.type_of(p).shape, [2, 1, 1]);
    assert_eq!(g.type_of(q).shape, [2, 1]);
    let reduced = run_pass(&g, DotStrength);
    let dots = names(&reduced)
        .iter()
        .filter(|&&n| n == "dot_general")
        .count();
    assert_eq!(dots, 1, "{reduced}");
    for op in ["transpose", "broadcast_in_dim", "reduce_sum", "reshape"] {
        assert!(names(&reduced).contains(&op), "{op}: {reduced}");
    }
    same_values(
        &g,
        &reduced,
        &[
            data(&[3, 1, 2], 1),
            data(&[2, 3, 1], 2),
            data(&[2, 3], 3),
            data(&[1, 2, 3], 4),
            data(&[3, 4], 5),
        ],
    );
    // Integer dots are left alone.
    let mut g = Graph::new();
    let a = g.input(TensorType::new(DType::I32, &[3]));
    let b = g.input(TensorType::new(DType::I32, &[3]));
    let int_dot = DotGeneral {
        lhs_contracting: vec![0],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::I32,
        output_dtype: DType::I32,
    };
    let d = apply(&mut g, int_dot, &[a, b]);
    g.set_outputs(&[d]).unwrap();
    assert_eq!(names(&run_pass(&g, DotStrength)), ["dot_general"]);
}

/// Every device type's config reports its shared settings through `Configs::of` (meta devices use the
/// CPU one), and a zero memory limit means none.
#[test]
fn device_configs_expose_their_shared_settings() {
    use super::Configs;
    use crate::Device;
    let mut configs = Configs::default();
    assert_eq!(configs.of(Device::Tpu(0)).memory_limit(), None);
    macro_rules! set {
        ($($field:ident),*) => {$(
            configs.$field.merge_dots = false;
            configs.$field.flash_attention = false;
            configs.$field.fused_linear_cross_entropy_chunk_size = Some(64);
            configs.$field.memory_limit = 1 << 20;
            configs.$field.dump_to = Some(stringify!($field).into());
        )*};
    }
    set!(cpu, cuda, mps, tpu);
    for (device, dir) in [
        (Device::Cpu(0), "cpu"),
        (Device::Meta(0), "cpu"),
        (Device::Cuda(0), "cuda"),
        (Device::Mps(0), "mps"),
        (Device::Tpu(0), "tpu"),
    ] {
        let c = configs.of(device);
        assert!(!c.merge_dots() && !c.flash_attention(), "{device}");
        assert_eq!(c.fused_linear_cross_entropy_chunk_size(), Some(64));
        assert_eq!(c.memory_limit(), Some(1 << 20));
        assert_eq!(c.dump_to(), Some(dir));
    }
}

/// A staged program computes a constant output in its first device stage, writes a device output its
/// input is donated into in that input's buffer, and prints every stage.
#[test]
fn staged_plans_place_constants_and_donations() {
    use crate::graph_compiler::plan::Buffer;
    let device = match crate::device::mps::MPS.is_available() {
        true => crate::Device::Mps(0),
        false => crate::Device::Meta(0),
    };
    let mut g = Graph::new();
    let x = g.input(ty(&[2]));
    let three = apply(&mut g, full(&[2], 3.0), &[]);
    let y = apply(&mut g, Mul, &[x, three]);
    let h = apply(&mut g, ToHost, &[y]);
    let two = apply(&mut g, full(&[2], 2.0), &[]);
    let z = apply(&mut g, Mul, &[h, two]);
    let c = apply(&mut g, full(&[2], 5.0), &[]);
    g.set_outputs(&[y, z, c]).unwrap();
    let options = super::Options {
        // Output 1 comes from the host stage, so only the first pair donates.
        donate_into: vec![(0, 0), (0, 1)],
        ..super::Options::default()
    };
    let plan = super::compile_with(&g, device, &options).unwrap();
    let staged = plan.staged.as_ref().expect("a staged plan");
    assert_eq!(
        staged.outputs,
        [
            crate::graph_compiler::Source::Stage(0, 0),
            crate::graph_compiler::Source::Stage(1, 0),
            crate::graph_compiler::Source::Stage(0, 1),
        ]
    );
    let first = &staged.stages[0].plan;
    assert!(
        first.steps().iter().any(|s| s.output.0 == Buffer::Input(0)),
        "{first}"
    );
    let text = plan.to_string();
    assert!(text.starts_with("staged plan (workspace"), "{text}");
    assert!(
        text.contains("\nstage 1 (host, inputs [ToHost(0, 0)])"),
        "{text}"
    );
    assert!(
        text.ends_with("outputs [Stage(0, 0), Stage(1, 0), Stage(0, 1)]"),
        "{text}"
    );
    if device == crate::Device::Mps(0) {
        let x = crate::Tensor::from_slice(&[1.0f32, 2.0], DType::F32).to(device);
        let out = plan.run(&[x]).unwrap();
        assert_eq!(out[0].to_vec::<f32>(), [3.0, 6.0]);
        assert_eq!(out[1].to_vec::<f32>(), [6.0, 12.0]);
        assert_eq!(out[2].to_vec::<f32>(), [5.0, 5.0]);
    }
}

/// A softmax's diamonds become `row` fusions computing the same values; a backend that fits no row
/// body leaves the graph unchanged.
#[test]
fn diamonds_fuse_into_rows_the_backend_fits() {
    use super::common::diamonds::{diamonds, fused};
    let mut g = Graph::new();
    let x = g.input(ty(&[4, 8]));
    let per_row = |g: &mut Graph, v: Var| {
        let v = apply(
            g,
            Reshape {
                new_sizes: vec![4, 1],
            },
            &[v],
        );
        let bcast = BroadcastInDim {
            shape: vec![4, 8],
            broadcast_dimensions: vec![0, 1],
        };
        apply(g, bcast, &[v])
    };
    let m = apply(&mut g, ReduceMax { axes: vec![1] }, &[x]);
    let m = per_row(&mut g, m);
    let d = apply(&mut g, Sub, &[x, m]);
    let e = apply(&mut g, Exp, &[d]);
    let sum = ReduceSum {
        axes: vec![1],
        accum_dtype: DType::F32,
    };
    let s = apply(&mut g, sum, &[e]);
    let s = per_row(&mut g, s);
    let y = apply(&mut g, Div, &[e, s]);
    g.set_outputs(&[y]).unwrap();
    let rows = diamonds(&g, |_, _| true);
    let rowed = fused(&g, &rows, |_, _| true, |_| true);
    assert!(
        rowed
            .nodes()
            .iter()
            .all(|n| matches!(&n.primitive, Fusion { name, .. } if name == "row")),
        "{rowed}"
    );
    same_values(&g, &rowed, &[data(&[4, 8], 1)]);
    assert_eq!(names(&fused(&g, &rows, |_, _| true, |_| false)), names(&g));
}

/// On MPS, weights a device stage of a staged program packs into one block are lifted to program-level
/// packed inputs, appended after the program's own.
#[test]
fn staged_plans_lift_their_stages_packed_weights() {
    if !crate::device::mps::MPS.is_available() {
        return;
    }
    let dot = DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![1],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };
    let mut g = Graph::new();
    let x = g.input(ty(&[16, 32]));
    let ws: Vec<Var> = (0..3).map(|_| g.input(ty(&[64, 32]))).collect();
    let mut outs: Vec<Var> = ws
        .iter()
        .map(|&w| {
            let y = apply(&mut g, dot.clone(), &[x, w]);
            apply(&mut g, Neg, &[y])
        })
        .collect();
    let h = apply(&mut g, ToHost, &[outs[0]]);
    outs.push(apply(&mut g, Exp, &[h]));
    g.set_outputs(&outs).unwrap();
    let mask = vec![false, true, true, true];
    let options = super::Options {
        parameters: Some(mask.clone()),
        packable: mask,
        ..super::Options::default()
    };
    let plan = super::compile_with(&g, crate::Device::Mps(0), &options).unwrap();
    assert!(plan.staged.is_some(), "{plan}");
    assert_eq!(plan.packed(), [(vec![1, 2, 3], 0)], "{plan}");
    assert_eq!(plan.interleaved().len(), 1);
}
