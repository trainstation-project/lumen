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

/// `Strides::of` keeps only the operands not laid out contiguously in their own buffers.
#[test]
fn attention_strides_record_only_strided_operands() {
    use super::common::attention::Strides;
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
