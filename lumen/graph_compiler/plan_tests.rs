//! Tests for plan compilation (buffer reuse, donation, scheduling, scratch, views) and,
//! when MPS is linked, for running plans on MPS against the reference evaluator.
use super::Plan;
use super::plan::Buffer;
use crate::graph::Primitive::*;
use crate::graph::reference;
use crate::graph::{Graph, Primitive, TensorType};
use crate::{DType, Scalar, Tensor};

/// Shorthand for `TensorType::new`.
fn ty(dtype: DType, shape: &[usize]) -> TensorType {
    TensorType::new(dtype, shape)
}

/// Deterministic pseudo-random float32 values in `[-2, 2)` of the given shape.
pub(crate) fn data(shape: &[usize], seed: u64) -> Tensor {
    let mut state = seed;

    let values: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|_| {
            // 64-bit LCG step; the top 24 bits become a uniform float.
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);

            (state >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0
        })
        .collect();

    Tensor::from_slice(&values, DType::F32).reshape(shape)
}

/// Compiles `g`, checks it matches the reference evaluator bitwise and returns the plan.
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

/// A two-layer ReLU MLP followed by a softmax, on `[4, 8]` inputs.
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

/// The MLP plan matches the reference, elides reshapes and writes its output buffer last.
#[test]
fn plan_matches_reference() {
    let inputs = [data(&[4, 8], 1), data(&[8, 16], 2), data(&[16, 3], 3)];
    let plan = check_against_reference(&mlp(), &inputs);

    assert!(
        plan.steps()
            .iter()
            .all(|s| !matches!(s.primitive, Reshape { .. }))
    );

    assert_eq!(plan.steps().last().unwrap().output.0, Buffer::Output(0));
}

/// A chain of elementwise ops runs entirely in place in the caller's buffers.
#[test]
fn plan_reuses_workspace() {
    let mut g = Graph::new();
    let mut v = g.input(ty(DType::F32, &[256]));

    for _ in 0..10 {
        v = g.apply(Tanh, &[v]).unwrap();
    }

    g.set_outputs(&[v]).unwrap();
    let plan = check_against_reference(&g, &[data(&[256], 4)]);
    assert_eq!(plan.workspace_bytes(), 0, "{plan}");
}

/// Scheduling reduces each branch before computing the next, shrinking the workspace.
#[test]
fn plan_schedules_steps_to_shrink_the_workspace() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[1024]));

    let values: Vec<_> = [Exp, Tanh, Neg]
        .into_iter()
        .map(|p| g.apply(p, &[x]).unwrap())
        .collect();

    let sum = ReduceSum {
        axes: vec![0],
        accum_dtype: DType::F32,
    };

    let sums: Vec<_> = values
        .iter()
        .map(|&v| g.apply(sum.clone(), &[v]).unwrap())
        .collect();

    let total = g.apply(Add, &[sums[0], sums[1]]).unwrap();
    let total = g.apply(Add, &[total, sums[2]]).unwrap();
    g.set_outputs(&[total]).unwrap();
    let plan = check_against_reference(&g, &[data(&[1024], 4)]);
    assert!(plan.workspace_bytes() < 2 * 4096, "{plan}");
    let ops: Vec<_> = plan.steps().iter().map(|s| s.primitive.name()).collect();
    assert_eq!(&ops[..2], ["exp", "reduce_sum"], "{plan}");
}

/// A memory limit below the needed workspace recomputes a value instead of keeping it.
#[test]
#[cfg_attr(miri, ignore = "128 KiB values, too slow for Miri")]
fn plan_rematerializes_under_a_memory_limit() {
    let mut g = Graph::new();
    const N: usize = 32 << 10;
    let x = g.input(ty(DType::F32, &[N]));
    let h = g.apply(Exp, &[x]).unwrap();
    let b = g.apply(Tanh, &[h]).unwrap();
    let e = g.apply(Exp, &[b]).unwrap();
    let c = g.apply(Mul, &[e, b]).unwrap();
    let d = g.apply(Mul, &[c, h]).unwrap();
    g.set_outputs(&[d]).unwrap();
    let inputs = [data(&[N], 4)];
    let expected = reference::run(&g, &inputs).unwrap();

    let plan = |limit| {
        let options = super::PlanOptions {
            memory_limit: limit,
            ..Default::default()
        };

        let plan = Plan::compile_with(&g, &options);
        let actual = plan.run(&inputs).unwrap();

        assert_eq!(
            expected[0].to_vec::<f32>(),
            actual[0].to_vec::<f32>(),
            "{plan}"
        );

        let exps = plan.steps().iter().filter(|s| s.primitive == Exp).count();
        (plan.workspace_bytes(), exps)
    };

    let (kept, exps) = plan(None);
    assert_eq!((kept, exps), (2 * 4 * N, 2));
    assert_eq!(plan(Some(kept)), (kept, 2));
    let (recomputed, exps) = plan(Some(1));
    assert_eq!((recomputed, exps), (4 * N, 3));
}

/// Philox-4x32-10 matches the Random123 known-answer tests.
#[test]
fn philox_matches_random123() {
    use crate::graph::reference::philox4x32;

    let kat: [([u32; 2], [u32; 4], [u32; 4]); 3] = [
        (
            [0; 2],
            [0; 4],
            [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
        ),
        (
            [u32::MAX; 2],
            [u32::MAX; 4],
            [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
        ),
        (
            [0xa4093822, 0x299f31d0],
            [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
            [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
        ),
    ];

    for (key, counter, block) in kat {
        assert_eq!(philox4x32(key, counter), block);
    }
}

/// `RandomBits` draws from one counter stream, so offsets select overlapping windows.
#[test]
fn random_bits_are_a_counter_stream() {
    use crate::graph::reference::philox_bits;

    let draw = |offset: u64, shape: &[usize]| {
        let mut g = Graph::new();
        let seed = g.input(ty(DType::U64, &[]));
        let start = g.input(ty(DType::U64, &[]));

        let bits = RandomBits {
            shape: shape.to_vec(),
            offset,
        };

        let r = g.apply(bits, &[seed, start]).unwrap();
        g.set_outputs(&[r]).unwrap();
        let seed = Tensor::from_slice(&[42u64 << 40 | 7], DType::U64).reshape(&[]);
        let start = Tensor::from_slice(&[1000u64], DType::U64).reshape(&[]);

        reference::run(&g, &[seed, start])
            .unwrap()
            .remove(0)
            .to_vec::<u32>()
    };

    let (seed, start) = (42u64 << 40 | 7, 1005);
    let whole = draw(5, &[2, 3]);
    let want: Vec<u32> = (0..6).map(|i| philox_bits(seed, start + i)).collect();
    assert_eq!(whole, want);
    assert_eq!(draw(6, &[5]), whole[1..]);
    assert_ne!(draw(11, &[6]), whole);
}

/// Outputs that alias other values get copy steps and unused nodes are dropped.
#[test]
fn plan_copies_aliased_outputs_and_drops_dead_code() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[2, 3]));
    let e = g.apply(Exp, &[x]).unwrap();
    let _dead = g.apply(Log, &[e]).unwrap();
    let flat = g.apply(Reshape { new_sizes: vec![6] }, &[e]).unwrap();
    g.set_outputs(&[e, flat, x]).unwrap();
    let plan = check_against_reference(&g, &[data(&[2, 3], 5)]);
    let ops: Vec<_> = plan.steps().iter().map(|s| s.primitive.name()).collect();
    assert_eq!(ops, ["exp", "reshape", "reshape"]);
    assert_eq!(plan.workspace_bytes(), 0);
}

/// Owned plans keep outputs in the workspace, read parameters in place and copy arguments.
#[test]
fn owned_plans_place_outputs_in_the_workspace_and_read_inputs_in_place() {
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
    assert_eq!(plan.steps()[0].inputs[0].0, Buffer::Input(0));
    assert_eq!(plan.steps()[0].inputs[1].0, Buffer::Input(1));
    let workspace = Tensor::zeros(&[plan.workspace_bytes()], DType::U8);
    let w = Tensor::from_slice(&[2.0f32; 4], DType::F32);
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], DType::F32);
    let first = plan.run_in(&workspace, &[x.clone(), w.clone()]).unwrap();
    assert_eq!(first[0].to_vec::<f32>(), [2.0, 4.0, 6.0, 8.0]);
    assert_eq!(first[1].to_vec::<f32>(), [1.0, 2.0, 3.0, 4.0]);
    assert!(first[0].shares_storage_with(&workspace) && first[2].shares_storage_with(&w));
    assert!(!first[1].shares_storage_with(&x));

    let run = |x: &[f32]| {
        plan.run_in(&workspace, &[Tensor::from_slice(x, DType::F32), w.clone()])
            .unwrap()
    };

    let second = run(&[0.0, 1.0, 0.0, 1.0]);
    assert_eq!(second[0].to_vec::<f32>(), [0.0, 2.0, 0.0, 2.0]);
    assert_eq!(first[0].to_vec::<f32>(), [0.0, 2.0, 0.0, 2.0]);
    let small = Tensor::zeros(&[plan.workspace_bytes() - 1], DType::U8);
    assert!(plan.run_in(&small, &[w.clone(), w.clone()]).is_err());
}

/// An argument sharing storage with a donated parameter is copied before it is overwritten.
#[test]
fn owned_plans_copy_arguments_sharing_a_donated_parameter() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[4]));
    let w = g.input(ty(DType::F32, &[4]));
    let neww = g.apply(Add, &[w, w]).unwrap();
    let y = g.apply(Mul, &[x, neww]).unwrap();
    let z = g.apply(Add, &[y, x]).unwrap();
    g.set_outputs(&[z, neww]).unwrap();

    let options = super::PlanOptions {
        parameters: Some(vec![false, true]),
        donate_into: vec![(1, 1)],
        ..Default::default()
    };

    let plan = Plan::compile_with(&g, &options);
    let workspace = Tensor::zeros(&[plan.workspace_bytes()], DType::U8);
    let w = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], DType::F32);
    let out = plan.run_in(&workspace, &[w.clone(), w.clone()]).unwrap();
    assert_eq!(out[0].to_vec::<f32>(), [3.0, 10.0, 21.0, 36.0]);
}

/// Elementwise steps write over an operand that dies there, but not when it is read later
/// or the step is a transpose or a dtype-changing cast.
#[test]
fn steps_write_over_operands_that_die_there() {
    let ty = |shape: &[usize]| TensorType::new(DType::F32, shape);

    let at = |plan: &Plan, name: &str| {
        let step = plan
            .steps()
            .iter()
            .find(|s| s.primitive.name() == name)
            .unwrap();

        (
            step.output.0,
            step.inputs.iter().map(|i| i.0).collect::<Vec<_>>(),
        )
    };

    // `Exp` then `second`, with `e` optionally read again after it.
    let build = |second: Primitive, read_later: bool| {
        let mut g = Graph::new();
        let x = g.input(ty(&[3, 3]));
        let e = g.apply(Exp, &[x]).unwrap();
        let y = g.apply(second, &[e]).unwrap();
        let z = g.apply(Neg, &[y]).unwrap();
        let mut outputs = vec![z];

        if read_later {
            outputs.push(g.apply(Log, &[e]).unwrap());
        }

        g.set_outputs(&outputs).unwrap();
        g
    };

    let input = [data(&[3, 3], 1)];
    let plan = check_against_reference(&build(Tanh, false), &input);
    let (out, ins) = at(&plan, "tanh");
    assert_eq!(ins, [out], "{plan}");
    let plan = check_against_reference(&build(Tanh, true), &input);
    let (out, ins) = at(&plan, "tanh");
    assert_ne!(ins, [out], "{plan}");

    let transpose = Transpose {
        permutation: vec![1, 0],
    };

    let plan = check_against_reference(&build(transpose, false), &input);
    let (out, ins) = at(&plan, "transpose");
    assert_ne!(ins, [out], "{plan}");
    let mut g = Graph::new();
    let x = g.input(ty(&[3, 3]));
    let e = g.apply(Exp, &[x]).unwrap();

    let c = g
        .apply(
            Cast {
                new_dtype: DType::BF16,
            },
            &[e],
        )
        .unwrap();

    g.set_outputs(&[c]).unwrap();
    let plan = Plan::compile(&g);
    let (out, ins) = at(&plan, "cast");
    assert_ne!(ins, [out], "{plan}");
    let (out, ins) = at(&plan, "exp");

    assert_eq!(
        (ins, out),
        (vec![Buffer::Input(0)], Buffer::Workspace(0)),
        "{plan}"
    );
}

/// Compiles `g` donating input 0 and returns the plan with the first non-workspace output buffer.
fn donating(g: &Graph) -> (Plan, Buffer) {
    let options = super::PlanOptions {
        scratch: None,
        donate: vec![0],
        donate_into: Vec::new(),
        parameters: None,
        scalars: Vec::new(),
        memory_limit: None,
        blocks: Vec::new(),
        layout: None,
    };

    let plan = Plan::compile_with(g, &options);

    let out = plan
        .steps()
        .iter()
        .map(|s| s.output.0)
        .find(|b| !matches!(b, Buffer::Workspace(_)));

    (plan, out.expect("a step writes the output"))
}

/// A donated input holds an output written elementwise over it, but not a transpose's.
#[test]
fn donated_inputs_hold_outputs_written_in_place() {
    let ty = TensorType::new(DType::F32, &[2, 3]);
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
            donate_into: Vec::new(),
            parameters: None,
            scalars: Vec::new(),
            memory_limit: None,
            blocks: Vec::new(),
            layout: None,
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

/// An input donated into one output is not reused for another.
#[test]
fn inputs_donated_into_an_output_hold_it_alone() {
    let ty = TensorType::new(DType::F32, &[2, 3]);
    let mut g = Graph::new();
    let w = g.input(ty.clone());
    let new = g.apply(Exp, &[w]).unwrap();
    let later = g.apply(Mul, &[w, w]).unwrap();
    g.set_outputs(&[later, new]).unwrap();

    let at = |options: &super::PlanOptions| {
        let plan = Plan::compile_with(&g, options);
        let written: Vec<Buffer> = plan.steps().iter().map(|s| s.output.0).collect();
        (plan, written)
    };

    let (plan, written) = at(&super::PlanOptions {
        donate: vec![0],
        ..Default::default()
    });

    assert!(written.contains(&Buffer::Input(0)), "{plan}");

    let (plan, written) = at(&super::PlanOptions {
        donate_into: vec![(0, 1)],
        ..Default::default()
    });

    assert!(!written.contains(&Buffer::Input(0)), "{plan}");
    let input = data(&[2, 3], 1);
    let expected = reference::run(&g, std::slice::from_ref(&input)).unwrap();
    let result = plan.run(std::slice::from_ref(&input)).unwrap();

    for (e, r) in expected.iter().zip(&result) {
        assert_eq!(e.to_vec::<f32>(), r.to_vec::<f32>());
    }
}

/// A donated member of a packed block is not overwritten while the block is still read.
#[test]
fn donated_inputs_in_a_block_wait_for_its_readers() {
    let ty = TensorType::new(DType::F32, &[2, 3]);
    let mut g = Graph::new();
    let w = g.input(ty.clone());
    let b = g.input(TensorType::new(DType::F32, &[4, 3]));
    let new = g.apply(Exp, &[w]).unwrap();
    let later = g.apply(Mul, &[b, b]).unwrap();
    g.set_outputs(&[later, new]).unwrap();

    let plan = |blocks: Vec<(usize, Vec<usize>)>| {
        Plan::compile_with(
            &g,
            &super::PlanOptions {
                donate_into: vec![(0, 1)],
                blocks,
                ..Default::default()
            },
        )
    };

    let alone = plan(Vec::new());

    assert!(
        alone.steps().iter().any(|s| s.output.0 == Buffer::Input(0)),
        "{alone}"
    );

    let packed = plan(vec![(1, vec![0])]);
    let steps = packed.steps();
    let writes = steps.iter().position(|s| s.output.0 == Buffer::Input(0));

    let reads_block = steps
        .iter()
        .rposition(|s| s.inputs.iter().any(|(at, _)| *at == Buffer::Input(1)));

    if let (Some(w), Some(r)) = (writes, reads_block) {
        assert!(r < w, "{packed}");
    }

    assert_eq!(writes, None, "{packed}");
}

/// A custom call mutates its operands in place unless they are read afterwards.
#[test]
fn custom_calls_mutate_in_place_where_they_can() {
    let ty = TensorType::new(DType::F32, &[4]);

    let plan_of = |read_after: bool| {
        let mut g = Graph::new();
        let x = g.input(ty.clone());
        let a = g.apply(Exp, &[x]).unwrap();
        let b = g.apply(Neg, &[x]).unwrap();

        let call = CustomCall {
            label: "test::op",
            kernel: 0,
            mutated: vec![1, 2],
            overlappable: Vec::new(),
        };

        let new_a = g.apply(call, &[x, a, b]).unwrap();

        let new_b = g
            .apply(
                FusionOutput {
                    index: 1,
                    ty: ty.clone(),
                },
                &[new_a],
            )
            .unwrap();

        let y = g.apply(Add, &[new_a, new_b]).unwrap();
        let mut outputs = vec![y];

        if read_after {
            outputs.push(g.apply(Mul, &[a, b]).unwrap());
        }

        g.set_outputs(&outputs).unwrap();
        Plan::compile(&g)
    };

    let step = |plan: &Plan| {
        let step = plan
            .steps()
            .iter()
            .find(|s| s.label == "test::op")
            .unwrap()
            .clone();

        let written: Vec<Buffer> = std::iter::once(step.output.0)
            .chain(step.extra_outputs.iter().map(|o| o.0))
            .collect();

        (written, vec![step.inputs[1].0, step.inputs[2].0])
    };

    let plan = plan_of(false);
    let (written, operands) = step(&plan);
    assert_eq!(written, operands, "{plan}");
    let plan = plan_of(true);
    let (written, operands) = step(&plan);
    assert!(written.iter().zip(&operands).all(|(w, o)| w != o), "{plan}");
}

/// Operands of an in-place concatenate are written into the result at their offsets.
#[test]
fn in_place_concatenates_are_written_by_their_operands_steps() {
    let mut g = Graph::new();
    let x = g.input(ty(DType::F32, &[2, 3]));
    let a = g.apply(Exp, &[x]).unwrap();
    let b = g.apply(Neg, &[x]).unwrap();

    let c = g
        .apply(
            Concatenate {
                dimension: 1,
                in_place: true,
            },
            &[a, b],
        )
        .unwrap();

    let y = g.apply(Mul, &[c, c]).unwrap();
    g.set_outputs(&[y]).unwrap();
    let plan = Plan::compile(&g);
    let names: Vec<&str> = plan.steps().iter().map(|s| s.primitive.name()).collect();
    assert_eq!(names, ["exp", "neg", "mul"], "{plan}");

    let [exp, neg, mul] = plan.steps() else {
        panic!("{plan}")
    };

    assert_eq!(exp.output.0, neg.output.0, "{plan}");
    assert_eq!(mul.inputs[0].0, exp.output.0, "{plan}");
    let placements = g.placements();

    let at = |v: usize| {
        placements[v]
            .clone()
            .map(|(base, at)| (base, at.offset, at.strides))
    };

    assert_eq!(at(a), Some((c, 0, vec![6, 1])));
    assert_eq!(at(b), Some((c, 3, vec![6, 1])));

    assert_eq!(
        exp.output_views[0].as_ref().map(|at| at.offset),
        Some(0),
        "{plan}"
    );

    assert_eq!(
        neg.output_views[0].as_ref().map(|at| at.offset),
        Some(3),
        "{plan}"
    );
}

/// Kernel scratch gets its own workspace region not overlapping live operands.
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
        donate_into: Vec::new(),
        parameters: None,
        scalars: Vec::new(),
        memory_limit: None,
        blocks: Vec::new(),
        layout: None,
    };

    let plan = Plan::compile_with(&g, &options);
    let reduce = &plan.steps()[1];
    let (offset, bytes) = reduce.scratch.expect("the reduction's scratch");
    assert_eq!(bytes, 100);

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

/// Running on meta tensors returns typed outputs without computing; a CPU run of meta
/// inputs fails.
#[test]
fn plans_run_on_meta_tensors_compute_nothing() {
    let g = mlp();
    let meta = |t: Tensor| t.to(crate::Device::Meta(0));

    let inputs = [
        meta(data(&[4, 8], 1)),
        meta(data(&[8, 16], 2)),
        meta(data(&[16, 3], 3)),
    ];

    let outputs = Plan::compile(&g)
        .run_on(&inputs, crate::Device::Meta(0))
        .unwrap();

    let out = &outputs[0];

    assert_eq!(
        (out.device(), out.shape(), out.dtype()),
        (crate::Device::Meta(0), &[4, 3][..], DType::F32)
    );

    assert!(
        Plan::compile(&g)
            .run_on(&inputs, crate::Device::Cpu(0))
            .is_err()
    );
}

/// Plan tests that run on MPS and compare against the reference evaluator.
#[cfg(lumen_mps_linked)]
pub(crate) mod mps {
    use super::Primitive::*;
    use super::{data, mlp};
    use crate::graph::reference;
    use crate::graph::{Graph, Primitive, TensorType};
    use crate::graph_compiler::Plan;
    use crate::tensor::dtype::dispatch_dtype;
    use crate::{DType, Device, Element, Scalar, Tensor, TensorOptions};

    /// Every dtype MPS supports.
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

    /// Whether an MPS device is present; MPS tests pass vacuously otherwise.
    pub(crate) fn available() -> bool {
        crate::device::mps::MPS.is_available()
    }

    /// Test values of `dtype`: scaled floats, or integers in about `[-100, 100]`.
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

    /// Small integers every `period`-th element and zeros elsewhere, so sums of half-precision
    /// values stay exact.
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

    /// Reads any tensor's elements as float64.
    fn as_f64(t: &Tensor) -> Vec<f64> {
        dispatch_dtype!(t.dtype(), T => t.to_vec::<T>().into_iter().map(|v| v.to_scalar().to_f64()).collect())
    }

    /// Compiles `g` for MPS and checks its outputs match the reference within dtype tolerance.
    pub(crate) fn check(g: &Graph, inputs: &[Tensor]) {
        check_within(g, inputs, 0.0);
    }

    /// `check`, allowing `slack` extra absolute error for float dtypes.
    pub(crate) fn check_within(g: &Graph, inputs: &[Tensor], slack: f64) {
        let expected = reference::run(g, inputs).unwrap();
        let on_mps: Vec<Tensor> = inputs.iter().map(|t| t.to(Device::Mps(0))).collect();
        let plan = crate::graph_compiler::compile(g, Device::Mps(0)).unwrap();
        let actual = plan.run(&on_mps).unwrap();

        for (e, a) in expected.iter().zip(&actual) {
            assert_eq!(a.device(), Device::Mps(0));
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

    /// A softmax over the last axis written as reductions, broadcasts and elementwise ops.
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

    /// Checks a graph of the single node `p` on MPS.
    fn check_node(p: Primitive, types: &[TensorType]) {
        check_node_within(p, types, 0.0);
    }

    /// `check_node`, allowing `slack` extra error; half-precision reductions and contractions
    /// use exact values.
    fn check_node_within(p: Primitive, types: &[TensorType], slack: f64) {
        let mut g = Graph::new();
        let vars: Vec<_> = types.iter().map(|ty| g.input(ty.clone())).collect();
        let out = g.apply(p.clone(), &vars).unwrap();
        g.set_outputs(&[out]).unwrap();

        let count: Option<usize> = match &p {
            ReduceSum { axes, .. } => Some(axes.iter().map(|&d| types[0].shape[d]).product()),
            DotGeneral {
                lhs_contracting, ..
            } => Some(lhs_contracting.iter().map(|&d| types[0].shape[d]).product()),
            _ => None,
        };

        // Half-precision sums only stay exact with sparse small integers.
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

    /// Shorthand for `TensorType::new`.
    fn ty(dtype: DType, shape: &[usize]) -> TensorType {
        TensorType::new(dtype, shape)
    }

    /// Checks a constant node on MPS, alone and fed into a `Max`.
    fn check_constant(p: Primitive) {
        let out = p.infer(&[]).unwrap();
        let mut g = Graph::new();
        let x = g.input(out.clone());
        let c = g.apply(p, &[]).unwrap();
        let m = g.apply(Max, &[c, x]).unwrap();
        g.set_outputs(&[c, m]).unwrap();
        check(&g, &[values(out.dtype, &out.shape, 7)]);
    }

    /// Elementwise ops across dtypes; bfloat16 transcendentals are rejected.
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

            if dtype.is_float() && dtype != DType::BF16 {
                for p in [Exp, Log, Sqrt, Tanh, Logistic] {
                    check_node(p, std::slice::from_ref(&x));
                }
            }

            if dtype == DType::BF16 {
                for p in [Exp, Log, Sqrt, Tanh, Logistic] {
                    let mut g = Graph::new();
                    let v = g.input(x.clone());
                    let out = g.apply(p, &[v]).unwrap();
                    g.set_outputs(&[out]).unwrap();
                    let e = crate::graph_compiler::compile(&g, Device::Mps(0)).unwrap_err();
                    assert!(e.contains("no bfloat16 kernel on MPS"), "{e}");
                }
            }
        }
    }

    /// Casts between every dtype pair, including saturating and NaN float casts.
    #[test]
    fn conversions() {
        if !available() {
            return;
        }

        for from in DTYPES {
            for to in DTYPES {
                check_node(Cast { new_dtype: to }, &[ty(from, &[64])]);
            }
        }

        let f = Tensor::from_slice(&[300.0f32, -1e10, f32::NAN, 2.9], DType::F32);

        for to in [DType::I8, DType::U8, DType::I32, DType::U64, DType::Bool] {
            let mut g = Graph::new();
            let x = g.input(ty(DType::F32, &[4]));
            let y = g.apply(Cast { new_dtype: to }, &[x]).unwrap();
            g.set_outputs(&[y]).unwrap();
            check(&g, std::slice::from_ref(&f));
        }
    }

    /// Half-precision matmuls accumulate in float32 and write either dtype.
    #[test]
    fn widened_contractions() {
        if !available() {
            return;
        }

        for dtype in [DType::F16, DType::BF16] {
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

    /// Reductions over large and degenerate shapes across dtypes.
    #[test]
    fn large_reductions() {
        if !available() {
            return;
        }

        let cases: [(&[usize], Vec<usize>); 13] = [
            (&[3, 50_000], vec![1]),
            (&[50_000, 3], vec![0]),
            (&[300, 700], vec![0, 1]),
            (&[4, 5_000, 3], vec![1]),
            (&[2_000, 300], vec![1]),
            (&[300, 2_000], vec![0]),
            (&[40, 7, 300], vec![0, 2]),
            (&[400, 1, 300], vec![0, 2]),
            (&[3, 1, 50_000, 1], vec![1, 2]),
            (&[1, 50_000, 1, 3], vec![0, 1, 2]),
            (&[40, 1, 7, 1, 300], vec![0, 1, 4]),
            (&[300, 1, 7], vec![1]),
            (&[1, 1], vec![0, 1]),
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
                    let count: usize = axes.iter().map(|&d| shape[d]).product();
                    let slack = 1e-6 * 4.0 * count as f64;

                    let sum = ReduceSum {
                        axes: axes.clone(),
                        accum_dtype: dtype,
                    };

                    check_node_within(sum, std::slice::from_ref(&x), slack);

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

    /// Small reductions and batched or multi-axis contractions across dtypes.
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

            if dtype.is_float() {
                check_node(
                    dot(1, 0, vec![]),
                    &[ty(dtype, &[500, 20]), ty(dtype, &[20, 700])],
                );
            }
        }
    }

    /// Broadcasts, transposes, slices, softmax, concatenates and constants across dtypes.
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
                    view: false,
                };

                check_node(slice, &[ty(dtype, &[4, 5, 6])]);
            }

            if matches!(dtype, DType::F16 | DType::F32) {
                for shape in [vec![4, 7], vec![3, 1000], vec![2, 3, 1], vec![0, 5]] {
                    let g = written_softmax(dtype, &shape);
                    check(&g, &[values(dtype, &shape, 1)]);
                }
            }

            for dimension in 0..3 {
                let shape = |n| {
                    let mut shape = vec![3, 4, 5];
                    shape[dimension] = n;
                    ty(dtype, &shape)
                };

                check_node(
                    Concatenate {
                        dimension,
                        in_place: false,
                    },
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

            let mut g = Graph::new();
            let x = g.input(ty(dtype, &[2, 3]));
            let flat = g.apply(Reshape { new_sizes: vec![6] }, &[x]).unwrap();
            g.set_outputs(&[flat, x]).unwrap();
            check(&g, &[values(dtype, &[2, 3], 9)]);
        }
    }

    /// The MLP plan runs on MPS.
    #[test]
    fn mlp_plan() {
        if !available() {
            return;
        }

        let inputs = [data(&[4, 8], 1), data(&[8, 16], 2), data(&[16, 3], 3)];
        check(&mlp(), &inputs);
    }

    /// Forward and reverse cumulative sums along each axis on MPS.
    #[test]
    fn cumsum_runs_on_mps() {
        if !available() {
            return;
        }

        let cases: [(&[usize], usize); 6] = [
            (&[4, 1000], 1),
            (&[1, 70_000], 1),
            (&[3000, 3], 0),
            (&[3, 50, 7], 1),
            (&[2, 9000, 2], 1),
            (&[8, 3], 1),
        ];

        let dtypes = DTYPES[1..9].iter().map(|&d| (d, d)).chain([
            (DType::F32, DType::F32),
            (DType::F16, DType::F32),
            (DType::BF16, DType::F32),
        ]);

        for (dtype, accum_dtype) in dtypes {
            for &(shape, axis) in &cases {
                for reverse in [false, true] {
                    let mut g = Graph::new();
                    let x = g.input(ty(dtype, shape));

                    let scan = Cumsum {
                        axis,
                        reverse,
                        accum_dtype,
                    };

                    let y = g.apply(scan, &[x]).unwrap();
                    g.set_outputs(&[y]).unwrap();
                    let slack = shape[axis] as f64 * 2.0 * f64::from(f32::EPSILON);
                    check_within(&g, &[values(dtype, shape, 1)], slack);
                }
            }
        }
    }

    /// Random bits on MPS, alone and fused, match the reference, also with scalar inputs.
    #[test]
    fn random_bits_run_on_mps() {
        if !available() {
            return;
        }

        for shape in [&[7][..], &[1000, 33]] {
            for fused in [false, true] {
                let mut g = Graph::new();
                let seed = g.input(ty(DType::U64, &[]));
                let start = g.input(ty(DType::U64, &[]));

                let bits = RandomBits {
                    shape: shape.to_vec(),
                    offset: 3,
                };

                let r = g.apply(bits, &[seed, start]).unwrap();

                let outputs = match fused {
                    false => vec![r],
                    true => {
                        let to_f32 = Cast {
                            new_dtype: DType::F32,
                        };

                        let u = g.apply(to_f32, &[r]).unwrap();

                        let half = Full {
                            shape: shape.to_vec(),
                            fill_value: Scalar::Int(1 << 31),
                            dtype: DType::U32,
                        };

                        let half = g.apply(half, &[]).unwrap();
                        let kept = g.apply(Lt, &[r, half]).unwrap();
                        vec![u, kept]
                    }
                };

                g.set_outputs(&outputs).unwrap();
                let scalar = |v: u64| Tensor::from_slice(&[v], DType::U64).reshape(&[]);
                let inputs = [scalar(0xDEAD_BEEF_0123_4567), scalar(0xFFFF_FFF0)];
                check(&g, &inputs);

                if fused {
                    let expected = reference::run(&g, &inputs).unwrap();

                    let options = crate::graph_compiler::Options {
                        scalars: vec![true, true],
                        ..Default::default()
                    };

                    let plan =
                        crate::graph_compiler::compile_with(&g, Device::Mps(0), &options).unwrap();

                    let on_mps: Vec<Tensor> = inputs.iter().map(|t| t.to(Device::Mps(0))).collect();
                    let actual = plan.run_on(&on_mps, Device::Mps(0)).unwrap();

                    for (e, a) in expected.iter().zip(&actual) {
                        assert_eq!(as_f64(e), as_f64(a), "{plan}");
                    }
                }
            }
        }
    }

    /// Dynamic slices and updates with in-range and clamped starts; a donated update writes in place.
    #[test]
    fn dynamic_slices_run_on_mps() {
        if !available() {
            return;
        }

        let index = |dtype: DType, i: i64| match dtype {
            DType::I32 => Tensor::from_slice(&[i as i32], dtype).reshape(&[]),
            _ => Tensor::from_slice(&[i], dtype).reshape(&[]),
        };

        for &dtype in &DTYPES {
            for index_dtype in [DType::I32, DType::I64] {
                for starts in [[1, 4, 0], [3, 9, 2], [-2, -1, 5]] {
                    let mut g = Graph::new();
                    let x = g.input(ty(dtype, &[4, 10, 3]));
                    let u = g.input(ty(dtype, &[2, 3, 3]));
                    let idx: Vec<_> = (0..3).map(|_| g.input(ty(index_dtype, &[]))).collect();
                    let slice_sizes = vec![2, 3, 3];

                    let ds = g
                        .apply(DynamicSlice { slice_sizes }, &[&[x][..], &idx].concat())
                        .unwrap();

                    let dus = g
                        .apply(DynamicUpdateSlice, &[&[x, u][..], &idx].concat())
                        .unwrap();

                    g.set_outputs(&[ds, dus]).unwrap();

                    let mut inputs =
                        vec![values(dtype, &[4, 10, 3], 1), values(dtype, &[2, 3, 3], 2)];

                    inputs.extend(starts.map(|s| index(index_dtype, s)));
                    check(&g, &inputs);
                }
            }
        }

        let mut g = Graph::new();
        let x = g.input(ty(DType::F32, &[4, 6]));
        let u = g.input(ty(DType::F32, &[1, 6]));
        let i = g.input(ty(DType::I32, &[]));
        let y = g.apply(DynamicUpdateSlice, &[x, u, i, i]).unwrap();
        g.set_outputs(&[y]).unwrap();

        let options = crate::graph_compiler::Options {
            donate: vec![0],
            ..Default::default()
        };

        let plan = crate::graph_compiler::compile_with(&g, Device::Mps(0), &options).unwrap();

        assert!(
            plan.to_string()
                .contains("in0:f32[4,6] = dynamic_update_slice in0"),
            "{plan}"
        );

        let x = values(DType::F32, &[4, 6], 1).to(Device::Mps(0));

        let want = reference::run(
            &g,
            &[
                x.clone(),
                values(DType::F32, &[1, 6], 2),
                index(DType::I32, 2),
            ],
        )
        .unwrap();

        let got = plan
            .run(&[
                x,
                values(DType::F32, &[1, 6], 2).to(Device::Mps(0)),
                index(DType::I32, 2).to(Device::Mps(0)),
            ])
            .unwrap();

        assert_eq!(
            want[0].to_vec::<f32>(),
            got[0].to(Device::Cpu(0)).to_vec::<f32>()
        );
    }

    /// float64 and inputs on mixed devices are rejected on MPS.
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
            TensorOptions::new()
                .dtype(DType::F64)
                .device(Device::Mps(0)),
        );

        let err = Plan::compile(&g).run(&[x]).unwrap_err();
        assert!(err.contains("float64"), "{err}");

        let mut g = Graph::new();
        let a = g.input(ty(DType::F32, &[2]));
        let b = g.input(ty(DType::F32, &[2]));
        let c = g.apply(Add, &[a, b]).unwrap();
        g.set_outputs(&[c]).unwrap();

        let (a, b) = (
            Tensor::zeros(&[2], Device::Mps(0)),
            Tensor::zeros(&[2], DType::F32),
        );

        assert!(Plan::compile(&g).run(&[a, b]).is_err());
    }
}
