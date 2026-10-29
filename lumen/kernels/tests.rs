//! Tests of op dispatch between built-in and registered kernels, of `vector_size`, and of the CPU reference
//! interpreter's edge cases.
use super::*;
use crate::graph::{BinaryOp, Graph, Primitive, TensorType, UnaryOp};
use crate::tensor::dtype::f16;
use crate::{DType, Tensor};

/// Test kernel type returning which registry it came from.
type Kernel = fn() -> &'static str;

/// Stands in for a built-in kernel.
fn builtin() -> &'static str {
    "static"
}

/// Stands in for a runtime-registered kernel.
fn registered() -> &'static str {
    "dynamic"
}

/// Stands in for a second registration replacing the first.
fn replacement() -> &'static str {
    "replacement"
}

/// Built-in lookup with a kernel only for the CPU.
fn cpu_only(key: DispatchKey) -> Option<Kernel> {
    match key {
        DispatchKey::Cpu => Some(builtin),
        _ => None,
    }
}

/// A device maps to the dispatch key of its kind, whatever its index.
#[test]
fn dispatch_keys_follow_the_device() {
    assert_eq!(DispatchKey::of(Device::Cpu(0)), DispatchKey::Cpu);
    assert_eq!(DispatchKey::of(Device::Cuda(3)), DispatchKey::Cuda);
    assert_eq!(DispatchKey::of(Device::Mps(0)), DispatchKey::Mps);
    assert_eq!(DispatchKey::of(Device::Tpu(2)), DispatchKey::Tpu);
    assert_eq!(DispatchKey::of(Device::Meta(0)), DispatchKey::Meta);
}

/// A built-in kernel wins and registering over it is rejected.
#[test]
fn the_static_registry_is_used_first() {
    static OP: Op<Kernel> = Op::new("test::static_first", cpu_only);
    assert_eq!(OP.dispatch(Device::Cpu(0))(), "static");
    let err = OP.register(DispatchKey::Cpu, registered).unwrap_err();
    assert!(
        err.contains("test::static_first") && err.contains("Cpu"),
        "{err}"
    );
    assert_eq!(
        OP.dispatch(Device::Cpu(0))(),
        "static",
        "still the built-in"
    );
}

/// Registered kernels cover keys without a built-in, and re-registering replaces them.
#[test]
fn the_dynamic_registry_fills_in_missing_keys() {
    static OP: Op<Kernel> = Op::new("test::dynamic_fallback", cpu_only);
    assert!(OP.kernel(DispatchKey::Mps).is_none());
    OP.register(DispatchKey::Mps, registered).unwrap();
    assert_eq!(OP.dispatch(Device::Mps(0))(), "dynamic");
    OP.register(DispatchKey::Mps, replacement).unwrap();
    assert_eq!(OP.dispatch(Device::Mps(0))(), "replacement");
    assert!(OP.kernel(DispatchKey::Cuda).is_none(), "per key");
}

/// Registrations are keyed by op name, so one op's kernel doesn't leak to another.
#[test]
fn dynamic_kernels_are_per_op() {
    static A: Op<Kernel> = Op::new("test::per_op_a", cpu_only);
    static B: Op<Kernel> = Op::new("test::per_op_b", cpu_only);
    A.register(DispatchKey::Cuda, registered).unwrap();
    assert!(B.kernel(DispatchKey::Cuda).is_none());
}

/// Dispatching a key with no kernel panics naming the op, key and device.
#[test]
#[should_panic(expected = "test::missing has no kernel for Cuda (cuda:1)")]
fn a_missing_kernel_panics_with_the_op_and_key() {
    static OP: Op<Kernel> = Op::new("test::missing", cpu_only);
    OP.dispatch(Device::Cuda(1));
}

/// The vector width shrinks until the size, strides and address all align to it.
#[test]
fn the_vector_size_is_the_widest_aligned_store() {
    assert_eq!(vector_size(&[1024], &[1], 4, 256), 4);
    assert_eq!(vector_size(&[1024], &[1], 2, 256), 8);
    assert_eq!(vector_size(&[1024], &[1], 1, 256), 16);
    assert_eq!(vector_size(&[1024], &[1], 8, 256), 2);
    assert_eq!(vector_size(&[1024], &[1], 4, 256 + 8), 2);
    assert_eq!(vector_size(&[1024], &[1], 4, 256 + 4), 1);
    assert_eq!(vector_size(&[6], &[1], 4, 256), 2);
    assert_eq!(vector_size(&[1001], &[1], 4, 256), 1);
    assert_eq!(vector_size(&[1024], &[2], 4, 256), 1);
    assert_eq!(vector_size(&[4, 16], &[1, 8], 4, 256), 4);
    assert_eq!(vector_size(&[4, 16], &[1, 6], 4, 256), 2);
    assert_eq!(vector_size(&[], &[], 4, 256), 1);
}

/// Runs a one-node graph of `p` over inputs of `types` on the CPU reference, returning its output.
fn reference_of(p: Primitive, types: &[TensorType], inputs: &[Tensor]) -> Tensor {
    let mut g = Graph::new();
    let vars: Vec<_> = types.iter().map(|t| g.input(t.clone())).collect();
    let out = g.apply(p, &vars).unwrap();
    g.set_outputs(&[out]).unwrap();
    reference::run(&g, inputs).unwrap().remove(0)
}

/// `nextafter` steps one ulp of the operands' own dtype: float16's and float64's bit patterns, towards
/// either side, from zero to the smallest subnormal of the target's sign, and to `y` itself when equal.
#[test]
fn reference_nextafter_steps_one_ulp_of_its_dtype() {
    let x = [1.0f64, 1.0, 0.0, 0.0, -2.0, 3.0];
    let y = [2.0f64, 0.0, 1.0, -1.0, -3.0, 3.0];
    let ty = |d| TensorType::new(d, &[6]);
    let next = |d: DType| {
        let args = [Tensor::from_slice(&x, d), Tensor::from_slice(&y, d)];
        reference_of(
            Primitive::Binary(BinaryOp::Nextafter),
            &[ty(d), ty(d)],
            &args,
        )
    };
    let half: Vec<u16> = next(DType::F16)
        .to_vec::<f16>()
        .iter()
        .map(|h| h.to_bits())
        .collect();
    assert_eq!(half, [0x3C01, 0x3BFF, 0x0001, 0x8001, 0xC001, 0x4200]);
    let double: Vec<u64> = next(DType::F64)
        .to_vec::<f64>()
        .iter()
        .map(|d| d.to_bits())
        .collect();
    let want: Vec<u64> = vec![
        1f64.to_bits() + 1,
        1f64.to_bits() - 1,
        1,
        1 << 63 | 1,
        (-2f64).to_bits() + 1,
        3f64.to_bits(),
    ];
    assert_eq!(double, want);
}

/// `reduce_precision` to float16's widths flushes values below its smallest normal to a zero of their sign
/// and sends values past its largest to an infinity of their sign.
#[test]
fn reference_reduce_precision_flushes_and_overflows() {
    let x = [1e-6f32, -1e-6, 1e6, -1e6, 1.5];
    let p = Primitive::Unary(UnaryOp::ReducePrecision {
        exponent_bits: 5,
        mantissa_bits: 10,
    });
    let ty = TensorType::new(DType::F32, &[5]);
    let got = reference_of(p, &[ty], &[Tensor::from_slice(&x, DType::F32)]).to_vec::<f32>();
    let bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
    let want = [0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, 1.5];
    assert_eq!(bits, want.map(f32::to_bits));
}

/// Empty results: a dynamic slice of no elements and a dot of an empty output are empty, and a dot
/// contracting an empty axis is its zeros.
#[test]
fn reference_empty_slices_and_contractions() {
    let f32s = |shape: &[usize]| TensorType::new(DType::F32, shape);
    let scalar = TensorType::new(DType::I32, &[]);
    let start = || Tensor::from_slice(&[1i32], DType::I32).reshape(&[]);
    let x = Tensor::from_slice(&[0.0f32; 12], DType::F32).reshape(&[3, 4]);
    let slice = Primitive::DynamicSlice {
        slice_sizes: vec![0, 2],
    };
    let types = [f32s(&[3, 4]), scalar.clone(), scalar];
    let got = reference_of(slice, &types, &[x, start(), start()]);
    assert_eq!(got.shape(), [0, 2]);
    let dot = Primitive::DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };
    let empty = |shape: &[usize]| Tensor::from_slice::<f32>(&[], DType::F32).reshape(shape);
    let zeros = reference_of(
        dot.clone(),
        &[f32s(&[2, 0]), f32s(&[0, 3])],
        &[empty(&[2, 0]), empty(&[0, 3])],
    );
    assert_eq!(
        (zeros.shape(), zeros.to_vec::<f32>()),
        (&[2, 3][..], vec![0.0; 6])
    );
    let none = reference_of(
        dot,
        &[f32s(&[0, 2]), f32s(&[2, 3])],
        &[
            empty(&[0, 2]),
            Tensor::from_slice(&[1.0f32; 6], DType::F32).reshape(&[2, 3]),
        ],
    );
    assert_eq!(none.shape(), [0, 3]);
}

/// `linear_cross_entropy` of rows `h`, weights `w` and classes: each row's `logsumexp(h w^T) - logit[class]`
/// followed by each row's `logsumexp`, classes clamped into range.
#[test]
fn reference_linear_cross_entropy_is_losses_then_logsumexps() {
    let (h, w) = ([1.0f32, 0.0, 0.5, -1.0], [1.0f32, 2.0, 0.0, 1.0, -1.0, 0.5]);
    let classes = [2i64, 7];
    let types = [
        TensorType::new(DType::F32, &[2, 2]),
        TensorType::new(DType::F32, &[3, 2]),
        TensorType::new(DType::I64, &[2]),
    ];
    let args = [
        Tensor::from_slice(&h, DType::F32).reshape(&[2, 2]),
        Tensor::from_slice(&w, DType::F32).reshape(&[3, 2]),
        Tensor::from_slice(&classes, DType::I64),
    ];
    let got = reference_of(Primitive::LinearCrossEntropy, &types, &args);
    assert_eq!(got.shape(), [2, 2]);
    let got = got.to_vec::<f32>();
    for (i, &class) in [2usize, 2].iter().enumerate() {
        let logits: Vec<f64> = (0..3)
            .map(|j| (0..2).map(|k| f64::from(h[i * 2 + k] * w[j * 2 + k])).sum())
            .collect();
        let lse = logits.iter().map(|x| x.exp()).sum::<f64>().ln();
        assert!(
            (f64::from(got[i]) - (lse - logits[class])).abs() < 1e-6,
            "{got:?}"
        );
        assert!((f64::from(got[2 + i]) - lse).abs() < 1e-6, "{got:?}");
    }
}
