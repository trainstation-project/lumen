use crate::device::tpu::{ONE_AT_A_TIME, TPU};
use crate::graph::Primitive::DotGeneral;
use crate::graph::{Graph, Primitive, TensorType};
use crate::graph_compiler::plan_tests::data;
use crate::ops::reference;
use crate::{DType, Device, Tensor};

#[test]
fn base64_encodes_with_padding() {
    assert_eq!(super::base64(b""), "");
    assert_eq!(super::base64(b"f"), "Zg==");
    assert_eq!(super::base64(b"fo"), "Zm8=");
    assert_eq!(super::base64(b"foo"), "Zm9v");
    assert_eq!(super::base64(b"foobar\xff"), "Zm9vYmFy/w==");
}

macro_rules! require_tpu {
    () => {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        if !TPU.is_available() {
            eprintln!("no TPU device, skipping");
            return;
        }
    };
}

/// `shape` of `dtype`, values in [-2, 2) from `seed`.
fn values(dtype: DType, shape: &[usize], seed: u64) -> Tensor {
    data(shape, seed).to_dtype(dtype).unwrap()
}

fn floats(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().to_vec::<f32>()
}

#[test]
fn copies_round_trip_without_touching_their_neighbours() {
    require_tpu!();
    let tpu = Device::Tpu(0);
    // Sizes of a row (512 bytes), less and more: each copy writes its own
    // rows, padding included, and nothing past them.
    let shapes: [&[usize]; 4] = [&[1], &[128], &[129], &[10, 100]];
    let hosts: Vec<Tensor> = shapes
        .iter()
        .enumerate()
        .map(|(i, shape)| values(DType::F32, shape, i as u64))
        .collect();
    let devices: Vec<Tensor> = hosts.iter().map(|t| t.to(tpu)).collect();
    let bf16 = values(DType::BF16, &[3, 5], 9);
    assert_eq!(floats(&bf16.to(tpu).to(Device::Cpu(0))), floats(&bf16));
    for (host, device) in hosts.iter().zip(&devices) {
        assert_eq!(device.device(), tpu);
        assert_eq!(
            device.to(Device::Cpu(0)).to_vec::<f32>(),
            host.to_vec::<f32>()
        );
    }
}

fn ty(dtype: DType, shape: &[usize]) -> TensorType {
    TensorType::new(dtype, shape)
}

fn dot(
    lhs_contracting: &[usize],
    rhs_contracting: &[usize],
    batch: &[usize],
    output_dtype: DType,
) -> Primitive {
    DotGeneral {
        lhs_contracting: lhs_contracting.to_vec(),
        rhs_contracting: rhs_contracting.to_vec(),
        lhs_batch: batch.to_vec(),
        rhs_batch: batch.to_vec(),
        accum_dtype: DType::F32,
        output_dtype,
    }
}

/// The one-node graph of `p` on `lhs` and `rhs`, run on tpu:0.
fn run(
    p: Primitive,
    lhs: TensorType,
    rhs: TensorType,
) -> (Vec<Tensor>, Result<Vec<Tensor>, String>) {
    let mut g = Graph::new();
    let a = g.input(lhs.clone());
    let b = g.input(rhs.clone());
    let y = g.apply(p, &[a, b]).unwrap();
    g.set_outputs(&[y]).unwrap();
    let inputs = [
        values(lhs.dtype, &lhs.shape, 1),
        values(rhs.dtype, &rhs.shape, 2),
    ];
    let expected = reference::run(&g, &inputs).unwrap();
    let on_tpu: Vec<Tensor> = inputs.iter().map(|t| t.to(Device::Tpu(0))).collect();
    let plan = crate::graph_compiler::compile(&g, Device::Tpu(0)).unwrap();
    (expected, plan.run(&on_tpu))
}

/// `p` on tpu:0 agrees with the reference within `tol` of its largest
/// output: float32's rounding, or one bfloat16 rounding.
fn check(p: Primitive, lhs: TensorType, rhs: TensorType) {
    let (expected, actual) = run(p, lhs, rhs);
    let (e, a) = (&expected[0], &actual.unwrap()[0]);
    assert_eq!(a.device(), Device::Tpu(0));
    assert_eq!((a.dtype(), a.shape()), (e.dtype(), e.shape()));
    let tol = match e.dtype() {
        DType::BF16 => 1e-2,
        _ => 1e-5,
    };
    let (e, a) = (floats(e), floats(&a.to(Device::Cpu(0))));
    let max = e.iter().fold(0f32, |m, x| m.max(x.abs()));
    for (i, (x, y)) in e.iter().zip(&a).enumerate() {
        assert!(
            (x - y).abs() <= tol * max,
            "element {i}: expected {x}, got {y}"
        );
    }
}

#[test]
fn float32_dots_run_on_the_matrix_unit() {
    require_tpu!();
    let f32 = DType::F32;
    // One block; K in two blocks.
    check(
        dot(&[1], &[0], &[], f32),
        ty(f32, &[256, 512]),
        ty(f32, &[512, 384]),
    );
    check(
        dot(&[1], &[0], &[], f32),
        ty(f32, &[64, 1024]),
        ty(f32, &[1024, 128]),
    );
    // A batch and a transposed rhs (`x @ w.T`); one row.
    check(
        dot(&[2], &[2], &[0], f32),
        ty(f32, &[2, 64, 256]),
        ty(f32, &[2, 384, 256]),
    );
    check(
        dot(&[1], &[0], &[], f32),
        ty(f32, &[1, 256]),
        ty(f32, &[256, 256]),
    );
    // The lhs's free dimensions are one M.
    check(
        dot(&[2], &[0], &[], f32),
        ty(f32, &[2, 3, 256]),
        ty(f32, &[256, 128]),
    );
}

#[test]
fn bfloat16_dots_split_their_words_by_parity() {
    require_tpu!();
    let (bf16, f32) = (DType::BF16, DType::F32);
    check(
        dot(&[1], &[0], &[], bf16),
        ty(bf16, &[128, 512]),
        ty(bf16, &[512, 512]),
    );
    check(
        dot(&[1], &[1], &[], bf16),
        ty(bf16, &[8, 1024]),
        ty(bf16, &[512, 1024]),
    );
    check(
        dot(&[1], &[1], &[], f32),
        ty(bf16, &[8, 1024]),
        ty(bf16, &[512, 1024]),
    );
    check(
        dot(&[2], &[2], &[0], bf16),
        ty(bf16, &[2, 16, 256]),
        ty(bf16, &[2, 256, 256]),
    );
}

#[test]
fn unsupported_dots_say_why() {
    require_tpu!();
    let f32 = DType::F32;
    let (_, actual) = run(
        dot(&[1], &[0], &[], f32),
        ty(f32, &[4, 8]),
        ty(f32, &[8, 16]),
    );
    let error = actual.unwrap_err();
    assert!(
        error.ends_with("needs N (16) a multiple of 128 and K (8) a multiple of 128"),
        "{error}"
    );
}
