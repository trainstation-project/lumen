use super::mosaic::lower;
use super::{Kernel, Op, Space, Type};
use crate::graph::{Primitive, TensorType};
use crate::{DType, Scalar};

fn i32_scalar() -> TensorType {
    TensorType::new(DType::I32, &[])
}

fn constant(k: &mut Kernel, v: i64) -> usize {
    let full = Primitive::Full {
        shape: vec![],
        fill_value: Scalar::Int(v),
        dtype: DType::I32,
    };
    k.apply(full, &[]).unwrap()
}

#[test]
fn a_copy_kernel_lowers_to_a_dma_at_a_scalar_row() {
    // The host-to-TPU copy: host rows to the memory at scalar 0's row.
    let mut k = Kernel::new(1, Some(4), 64);
    let [scalars, host, _, result] = k.params()[..] else {
        unreachable!()
    };
    let sem = k.alloc(Type::Semaphore(vec![])).unwrap();
    let zero = constant(&mut k, 0);
    let row = k.load(scalars, &[zero]).unwrap();
    let to = k
        .slice(
            result,
            vec![None, Some(0), Some(0)],
            &[row],
            vec![4, 1, 128],
            vec![false; 3],
        )
        .unwrap();
    k.dma(host, to, sem, false).unwrap();
    k.dma(host, to, sem, true).unwrap();
    let text = lower(&k, "copy").unwrap();
    for expected in [
        "{stable_mosaic.version = 17 : i64}",
        "^bb0(%arg0: memref<1xi32, #tpu.memory_space<smem>>, %arg1: memref<4x1x128xi32, \
         #tpu.memory_space<any>>, %arg2: memref<64x1x128xi32, #tpu.memory_space<any>>, %arg3: \
         memref<64x1x128xi32, #tpu.memory_space<any>>, %arg4: memref<!tpu.dma_semaphore, \
         #tpu.memory_space<semaphore_mem>>):",
        "\"stable_mosaic.memref.load\"(%arg0, %t1) : (memref<1xi32, #tpu.memory_space<smem>>, index) -> i32",
        "\"stable_mosaic.tpu.memref_slice\"(%arg3, %t2, %t3, %t4) {operandSegmentSizes = array<i32: 1, 3, 0>}",
        "\"stable_mosaic.tpu.enqueue_dma\"(%arg1, %t5, %arg4)",
        "\"stable_mosaic.tpu.wait_dma2\"(%arg4, %arg1, %t5)",
        "scratch_operands = 1 : i64, sym_name = \"copy\"",
    ] {
        assert!(text.contains(expected), "{expected}\nin\n{text}");
    }
}

#[test]
fn a_loop_accumulates_a_matmul_in_vmem() {
    // acc = 0; for i in 0..4: if i == 0: acc = 0; acc += x @ y.
    let mut k = Kernel::new(1, None, 64);
    let f32 = |shape: &[usize]| TensorType::new(DType::F32, shape);
    let x = k.alloc(Type::Ref(f32(&[8, 128]), Space::Vmem)).unwrap();
    let y = k.alloc(Type::Ref(f32(&[256, 128]), Space::Vmem)).unwrap();
    let acc = k.alloc(Type::Ref(f32(&[8, 256]), Space::Vmem)).unwrap();
    let (lo, hi) = (constant(&mut k, 0), constant(&mut k, 4));
    let zero = constant(&mut k, 0);
    let (i, carried) = k.begin_for(lo, hi, &[zero]).unwrap();
    let first = k.apply(Primitive::Eq, &[i, carried[0]]).unwrap();
    k.begin_if(first).unwrap();
    let zeros = Primitive::Full {
        shape: vec![8, 256],
        fill_value: Scalar::Float(0.0),
        dtype: DType::F32,
    };
    let zeros = k.apply(zeros, &[]).unwrap();
    k.store(acc, zeros, &[]).unwrap();
    k.end_if().unwrap();
    let (xv, yv, av) = (
        k.load(x, &[]).unwrap(),
        k.load(y, &[]).unwrap(),
        k.load(acc, &[]).unwrap(),
    );
    // x @ y.T: y is [n, k].
    let dot = Primitive::DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![1],
        lhs_batch: vec![],
        rhs_batch: vec![],
        accum_dtype: DType::F32,
        output_dtype: DType::F32,
    };
    let product = k.apply(dot, &[xv, yv]).unwrap();
    let sum = k.apply(Primitive::Add, &[av, product]).unwrap();
    k.store(acc, sum, &[]).unwrap();
    let out = k.end_for(&carried).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(*k.type_of(out[0]), Type::Value(i32_scalar()));
    let text = lower(&k, "matmul").unwrap();
    for expected in [
        "= \"stable_mosaic.scf.for\"(",
        "^bb0(%v",
        "\"stable_mosaic.scf.if\"(",
        "}, {\n",
        "\"stable_mosaic.tpu.vector_store\"(",
        "#tpu.dot_dimension_numbers<[1], [1], [0], [0], [0, 0, 1, 0], [], []>, precision = \
         #tpu.contract_precision<fp32>",
        "\"stable_mosaic.arith.addf\"(",
        "\"stable_mosaic.scf.yield\"(%v",
        ": (i32, i32, i32, i32) -> i32",
    ] {
        assert!(text.contains(expected), "{expected}\nin\n{text}");
    }
}

#[test]
fn bit_ops_unpack_a_bfloat16_word() {
    let mut k = Kernel::new(1, None, 64);
    let words = k
        .alloc(Type::Ref(
            TensorType::new(DType::U32, &[8, 128]),
            Space::Vmem,
        ))
        .unwrap();
    let w = k.load(words, &[]).unwrap();
    let sixteen = Primitive::Full {
        shape: vec![8, 128],
        fill_value: Scalar::Int(16),
        dtype: DType::U32,
    };
    let sixteen = k.apply(sixteen, &[]).unwrap();
    let low = k.bits(Op::ShiftLeft, &[w, sixteen]).unwrap();
    let float = k.bits(Op::Bitcast(DType::F32), &[low]).unwrap();
    let narrow = k
        .apply(
            Primitive::Cast {
                new_dtype: DType::BF16,
            },
            &[float],
        )
        .unwrap();
    assert_eq!(
        *k.type_of(narrow),
        Type::Value(TensorType::new(DType::BF16, &[8, 128]))
    );
    let text = lower(&k, "unpack").unwrap();
    for expected in [
        "{value = dense<16> : vector<8x128xi32>}",
        "\"stable_mosaic.arith.shli\"(",
        "\"stable_mosaic.tpu.bitcast\"(",
        "\"stable_mosaic.arith.truncf\"(",
    ] {
        assert!(text.contains(expected), "{expected}\nin\n{text}");
    }
}

#[test]
fn constants_are_exact() {
    let mut k = Kernel::new(1, None, 64);
    let full = |v, dtype| Primitive::Full {
        shape: vec![],
        fill_value: v,
        dtype,
    };
    k.apply(full(Scalar::Int(0xFFFF0000), DType::U32), &[])
        .unwrap();
    k.apply(full(Scalar::Float(0.1), DType::F32), &[]).unwrap();
    k.apply(full(Scalar::Float(f64::NEG_INFINITY), DType::F32), &[])
        .unwrap();
    let text = lower(&k, "constants").unwrap();
    assert!(text.contains("{value = -65536 : i32}"), "{text}");
    assert!(
        text.contains("{value = 1.00000000000000006e-1 : f32}"),
        "{text}"
    );
    assert!(text.contains("{value = 0xFF800000 : f32}"), "{text}");
}

#[test]
fn kernels_refuse_what_tpus_cannot_do() {
    let mut k = Kernel::new(1, None, 64);
    let [scalars, memory, _] = k.params()[..] else {
        unreachable!()
    };
    // Loads read VMEM whole, or an SMEM element.
    assert!(k.load(memory, &[]).is_err());
    assert!(k.load(scalars, &[]).is_err());
    // Scratch is VMEM or semaphores.
    let hbm = Type::Ref(TensorType::new(DType::F32, &[8, 128]), Space::Any);
    assert!(k.alloc(hbm).is_err());
    // A slice stays inside its ref.
    assert!(
        k.slice(
            memory,
            vec![Some(60), Some(0), Some(0)],
            &[],
            vec![8, 1, 128],
            vec![false; 3]
        )
        .is_err()
    );
    // An unended `If` is not lowered.
    let t = Primitive::Full {
        shape: vec![],
        fill_value: Scalar::Bool(true),
        dtype: DType::Bool,
    };
    let t = k.apply(t, &[]).unwrap();
    k.begin_if(t).unwrap();
    assert!(lower(&k, "open").is_err());
    k.end_if().unwrap();
    assert!(lower(&k, "closed").is_ok());
}
