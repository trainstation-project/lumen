//! Tests of building Pallas TPU kernels and the Mosaic text they lower to.

use super::mosaic::lower;
use super::{Kernel, Op, Space, Type};
use crate::graph::{Primitive, TensorType};
use crate::{DType, Scalar};

/// The type of an int32 scalar.
fn i32_scalar() -> TensorType {
    TensorType::new(DType::I32, &[])
}

/// Adds an int32 constant `v` to `k`, returning its variable.
fn constant(k: &mut Kernel, v: i64) -> usize {
    let full = Primitive::Full {
        shape: vec![],
        fill_value: Scalar::Int(v),
        dtype: DType::I32,
    };

    k.apply(full, &[]).unwrap()
}

/// A host-to-memory copy loads its row from SMEM and lowers to a DMA start and wait.
#[test]
fn a_copy_kernel_lowers_to_a_dma_at_a_scalar_row() {
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

/// A for loop with a first-step `if` accumulates a matmul in a VMEM ref.
#[test]
fn a_loop_accumulates_a_matmul_in_vmem() {
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
    k.end_if(&[]).unwrap();

    let (xv, yv, av) = (
        k.load(x, &[]).unwrap(),
        k.load(y, &[]).unwrap(),
        k.load(acc, &[]).unwrap(),
    );

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

/// Shifts and bitcasts unpack a bfloat16 from a 32-bit word.
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

/// Integer and float constants lower bit-exactly, including -inf.
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

/// Loads from HBM or whole SMEM, HBM allocations, out-of-range slices and unclosed regions fail.
#[test]
fn kernels_refuse_what_tpus_cannot_do() {
    let mut k = Kernel::new(1, None, 64);

    let [scalars, memory, _] = k.params()[..] else {
        unreachable!()
    };

    assert!(k.load(memory, &[]).is_err());
    assert!(k.load(scalars, &[]).is_err());
    let hbm = Type::Ref(TensorType::new(DType::F32, &[8, 128]), Space::Any);
    assert!(k.alloc(hbm).is_err());

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

    let t = Primitive::Full {
        shape: vec![],
        fill_value: Scalar::Bool(true),
        dtype: DType::Bool,
    };

    let t = k.apply(t, &[]).unwrap();
    k.begin_if(t).unwrap();
    assert!(lower(&k, "open").is_err());
    k.end_if(&[]).unwrap();
    assert!(lower(&k, "closed").is_ok());
}

/// A SparseCore kernel gathers HBM rows by VMEM offsets with an indirect DMA.
#[test]
fn a_sparse_core_kernel_gathers_rows_by_their_offsets() {
    let view = TensorType::new(DType::U32, &[64, 256]);
    let offsets = TensorType::new(DType::I32, &[16]);
    let mut k = Kernel::sparse_core(&[view.clone(), offsets.clone(), view], [2, 16]);

    let [core, _, _, src, region] = k.params()[..] else {
        unreachable!()
    };

    let at = k
        .alloc(Type::Ref(TensorType::new(DType::I32, &[8]), Space::Vmem))
        .unwrap();

    let rows = k
        .alloc(Type::Ref(
            TensorType::new(DType::U32, &[8, 256]),
            Space::Vmem,
        ))
        .unwrap();

    let sem = k.alloc(Type::Semaphore(vec![])).unwrap();
    let eight = constant(&mut k, 8);
    let start = k.apply(Primitive::Mul, &[core, eight]).unwrap();

    let window = k
        .slice(src, vec![None], &[start], vec![8], vec![false])
        .unwrap();

    k.dma(window, at, sem, false).unwrap();
    k.dma(window, at, sem, true).unwrap();
    k.indirect_dma(region, rows, at, sem, false).unwrap();
    k.indirect_dma(region, rows, at, sem, true).unwrap();

    assert!(
        k.indirect_dma(rows, rows, at, sem, false).is_err(),
        "between HBM rows and VMEM"
    );

    let text = lower(&k, "gather").unwrap();

    for expected in [
        "^bb0(%arg0: i32, %arg1: i32, %arg2: memref<64x256xi32, #tpu.memory_space<hbm>>, %arg3: \
         memref<16xi32, #tpu.memory_space<hbm>>, %arg4: memref<64x256xi32, #tpu.memory_space<hbm>>,",
        "\"stable_mosaic.tpu.enqueue_indirect_dma\"(%arg4, %arg6, %arg5, %arg7) {add = false}",
        "\"stable_mosaic.tpu.wait_indirect_dma\"(%arg7, %arg4, %arg6)",
        "iteration_bounds = array<i64: 2, 16>, scalar_prefetch = 0 : i64, scratch_operands = 3 : i64",
        "tpu.core_type = #tpu.core_type<sc_vector_subcore>",
        "window_params = [{transform_indices = affine_map<(d0, d1) -> (0, 0)>}, \
         {transform_indices = affine_map<(d0, d1) -> (0)>}, {transform_indices = affine_map<(d0, d1) -> (0, 0)>}]",
    ] {
        assert!(text.contains(expected), "{expected}\nin\n{text}");
    }
}

/// `pow` with an integer exponent lowers via `|x|` and restores the sign from the exponent's parity.
#[test]
fn an_integer_power_takes_its_sign_from_the_exponents_parity() {
    let mut k = Kernel::new(1, None, 64);
    let shape = [8, 128];

    let base = k
        .alloc(Type::Ref(TensorType::new(DType::BF16, &shape), Space::Vmem))
        .unwrap();

    let exponent = k
        .alloc(Type::Ref(TensorType::new(DType::I32, &shape), Space::Vmem))
        .unwrap();

    let (x, e) = (k.load(base, &[]).unwrap(), k.load(exponent, &[]).unwrap());

    let y = k
        .apply(Primitive::Binary(crate::graph::BinaryOp::Pow), &[x, e])
        .unwrap();

    assert_eq!(
        *k.type_of(y),
        Type::Value(TensorType::new(DType::BF16, &shape))
    );

    let text = lower(&k, "power").unwrap();

    for expected in [
        "\"stable_mosaic.arith.sitofp\"(%t3) : (vector<8x128xi32>) -> vector<8x128xf32>",
        "\"stable_mosaic.math.absf\"(",
        "\"stable_mosaic.math.powf\"(%t6, %t5)",
        "\"stable_mosaic.arith.andi\"(%t3, ",
        "\"stable_mosaic.arith.select\"(",
        "-> vector<8x128xbf16>",
    ] {
        assert!(text.contains(expected), "{expected}\nin\n{text}");
    }

    assert!(!text.contains("i32>) -> vector<8x128xbf16>"), "{text}");
}
