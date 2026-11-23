//! Encodes row gathers from `[vocab, width]` tables on TPU, on the SparseCore when the layout allows.

use crate::DType;
use crate::codegen::devices::tpu::Tpu;
use crate::codegen::devices::tpu::{Kernel, Pallas, Rows};
use crate::codegen::dot_general::tpu::block;
use crate::device::tpu::layout::Layout;
use crate::graph::Primitive;
use crate::plan::StepBase;
use crate::plan::TpuStep;
use crate::plan::step::tpu::Pointers;
use crate::runtime::kernel::Launch as _;
use crate::runtime::kernel::tpu::TpuLaunch;
use crate::runtime::kernel::tpu::{Indices, SPARSE_CORE_MESH, SparseCore, Values};

/// TPU lane width.
const LANES: usize = 128;

/// Most bytes of fetched tiles per output block.
const BLOCK_BYTES: usize = 8 << 20;

/// Most indices loaded into scalar memory at once.
const WINDOW: usize = 8192;

/// Most output rows per block.
const ROWS: usize = 64;

/// Most bytes one SparseCore chunk of gathered rows may take.
const SPARSE_CORE_CHUNK: usize = 128 << 10;

/// Rejects gathers the kernels don't cover: only table rows at int32 indices along axis 0.
fn checked(step: &StepBase) -> Result<(), String> {
    let (table, ids) = (&step.inputs[0].1, &step.inputs[1].1);
    let failed = |what: &str| Err(format!("{what} does not run on TPU yet: {table} at {ids}"));

    let Primitive::Gather { axis: 0 } = step.primitive else {
        return failed("a gather along other than axis 0");
    };

    if table.shape.len() != 2 || ids.shape.is_empty() || ids.dtype != DType::I32 {
        return failed("a gather of other than a [vocab, width] table's rows at int32 indices");
    }

    if !matches!(table.dtype, DType::F32 | DType::BF16 | DType::I32) {
        return failed("a table of other than float32, bfloat16 or int32");
    }

    Ok(())
}

/// The TensorCore kernel gathering table rows at int32 indices along axis 0, rejecting layouts it doesn't cover.
pub(crate) fn kernel(step: &StepBase) -> Result<Kernel, String> {
    checked(step)?;

    let (table, ids) = (&step.inputs[0].1, &step.inputs[1].1);
    let failed = |what: &str| Err(format!("{what} does not run on TPU yet: {table} at {ids}"));
    let (vocab, width) = (table.shape[0], table.shape[1]);

    let tile = Layout::new(table.dtype, &[1, 1])?.rows;
    // `tile` table rows share one TPU tile; output blocks hold whole tiles of rows.
    let rank = ids.shape.len();
    let s = ids.shape[rank - 1];
    let r = if rank >= 2 { ids.shape[rank - 2] } else { 1 };

    let indices: Vec<usize> = match rank {
        1 => vec![s],
        _ => vec![ids.shape[..rank - 2].iter().product(), r, s],
    };

    // Index windows read `rw` rows of `cw` indices; they must tile the index array.
    let rw = match r {
        _ if rank == 1 => 1,
        r if r.is_multiple_of(8) => 8,
        r if r < 8 => r,
        _ => return failed("indices of rows not a multiple of 8 (or fewer)"),
    };

    let cw = match s {
        s if rw * s <= WINDOW => s,
        s if s.is_multiple_of(LANES) => LANES,
        _ => return failed("a long last dimension of indices not a multiple of 128"),
    };

    let whole = |b: &usize| s.is_multiple_of(*b) && cw.is_multiple_of(*b);

    // Else one block of every index of a window that spans the last dim (a block of the output's whole rows).
    let bm = match (1..=ROWS / tile).rev().map(|k| k * tile).find(whole) {
        Some(bm) => bm,
        None if s < tile || (cw == s && s <= ROWS) => s,
        None => {
            return failed("a last dimension of indices not a multiple of the table's tile rows");
        }
    };

    let cap = (BLOCK_BYTES / (bm * tile * 4)).max(LANES);

    let bn = block(width, LANES, cap)
        .filter(|&b| b == width || b.is_multiple_of(LANES))
        .ok_or_else(|| format!("gather on TPU: {width} columns have no block of whole tiles"))?;

    let spec = format!(
        "\"vocab\": {vocab}, \"width\": {width}, \"dtype\": \"{}\", \"tile\": {tile}, \"indices\": {indices:?}, \"bm\": {bm}, \"bn\": {bn}, \"rw\": {rw}, \"cw\": {cw}",
        table.dtype.full_name()
    );

    let mut kernel = Kernel::new(
        "lumen.codegen.gather.tpu.kernel",
        format!("gather_{}_{bm}x{bn}", table.dtype),
        spec,
        Rows::Step,
    );

    kernel.context = Some(step.label.to_string());
    Ok(kernel)
}

/// The kernel of the `take_along_axis` step: a 32-bit operand of rank 2 or more at int32 indices, each viewed at its
/// tile-padded shape (`lumen.codegen.gather.tpu.take_along_axis`, by the axis: the last, the second-last or an earlier
/// one); rejects others.
pub(crate) fn take_along_axis(step: &StepBase) -> Result<Kernel, String> {
    let Primitive::TakeAlongAxis { axis } = step.primitive else {
        unreachable!("a take_along_axis step")
    };

    let (x, ids) = (&step.inputs[0].1, &step.inputs[1].1);

    let failed = |what: &str| {
        Err(format!(
            "{what} does not run on TPU yet: take_along_axis of {x} at {ids}"
        ))
    };

    if x.dtype.bits() != 32 {
        return failed("an operand of other than 32 bits");
    }

    if !matches!(ids.dtype, DType::I32 | DType::U32) {
        return failed("indices of other than 32 bits");
    }

    let rank = x.shape.len();

    if rank < 2 {
        return failed("an operand of fewer than two dimensions");
    }

    let (operand, indices) = (
        Layout::new(x.dtype, &x.shape)?,
        Layout::new(ids.dtype, &ids.shape)?,
    );
    let (xv, iv) = (operand.padded(), indices.padded());
    let lead = &ids.shape[..rank - 2];

    // The operand's row-major strides over its leading dims, in slabs.
    let mut strides = vec![1; rank - 2];

    for d in (0..(rank - 2).saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * x.shape[d + 1];
    }

    let kind = match axis {
        a if a == rank - 1 => "lanes",
        a if a == rank - 2 => "sublanes",
        _ => "outer",
    };

    let spec = format!(
        "\"kind\": \"{kind}\", \"dtype\": \"{}\", \"x\": [{}, {}, {}], \"idx\": [{}, {}, {}], \"lead\": {lead:?}, \"strides\": {strides:?}, \"bm\": {}, \"bn\": {LANES}, \"tile\": {}, \"axis\": {axis}, \"n\": {}",
        x.dtype.full_name(),
        xv.0,
        xv.1,
        xv.2,
        iv.0,
        iv.1,
        iv.2,
        operand.rows,
        operand.rows,
        x.shape[axis]
    );

    let mut kernel = Kernel::new(
        "lumen.codegen.gather.tpu.take_along_axis",
        format!("take_along_axis_{kind}_{}", x.dtype),
        spec,
        Rows::Step,
    );

    kernel.context = Some(step.label.to_string());
    Ok(kernel)
}

/// Encodes the gather on the SparseCore when rows are 32-bit, whole lanes and tile-aligned in the shared region;
/// returns whether it did. Rejects gathers no kernel covers.
pub(crate) fn sparse_core(
    index: usize,
    step: &StepBase,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<bool, String> {
    checked(step)?;

    let (table, ids) = (&step.inputs[0].1, &step.inputs[1].1);
    let width = table.shape[1];

    let words = match table.dtype.bits() {
        32 if width.is_multiple_of(LANES) => width,
        _ => return Ok(false),
    };

    let Some(k) = [256, 128, 64, 32, 16, 8]
        .into_iter()
        .find(|k| k * words * 4 <= SPARSE_CORE_CHUNK)
    else {
        return Ok(false);
    };

    let column = words / LANES;
    let region = crate::allocator::tpu::rows(index)?;
    let rows = crate::allocator::tpu::rows_of(index, &[inputs[0], output.cast_const()])?;

    if [region, rows[0] as usize, rows[1] as usize]
        .iter()
        .any(|r| !r.is_multiple_of(8 * column))
    {
        return Ok(false);
    }

    let n = ids.numel();
    let s = ids.shape[ids.shape.len() - 1];
    // Indices map to consecutive output rows unless padding of a ragged last index dim breaks that.
    let linear = n == s || s.is_multiple_of(8);
    let workers = SPARSE_CORE_MESH[0] * SPARSE_CORE_MESH[1];
    let count = n.next_multiple_of(k * workers);

    let spec = format!(
        "\"sparse_core\": true, \"rows_of_view\": {}, \"width\": {words}, \"count\": {count}, \"k\": {k}, \"n\": {n}, \"linear\": {linear}, \"scalars\": 3, \"out\": 2, \"mesh\": {SPARSE_CORE_MESH:?}",
        region * LANES / words
    );

    let mut indices = vec![Indices {
        values: Values::Rows {
            scalar: 0,
            shape: ids.shape.clone(),
        },
        base: 1,
    }];

    if !linear {
        indices.push(Indices {
            values: Values::Iota {
                dims: vec![n / s, s],
                strides: vec![s.div_ceil(8) * 8, 1],
            },
            base: 2,
        });
    }

    let sparse = SparseCore {
        width: words,
        count,
        indices,
        scalars: linear,
    };

    let operands = [inputs[1], inputs[0], output.cast_const()];

    let launched = TpuLaunch {
        device: index,
        module: "lumen.codegen.gather.tpu.kernel",
        kernel: &format!("gather_sparse_core_{}_{k}x{words}", table.dtype),
        spec,
        rows: &operands,
        scalars: &[],
        host: &[],
        sparse: Some(sparse),
        lowered: None,
    };

    Tpu::launch(launched).map_err(|e| format!("{}: {e}", step.label))?;
    Ok(true)
}

/// A gather: on the SparseCore when its buffers' rows allow (known only at launch), else its TensorCore kernel.
pub(crate) struct Gather;

impl crate::codegen::kernel::Kernel for Gather {
    type Step = TpuStep;
    type Buffers = Pointers;
    type Config = ();

    /// Runs the gather on the SparseCore if it takes it, else its Pallas kernel; with neither, fails with the
    /// TensorCore kernel's error.
    fn run(&self, step: &TpuStep, config: Option<&()>, p: &Pointers) -> Result<(), String> {
        if sparse_core(p.index, step, &p.inputs, p.output)? {
            return Ok(());
        }

        match step.code {
            Some(_) => crate::codegen::kernel::Kernel::run(&Pallas, step, config, p),
            None => kernel(step).map(|_| ()),
        }
    }
}
