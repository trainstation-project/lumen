//! gather and scatter_add on MPS (`kernels.metal`): a thread an element of
//! the gathered values (a gather's result, a scatter's updates). A
//! scatter_add whose value the plan put in its operand's buffer
//! (`graph/plan.rs`: nothing reads the operand after it) adds the updates
//! there, in place; otherwise it copies the operand first.

use crate::graph::Primitive;
use crate::graph::plan::Step;
use crate::ops::mps::{Grid, launch, u64_arg};
use crate::{DType, Tensor};

/// Threads a grid row takes: past it, rows (Metal grids are 32-bit a
/// dimension).
const ROW: usize = 1 << 20;

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let (x, indices) = (&step.inputs[0].1, &step.inputs[1].1);
    let axis = match step.primitive {
        Primitive::Gather { axis } | Primitive::ScatterAdd { axis } => axis,
        _ => unreachable!("a gather or scatter_add"),
    };
    let index = match indices.dtype {
        DType::I32 => "i32",
        _ => "i64",
    };
    let (n, m) = (x.shape[axis], indices.numel());
    let inner: usize = x.shape[axis + 1..].iter().product();
    let scatter = matches!(step.primitive, Primitive::ScatterAdd { .. });
    // The gathered values: the result, or the updates.
    let total = match scatter {
        true => step.inputs[2].1.numel(),
        false => step.output.1.numel(),
    };
    let (kernel, buffers) = match scatter {
        false => (
            format!("gather_{}_{index}", x.dtype.size_of()),
            vec![inputs[0], inputs[1], output.cast_const()],
        ),
        true => {
            let dtype = match x.dtype {
                DType::F32 => "f32",
                DType::I32 => "i32",
                DType::U32 => "u32",
                d => {
                    return Err(format!(
                        "{}: MPS adds float32, int32 and uint32 atomically, got {d}",
                        step.label
                    ));
                }
            };
            // The operand's value, if not already in the output's buffer.
            if output.cast_const() != inputs[0] {
                let name = step.label;
                crate::ops::dynamic_slice::mps::copy(inputs[0], output, x, keep.clone(), name)?;
            }
            (
                format!("scatter_add_{dtype}_{index}"),
                vec![inputs[2], inputs[1], output.cast_const()],
            )
        }
    };
    if total == 0 || n == 0 {
        crate::profiler::clear_next_launch();
        return Ok(());
    }
    let args = [
        u64_arg(n),
        u64_arg(m),
        u64_arg(inner.max(1)),
        u64_arg(total),
    ];
    let grid = Grid::Threads([total.min(ROW), total.div_ceil(ROW), 1]);
    launch(&kernel, &buffers, &args, grid, keep, step.label)
}
