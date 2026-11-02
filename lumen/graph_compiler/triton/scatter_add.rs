//! Running `scatter_add` steps: by default one kernel adding each update element atomically; with
//! `cuda.deterministic`, or a dtype Triton has no atomic add for, a counting sort of the picks by position,
//! then a program a block of each position's entries adding its updates in pick order (as MPS's).

use super::{Kernel, Launches, Template, fill, launch};
use crate::codegen::elementwise::triton::LoopConfig;
use crate::codegen::scatter_add::triton::{ELEMENTS, SEGMENT, STARTS, rows};
use crate::{DType, Scalar, Tensor, TensorOptions};

/// A Triton scatter-add kernel (lumen.codegen.scatter_add.triton.kernel).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub enum Scatter {
    /// An update element a lane, added atomically.
    Atomic,
    /// A pick a lane, counting the picks at each position.
    Count,
    /// One program: each position's segment start, a prefix sum of the counts.
    Starts,
    /// A pick a lane, written into its position's segment.
    Place,
    /// A position a program, its segment sorted.
    Sort,
    /// A block of a position's entries off the axis a program, adding its updates in pick order; a step
    /// running it sorts its picks by position first (`Count`, `Starts`, `Place`, `Sort`).
    Rows,
}

#[cfg_attr(not(feature = "python"), allow(dead_code))]
impl Scatter {
    /// The kernel's name in lumen.codegen.scatter_add.triton.kernel.
    pub fn name(self) -> &'static str {
        match self {
            Scatter::Atomic => "scatter_add_atomic_kernel",
            Scatter::Count => "scatter_add_count_kernel",
            Scatter::Starts => "scatter_add_starts_kernel",
            Scatter::Place => "scatter_add_place_kernel",
            Scatter::Sort => "scatter_add_sort_kernel",
            Scatter::Rows => "scatter_add_rows_kernel",
        }
    }
}

/// Adds the updates of `inputs` (`[operand, indices, updates]`) into `out`, which already holds the operand,
/// with step kernel `kernel`: its atomic kernel, or its picks sorted by position then its rows kernel.
pub(super) fn run(
    kernel: &Kernel,
    launches: &Launches,
    inputs: &[Tensor],
    out: &Tensor,
) -> Result<(), String> {
    let Some(Template::ScatterAdd {
        outer,
        length,
        after,
        picks,
        kernel: scatter,
    }) = kernel.template
    else {
        unreachable!("a scatter-add's kernel")
    };

    let [_, indices, updates] = inputs else {
        unreachable!("scatter_add reads its operand, indices and updates")
    };

    if out.numel() == 0 || picks == 0 {
        return Ok(());
    }

    // One launch of `scatter` with `config` over `n` (picks, positions, ...), `lanes` a program's elements.
    let stage = |scatter: Scatter,
                 config: &LoopConfig,
                 lanes: usize,
                 inputs: &[Tensor],
                 into: &Tensor,
                 n: usize| {
        let kernel = Kernel {
            name: format!("{}_{}", kernel.name, scatter.name()),
            by_value: Vec::new(),
            lanes,
            template: Some(Template::ScatterAdd {
                outer,
                length,
                after,
                picks,
                kernel: scatter,
            }),
            ..kernel.clone()
        };

        launch(&kernel, config, &[], launches, inputs, into, n)
    };

    if scatter == Scatter::Atomic {
        return stage(Scatter::Atomic, &ELEMENTS, 1, inputs, out, updates.numel());
    }

    let options = TensorOptions::new().dtype(DType::I32).device(out.device());

    // Each position's pick count (then its place cursor) and segment start, and the picks by position.
    let (counts, starts, order) = unsafe {
        (
            Tensor::empty(&[length], options),
            Tensor::empty(&[length + 1], options),
            Tensor::empty(&[picks], options),
        )
    };

    fill(&counts, Scalar::Int(0))?;

    stage(
        Scatter::Count,
        &ELEMENTS,
        1,
        std::slice::from_ref(indices),
        &counts,
        picks,
    )?;

    stage(
        Scatter::Starts,
        &STARTS,
        STARTS.block_size,
        std::slice::from_ref(&counts),
        &starts,
        1,
    )?;

    stage(
        Scatter::Place,
        &ELEMENTS,
        1,
        &[indices.clone(), starts.clone(), counts],
        &order,
        picks,
    )?;

    stage(
        Scatter::Sort,
        &SEGMENT,
        SEGMENT.block_size,
        &[indices.clone(), starts.clone()],
        &order,
        length,
    )?;

    // A program a block of each position's entries off the axis (outer and inner indices).
    let config = rows(outer * after);
    let programs = length * (outer * after).div_ceil(config.block_size);

    stage(
        Scatter::Rows,
        &config,
        config.block_size,
        &[updates.clone(), starts, order],
        out,
        programs,
    )
}
