//! Launching a Metal kernel.

use super::Launch;
use crate::Tensor;
use crate::codegen::utils::mps::Mps;
use crate::runtime::mps::{CAPTURED, Captured, Grid, MAX_ARGS, dispatch, pipeline};

/// One Metal launch: a kernel by name, its buffers and by-value arguments, its grid, the tensors its command
/// buffer keeps alive, and the step label it is profiled under.
pub(crate) struct MpsLaunch<'a> {
    pub(crate) kernel: &'a str,
    pub(crate) buffers: &'a [*const u8],
    pub(crate) args: &'a [Vec<u8>],
    pub(crate) grid: Grid,
    pub(crate) keep: Vec<Tensor>,
    pub(crate) name: &'static str,
}

/// MPS dispatches a kernel's pipeline on the MPS stream, or records it inside `capture`.
impl Launch for Mps {
    type Args<'a> = MpsLaunch<'a>;

    /// Launches `kernel`, or records it when inside `capture`; a grid with a zero dimension is skipped.
    fn launch(args: MpsLaunch<'_>) -> Result<(), String> {
        let MpsLaunch {
            kernel,
            buffers,
            args,
            grid,
            keep,
            name,
        } = args;

        let (sizes, groups, group) = match grid {
            Grid::Threads(sizes) => (sizes, false, 0),
            Grid::Groups(sizes) => (sizes, true, 0),
            Grid::GroupsOf { groups, threads } => (groups, true, threads),
            Grid::Elementwise { threads, group } => ([threads, 1, 1], false, group),
        };

        if sizes.contains(&0) {
            crate::profiler::clear_next_launch();
            return Ok(());
        }

        let pipeline = pipeline(kernel)?;

        if args.len() > MAX_ARGS {
            return Err(format!(
                "the MPS kernel {kernel} takes at most {MAX_ARGS} arguments"
            ));
        }

        let captured = CAPTURED.with_borrow_mut(|c| {
            c.as_mut().map(|c| {
                c.push(Captured {
                    pipeline,
                    buffers: buffers.to_vec(),
                    args: args.to_vec(),
                    sizes,
                    groups,
                    group,
                    kernel: kernel.to_owned(),
                    label: name,
                })
            })
        });

        if captured.is_some() {
            return Ok(());
        }

        dispatch(
            pipeline,
            buffers,
            args,
            (sizes, groups, group),
            keep,
            name,
            kernel,
        )
    }
}
