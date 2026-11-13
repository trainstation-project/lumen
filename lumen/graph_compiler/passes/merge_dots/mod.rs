//! Merging dots that share an operand into one dot against a packed block: `forward` merges the forward
//! dots (e.g. the Q, K and V projections of one activation), `backward` their gradients' dots.

mod cuda;
#[cfg(lumen_mps_linked)]
mod mps;
mod tpu;
mod triton;

pub(crate) use backward::MergeDotsBackward;
pub(crate) use forward::{Blocks, MergeDots, Packs, Target};

#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
mod backward;
mod forward;

#[cfg(all(test, lumen_mps_linked))]
pub(crate) use forward::merge_dots;

/// The late dot merge, once the graph is simplified (TPU's, packing whole vector lanes).
pub(crate) struct MergeDotsLate;
