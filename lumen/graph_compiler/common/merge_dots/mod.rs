//! Merging dots that share an operand into one dot against a packed block: `forward` merges the forward
//! dots (e.g. the Q, K and V projections of one activation), `backward` their gradients' dots.

#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
mod backward;
mod forward;

pub(crate) use backward::MergeDotsBackward;
#[cfg(all(test, lumen_mps_linked))]
pub(crate) use forward::merge_dots;
pub(crate) use forward::{Blocks, MergeDots, Packs, Target};
