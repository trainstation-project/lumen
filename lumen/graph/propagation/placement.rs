//! How a value is laid out along one mesh dimension, and the strategies built from those layouts.

/// The reduction a partial value is pending.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Sum,
    Max,
    Min,
    Avg,
}

/// How a value is laid out along one mesh dimension (DTensor's `Shard`, `Replicate` and `Partial`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Split along this tensor dimension, one equal block per process.
    Shard(usize),
    /// Every process holds the whole value.
    Replicate,
    /// Every process holds a part; the value is their reduction, not yet done.
    Partial(Op),
}

/// A sharding strategy on one mesh dimension: each operand's placement on it, and the result's.
pub type Strategy = (Vec<Placement>, Placement);

impl Placement {
    /// Whether this is a partial value a linear op keeps partial: a sum's or an average's.
    pub(super) fn summed(self) -> bool {
        matches!(self, Placement::Partial(Op::Sum | Op::Avg))
    }
}
