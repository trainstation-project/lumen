//! Launch tunings: the generated elementwise loop's candidates and defaults, the sizes worth
//! autotuning, and the flash-attention kernels' block sizes.
use super::*;

/// Launch shape for a generated kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Tuning {
    /// Elementwise loop: each thread handles `per_thread` elements in threadgroups of `group`;
    /// `contiguous` gives each thread a contiguous run instead of a grid-strided one.
    Loop {
        per_thread: usize,
        group: usize,
        contiguous: bool,
    },
}

/// Output sizes, in elements, for which loop fusions are worth tuning.
pub(crate) const TUNED_ELEMENTS: std::ops::RangeInclusive<usize> = (64 << 10)..=(16 << 20);
/// Thread count the default tuning aims to launch.
const GRID_THREADS: usize = 32 << 10;
/// Threadgroup sizes tried when autotuning.
const LOOP_GROUPS: [usize; 5] = [64, 128, 256, 512, 1024];
/// Default threadgroup size.
const LOOP_GROUP: usize = 256;

/// Most elements one thread may handle, from the per-thread byte budget and output dtype size.
fn most_per_thread(body: &Graph) -> usize {
    let dtype = body.type_of(body.outputs()[0]).dtype;
    (crate::kernels::mps::BYTES_PER_THREAD / dtype.size_of()).max(1)
}

/// Default tuning: enough elements per thread to cap the grid near `GRID_THREADS`.
pub(crate) fn loop_tuning(body: &Graph) -> Tuning {
    let n = body.type_of(body.outputs()[0]).numel();
    Tuning::Loop {
        per_thread: (n / GRID_THREADS).clamp(1, most_per_thread(body)),
        group: LOOP_GROUP,
        contiguous: false,
    }
}

/// Every power-of-two per-thread count up to the cap, crossed with threadgroup sizes and
/// (when more than one element per thread) both access orders.
pub(crate) fn loop_tunings(body: &Graph) -> Vec<Tuning> {
    let most = most_per_thread(body);
    let per_threads = (0..).map(|k| 1 << k).take_while(|&p| p <= most);
    per_threads
        .flat_map(|per_thread| {
            let contiguous = match per_thread {
                1 => &[false][..],
                _ => &[false, true][..],
            };
            contiguous.iter().flat_map(move |&contiguous| {
                LOOP_GROUPS.map(|group| Tuning::Loop {
                    per_thread,
                    group,
                    contiguous,
                })
            })
        })
        .collect()
}

/// Query rows per block for the backward kernel, the largest that fits in threadgroup memory.
pub(super) fn backward_block(b: &Backward) -> usize {
    let bytes = |n: usize| n * (b.h + b.hv) * b.dtype.size_of() + 2 * n * 4;
    [32, 16, 8]
        .into_iter()
        .find(|&n| bytes(n) <= 28 << 10)
        .unwrap_or(8)
}

/// Query rows per block when dQ is fused into the dK/dV kernel, or `None` if even 16 rows
/// don't fit in threadgroup memory alongside the key tile.
pub(crate) fn backward_block_with_dq(b: &Backward) -> Option<usize> {
    const KEYS: usize = 64;
    let t = b.dtype.size_of();
    let bytes = |n: usize| n * (b.h + b.hv) * t + 2 * n * 4 + KEYS * b.h * t + KEYS * n * t;
    [32, 16].into_iter().find(|&n| bytes(n) <= 28 << 10)
}

/// Whether the forward uses the decode kernel (few query rows).
pub(crate) fn decodes(a: &Attention) -> bool {
    a.sq <= 8
}

/// Keys per block for the forward kernel, the largest whose K/V tiles fit in threadgroup memory.
pub(super) fn key_block(a: &Attention) -> usize {
    let t = a.dtype.size_of();
    let bytes = |bk: usize| bk * (a.h + a.hv) * t;
    [32, 16, 8]
        .into_iter()
        .find(|&bk| bytes(bk) <= 28 << 10)
        .unwrap_or(8)
}
