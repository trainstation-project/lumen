//! Metal flash-attention block sizes, the largest whose tiles fit in threadgroup memory, and the forward's
//! tunings.

use crate::DType;
use crate::graph_compiler::passes::attention::{Attention, Backward, grid_batch};

/// Query rows per block for the backward kernel, the largest that fits in threadgroup memory.
pub(crate) fn backward_block(b: &Backward) -> usize {
    // Each tile row is padded by 16 bytes (`ATTN_PAD`).
    let bytes = |n: usize| n * ((b.h + b.hv) * b.dtype.size_of() + 32) + 2 * n * 4;

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
    // Each tile row is padded by 16 bytes (`ATTN_PAD`).
    let bytes =
        |n: usize| n * ((b.h + b.hv) * t + 32) + 2 * n * 4 + KEYS * (b.h * t + 16) + KEYS * n * t;
    [32, 16].into_iter().find(|&n| bytes(n) <= 28 << 10)
}

/// How the forward kernel runs: on Metal Performance Primitives (`mpp`, 16 queries a simdgroup and 32 keys a
/// block) or 8x8 simdgroup matrices (8 queries a simdgroup, `keys` a block), with `groups` simdgroups a
/// threadgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Tuning {
    pub mpp: bool,
    pub groups: usize,
    pub keys: usize,
}

impl Tuning {
    /// Query rows each simdgroup computes.
    pub(crate) fn rows(&self) -> usize {
        match self.mpp {
            true => 16,
            false => 8,
        }
    }
}

/// The forward's untuned kernel. 8x8 matrices with 8 simdgroups (64 queries a block) when that still makes 512
/// threadgroups, else 4 (32, as MLX's steel attention) to keep the GPU busy; on an M4 Pro, 8 heads of 64:
/// S = 1024 runs 0.8x torch with 64-query blocks and 1.0-1.1x with 32; S = 4096 runs 1.07x with 64 and 0.92x
/// with 32, its keys and values read twice as often. Metal Performance Primitives from 4096 keys, where they
/// fit: below, their round trip of scores through threadgroup memory costs more than their matmuls save (causal,
/// 8 heads of 64 in bfloat16: S = 4096 and 8192 run 5% faster than the 8x8 kernel, S = 256 to 2048 4-25%
/// slower).
pub(crate) fn tuning(a: &Attention) -> Tuning {
    if mpp_fits(a) && a.sk >= 4096 {
        return Tuning {
            mpp: true,
            groups: 4,
            keys: 32,
        };
    }

    let (outer, heads) = grid_batch(&a.batch);

    let groups = match a.sq.div_ceil(64) * heads * outer >= 512 {
        true => 8,
        false => 4,
    };

    Tuning {
        mpp: false,
        groups,
        keys: key_blocks(a)[0],
    }
}

/// Fewest query-key pairs a head the forward is autotuned at: below, its launches are too short to time apart.
const TUNED_PAIRS: usize = 1 << 20;

/// The tunings autotuning times for the forward: 8x8 matrices with 4 or 8 simdgroups and each key block that
/// fits, and Metal Performance Primitives with 4 or 8 where they fit; none for the decode kernel or small
/// attentions.
pub(crate) fn tunings(a: &Attention) -> Vec<Tuning> {
    if decodes(a) || a.sq * a.sk < TUNED_PAIRS {
        return Vec::new();
    }

    let tiled = key_blocks(a).into_iter().flat_map(|keys| {
        [4, 8].map(|groups| Tuning {
            mpp: false,
            groups,
            keys,
        })
    });

    let mpp = [4, 8]
        .into_iter()
        .filter(|_| mpp_fits(a))
        .map(|groups| Tuning {
            mpp: true,
            groups,
            keys: 32,
        });

    tiled.chain(mpp).collect()
}

/// Whether the forward can run on Metal Performance Primitives (`flash_attention_mpp`): half or bfloat16
/// operands with contiguous columns, head dims of whole 16-wide fragments, whole 16-query and 32-key blocks,
/// no dropout, and a toolchain that has them.
fn mpp_fits(a: &Attention) -> bool {
    matches!(a.dtype, DType::F16 | DType::BF16)
        && !decodes(a)
        && a.dropout.is_none()
        && [&a.q, &a.k, &a.v].iter().all(|x| x.col == 1)
        && a.h.is_multiple_of(16)
        && a.hv.is_multiple_of(16)
        && a.sq.is_multiple_of(16)
        && a.sk.is_multiple_of(32)
        && crate::graph_compiler::mps::mpp_available()
}

/// Whether the forward uses the decode kernel (few query rows).
pub(crate) fn decodes(a: &Attention) -> bool {
    a.sq <= 8
}

/// Keys per block the forward's 8x8 kernel can take: of 32, 16 and 64, those whose K/V tiles fit in threadgroup
/// memory, the untuned one (32, else 16) first; 8 when none does.
fn key_blocks(a: &Attention) -> Vec<usize> {
    let t = a.dtype.size_of();
    // Each tile row is padded by 16 bytes (`ATTN_PAD`).
    let bytes = |bk: usize| bk * ((a.h + a.hv) * t + 32);
    let fits: Vec<usize> = [32, 16, 64]
        .into_iter()
        .filter(|&bk| bytes(bk) <= 28 << 10)
        .collect();

    match fits.is_empty() {
        true => vec![8],
        false => fits,
    }
}

#[cfg(feature = "python")]
crate::autotune::python::config_class! {
    /// `lumen.autotune.MpsAttentionConfig`: how the Metal flash-attention forward runs (`mpp`: on Metal Performance
    /// Primitives, 32 keys a block), its simdgroups a threadgroup and its keys a block.
    PyMpsAttentionConfig(Tuning) = "MpsAttentionConfig" {
        mpp: bool,
        groups: usize,
        keys: usize,
    }

    /// Every config the autotuner can time: 8x8 matrices with 4 or 8 simdgroups and each key block (one fits a
    /// given attention only when its K/V tiles fit threadgroup memory), and Metal Performance Primitives with 4
    /// or 8.
    #[staticmethod]
    fn candidates() -> Vec<Self> {
        let tiled = [32, 16, 64, 8]
            .into_iter()
            .flat_map(|keys| [4, 8].map(|groups| Tuning { mpp: false, groups, keys }));
        let mpp = [4, 8].map(|groups| Tuning { mpp: true, groups, keys: 32 });

        tiled.chain(mpp).map(PyMpsAttentionConfig).collect()
    }
}
