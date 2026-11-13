"""lumen's Ampere attention forward (FlashAttention-2): O = softmax(Q K^T * scale) V, with warp-level tensor-core MMA
(mma.sync), ldmatrix and cp.async loads, and the row log-sum-exp.

Ported from FlashAttention's CuTe DSL SM80 forward (https://github.com/Dao-AILab/flash-attention, commit f63c472c,
BSD-3-Clause): `FlashAttentionForwardSm80` and its base in `flash_attn/cute/flash_fwd.py`, the SM80 GEMMs of
`ampere_helpers.py`, `Softmax` of `softmax.py`, the SM80 path of `AttentionMask.apply_mask` in `mask.py` and the
causal block range of `BlockInfo` in `block_info.py`. Kept: causal masking (bottom-right aligned, as FlashAttention's),
sequence lengths that aren't tile multiples, head dims that aren't multiples of 16, grouped-query attention by head
index, the K/V cp.async pipeline (`num_stages`), `Q_in_regs` and the LSE output. Dropped: score and mask mods,
local windows, variable lengths, paged KV, learnable sinks, block sparsity, pack-GQA and the tile scheduler (one CTA
per (m block, head, batch) of the grid).
"""

import math
import operator
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.utils import get_smem_capacity_in_bytes

from lumen.kernels.cuda import copy_utils, layout_utils
from lumen.kernels.cuda.cute_dsl_utils import ParamsBase
from lumen.kernels.cuda.reduce import warp_reduce

LOG2_E = math.log2(math.e)


def smem_layout_atom(dtype: type[cutlass.Numeric], k_dim: int) -> cute.ComposedLayout:
    """The swizzled shared-memory layout atom of a `dtype` tile whose rows hold `k_dim` elements: rows of the widest
    of 128, 64, 32 or 16 bytes dividing a row, swizzled so ldmatrix reads them without bank conflicts."""
    dtype_byte = dtype.width // 8
    bytes_per_row = k_dim * dtype_byte
    smem_k_block_size = (
        128
        if bytes_per_row % 128 == 0
        else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))
    ) // dtype_byte
    swizzle_bits = (
        4 if smem_k_block_size == 128 else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)

    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout((8 if k_dim % 32 == 0 else 16, smem_k_block_size), order=(1, 0)),
    )


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    A_in_regs: cutlass.Constexpr[bool] = False,
    hook_fn: Callable | None = None,
) -> None:
    """acc += A B over the k blocks of shared-memory tiles `tCsA` and `tCsB`, loading each k block into registers
    (`tCrA`, `tCrB`) with ldmatrix one block ahead of its MMA; with `A_in_regs`, A is already in `tCrA`. `hook_fn`,
    when given, runs after the first k block's MMA (the backward issues its next tile's loads there)."""
    tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)

    if const_expr(not A_in_regs):
        cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])

    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])

    for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
        # The next k block's loads overlap this one's MMA.
        if k < cute.size(tCsA.shape[2]) - 1:
            if const_expr(not A_in_regs):
                cute.copy(
                    smem_thr_copy_A,
                    tCsA[None, None, k + 1],
                    tCrA_copy_view[None, None, k + 1],
                )

            cute.copy(
                smem_thr_copy_B,
                tCsB[None, None, k + 1],
                tCrB_copy_view[None, None, k + 1],
            )

        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)

        if const_expr(k == 0 and hook_fn is not None):
            hook_fn()


@cute.jit
def gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
) -> None:
    """acc += A B with A already in registers (`tCrA`) and B's k blocks loaded from shared `tCsB` one ahead."""
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])

    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if const_expr(k < cute.size(tCrA.shape[2]) - 1):
            cute.copy(
                smem_thr_copy_B,
                tCsB[None, None, k + 1],
                tCrB_copy_view[None, None, k + 1],
            )

        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def fmax_reduce(x: cute.TensorSSA, init_val: Float32 | None = None) -> Float32:
    """The max of `x` (and `init_val`, when given), over four independent chains for instruction-level parallelism."""
    res = cute.make_rmem_tensor(x.shape, Float32)
    res.store(x)
    local_max = [res[0], res[1], res[2], res[3]]

    for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        local_max[0] = cute.arch.fmax(local_max[0], res[i + 0])
        local_max[1] = cute.arch.fmax(local_max[1], res[i + 1])
        local_max[2] = cute.arch.fmax(local_max[2], res[i + 2])
        local_max[3] = cute.arch.fmax(local_max[3], res[i + 3])

    local_max[0] = cute.arch.fmax(local_max[0], local_max[1])
    local_max[2] = cute.arch.fmax(local_max[2], local_max[3])
    local_max[0] = cute.arch.fmax(local_max[0], local_max[2])

    return local_max[0] if const_expr(init_val is None) else cute.arch.fmax(local_max[0], init_val)


@dataclass
class Softmax(ParamsBase):
    """The online softmax state of a thread's accumulator rows: each row's running max (of the unscaled scores) and
    running sum of exp2(score * scale_log2 - max * scale_log2) over this thread's columns."""

    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor

    @staticmethod
    def create(scale_log2: Float32, num_rows: cutlass.Constexpr[int]) -> "Softmax":
        """The state of `num_rows` rows, with scores scaled by `scale_log2` (the softmax scale times log2(e))."""
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum)

    def reset(self) -> None:
        """Starts every row with no columns seen: max -inf, sum 0."""
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        """Folds the scores `acc_S` of one n block into the state, overwriting them with their exponentials, and
        returns each row's factor to rescale the output accumulated so far by (1 on the first block)."""
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)
        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2

        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur = fmax_reduce(acc_S_row, init_val=row_max[r] if const_expr(not is_first) else None)
            # A row's columns are spread over the 4 threads of a quad.
            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            row_max_prev = row_max[r]
            row_max[r] = row_max_cur

            # A fully masked row so far (max -inf) exponentiates against 0, so its scores stay 0 rather than NaN.
            if const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

            row_max_cur_scaled = row_max_cur * scale_log2
            acc_S_row_exp = cute.math.exp2(acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True)

            # The sum stays per-thread until `finalize` reduces it over the quad.
            if const_expr(is_first):
                acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32.zero, 0)
                row_scale[r] = 1.0
            else:
                row_scale[r] = cute.math.exp2((row_max_prev - row_max_cur) * scale_log2, fastmath=True)
                acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, row_sum[r] * row_scale[r], 0)

            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)

        return row_scale

    @cute.jit
    def finalize(self) -> cute.Tensor:
        """Returns each row's factor to normalize the output by (1 / its sum), and turns `row_sum` into the row's
        natural log-sum-exp of the scaled scores (-inf for a row with no unmasked column, whose output stays 0)."""
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2
        row_scale = cute.make_fragment_like(row_max, Float32)

        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            row_sum[r] = warp_reduce(row_sum[r], operator.add, threads_in_group=4)

            # A row whose sum is zero or NaN keeps its output (zero) unscaled.
            is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = cute.arch.rcp_approx(row_sum[r] if not is_zero_or_nan else 1.0)
            row_sum[r] = (
                (row_max[r] * scale_log2 + cute.math.log2(row_sum[r], fastmath=True)) * math.log(2.0)
                if not is_zero_or_nan
                else -Float32.inf
            )

        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        """Multiplies each row of accumulator `acc_O` by its factor in `row_scale`."""
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)

        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])


class AmpereAttentionForward:
    """The FlashAttention-2 forward on SM80: one CTA per `tile_m` query rows of a head, streaming K/V blocks through
    shared memory with an online softmax."""

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int | None = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        softmax_scale: float | None = None,
        causal_shift: int | None = None,
        scores: tuple = (),
        tile_m: int = 128,
        tile_n: int = 128,
        num_stages: int = 1,
        num_threads: int = 128,
        Q_in_regs: bool = False,
    ):
        """The forward for float16 or bfloat16 `dtype` Q, K, V and O with `head_dim` (Q and K) and `head_dim_v` (V and
        O, default `head_dim`), both multiples of 8, `qhead_per_kvhead` query heads per K/V head, causal or not, scores
        scaled by `softmax_scale` (default 1 / sqrt(head_dim)). Causal keeps key `k` for query `q` when
        `k <= q + causal_shift`, by default `seqlen_k - seqlen_q` (FlashAttention's bottom-right diagonal). `scores`
        are ops applied to each float32 score of Q K^T before the softmax, in order: `("round", dtype)` rounds to
        CuTe type `dtype`, `("mul", c, dtype)` rounds to `dtype` and multiplies by `c` (already of that type) there.
        Each CTA of `num_threads` threads computes `tile_m`
        query rows, over K/V blocks of `tile_n` rows double-buffered `num_stages` deep; `Q_in_regs` holds Q in
        registers, letting V reuse Q's shared memory."""
        self.dtype = dtype
        # Head dims padded to a multiple of 16, the MMA's k.
        self.tile_hdim = int(math.ceil(head_dim / 16) * 16)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / 16) * 16)
        # Without padding, no head-dim bounds checks (fewer registers).
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.causal_shift = causal_shift
        self.scores = tuple(scores)
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.Q_in_regs = Q_in_regs

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        Q_in_regs=False,
    ) -> bool:
        """Whether these parameters make a valid kernel: float16 or bfloat16, head dims multiples of 8, `tile_n` a
        multiple of 16, whole warps, tiles fitting in SM80 shared memory and `2 * tile_m` threads' multiple."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False

        if head_dim % 8 != 0 or head_dim_v % 8 != 0 or tile_n % 16 != 0 or num_threads % 32 != 0:
            return False

        smem_usage_Q = tile_m * head_dim * 2
        smem_usage_K = tile_n * head_dim * num_stages * 2
        smem_usage_V = tile_n * head_dim_v * num_stages * 2
        smem_usage_QV = (smem_usage_Q + smem_usage_V) if not Q_in_regs else max(smem_usage_Q, smem_usage_V)

        if smem_usage_QV + smem_usage_K > get_smem_capacity_in_bytes("sm_80"):
            return False

        return (tile_m * 2) % num_threads == 0

    def _setup_attributes(self):
        """Sets the shared-memory layouts of Q, K, V (`num_stages` each) and O, and the global-memory tiled copies:
        cp.async 128-bit loads of Q, K and V and 128-bit stores of O."""
        sQ_layout_atom = smem_layout_atom(self.dtype, self.tile_hdim)
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = smem_layout_atom(self.dtype, self.tile_hdimv)
        sO_layout_atom = sV_layout_atom

        self.sQ_layout = cute.tile_to_shape(sQ_layout_atom, (self.tile_m, self.tile_hdim), (0, 1))
        self.sK_layout = cute.tile_to_shape(sK_layout_atom, (self.tile_n, self.tile_hdim, self.num_stages), (0, 1, 2))
        self.sV_layout = cute.tile_to_shape(sV_layout_atom, (self.tile_n, self.tile_hdimv, self.num_stages), (0, 1, 2))
        self.sO_layout = cute.tile_to_shape(sO_layout_atom, (self.tile_m, self.tile_hdimv), (0, 1))

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        # Threads per row of a Q/K tile (and of a V/O tile) copy one 16-byte vector each.
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_threads % tQK_shape_dim_1 == 0, "num_threads must be divisible by tQK_shape_dim_1"
        tQK_layout = cute.make_ordered_layout((self.num_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0))
        # So loading Q never overshoots tile_m.
        assert self.tile_m % tQK_layout.shape[0] == 0
        tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tVO_layout = cute.make_ordered_layout((self.num_threads // tV_shape_dim_1, tV_shape_dim_1), order=(1, 0))
        # So storing O never overshoots tile_m.
        assert self.tile_m % tVO_layout.shape[0] == 0
        vQKVO_layout = cute.make_layout((1, async_copy_elems))

        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async_copy, tQK_layout, vQKVO_layout)
        self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(atom_async_copy, tQK_layout, vQKVO_layout)
        self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(atom_async_copy, tVO_layout, vQKVO_layout)
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tVO_layout, vQKVO_layout)

    def _get_tiled_mma(self):
        """The m16n8k16 MMAs of S = Q K^T and O = P V: the CTA's warps stacked along m, 16 rows each."""
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        """The CTA's shared memory: Q, K and V each 1024-byte aligned, or with `Q_in_regs` K and one region Q and V
        share."""
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]

        @cute.struct
        class SharedStorageQKV:
            """Separate Q, K and V tiles."""

            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

        @cute.struct
        class SharedStorageSharedQV:
            """K's tiles and one region for Q, then V."""

            sQ: sQV_struct
            sK: sK_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def compute_one_n_block(
        self,
        n_block: Int32,
        smem_pipe_read: Int32,
        smem_pipe_write: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        load_K: Callable,
        load_V: Callable,
        mask_fn: Callable | None = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        """S = Q K^T of K/V block `n_block` (in pipeline stage `smem_pipe_read`), masked by `mask_fn`, folded into the
        online softmax, and O += P V; meanwhile it loads the V block `num_stages` - 1 ahead and the next K block into
        stage `smem_pipe_write`."""

        def sync():
            """Waits for this block's K (or V) tile and for every thread to be past the stage it overwrites next."""
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 2)
            cute.arch.barrier()

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        sync()

        def load_V_next():
            """Starts loading the V block `num_stages` - 1 ahead (this block's when single-staged, then the only load
            of a partial block that needs predicates)."""
            if self.num_stages == 1 or n_block - self.num_stages + 1 >= 0:
                load_V(
                    n_block - self.num_stages + 1,
                    smem_pipe_write,
                    need_predicates=is_first_n_block and self.num_stages == 1,
                )

            cute.arch.cp_async_commit_group()

        load_V_next()
        gemm(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[
                None,
                None,
                None,
                smem_pipe_read if const_expr(self.num_stages > 1) else 0,
            ],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
            A_in_regs=self.Q_in_regs,
        )
        self.apply_scores(acc_S)
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        def load_K_next():
            """Starts loading the K block `num_stages` ahead (the next one when single-staged)."""
            if n_block - self.num_stages >= 0:
                load_K(n_block - self.num_stages, smem_pipe_write, need_predicates=False)

            cute.arch.cp_async_commit_group()

        # Single-staged, V's tile must land (and K's be read) before the next K overwrites it.
        if const_expr(self.num_stages == 1):
            sync()
            load_K_next()

        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        # P in the input dtype, laid out as the A operand of P V.
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)

        if const_expr(self.num_stages > 1):
            sync()
            load_K_next()

        gemm_rs(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            tOrP,
            mma_params.tOrVt,
            smem_copy_params.tOsVt[
                None,
                None,
                None,
                smem_pipe_read if const_expr(self.num_stages > 1) else 0,
            ],
            smem_copy_params.smem_thr_copy_V,
        )

    def shift(self, seqlen_q, seqlen_k):
        """The causal diagonal's shift: key `k` is seen by query `q` when `k <= q + shift`."""
        return self.causal_shift if const_expr(self.causal_shift is not None) else seqlen_k - seqlen_q

    @cute.jit
    def apply_scores(self, acc_S: cute.Tensor) -> None:
        """Applies the traced score ops (`scores`) to the float32 scores `acc_S` in place."""
        for op in self.scores:
            dtype = op[-1]

            # A float32 product of two 16-bit floats is exact, so rounding it once is the 16-bit product.
            if const_expr(op[0] == "round"):
                acc_S.store(acc_S.load().to(dtype).to(Float32))
            else:
                acc_S.store((acc_S.load().to(dtype).to(Float32) * Float32(op[1])).to(dtype).to(Float32))

    @cute.jit
    def get_n_block_min_max(self, seqlen_q: Int32, seqlen_k: Int32, m_block: Int32):
        """The K/V blocks [min, max) query block `m_block` attends to: all of them, or (causal) up to the diagonal of
        its last row."""
        n_block_max = cute.ceil_div(seqlen_k, self.tile_n)

        if const_expr(self.is_causal):
            n_idx_right = (m_block + 1) * self.tile_m + self.shift(seqlen_q, seqlen_k)
            n_block_max = cutlass.min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))

        return 0, n_block_max

    @cute.jit
    def get_n_block_min_causal_mask(
        self, seqlen_q: Int32, seqlen_k: Int32, m_block: Int32, n_block_min: Int32
    ) -> Int32:
        """The first K/V block (at least `n_block_min`) crossing query block `m_block`'s diagonal: blocks before it
        need no causal mask."""
        n_idx = m_block * self.tile_m + self.shift(seqlen_q, seqlen_k)
        return cutlass.max(n_block_min, n_idx // self.tile_n)

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.ThrMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
    ) -> None:
        """Sets to -inf the scores of block (`m_block`, `n_block`) past seqlen_k (`mask_seqlen`) or above the causal
        diagonal (`mask_causal`)."""
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS))
        # Thread 0's coordinates are compile-time constants: compare against them, with the limits shifted by this
        # thread's column offset.
        t0ScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cS))
        thr_col_offset = tScS_mn[0][1]

        # A CTA with no key (n_block_max = 0) masks block 0.
        if n_block < 0:
            n_block = 0

        seqlenk_col_limit = seqlen_k - n_block * self.tile_n - thr_col_offset

        if const_expr(not mask_causal):
            if const_expr(mask_seqlen):
                for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                    oob = t0ScS_mn[0, c][1] >= seqlenk_col_limit

                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
        else:
            causal_row_offset = 1 + self.shift(seqlen_q, seqlen_k) - n_block * self.tile_n - thr_col_offset

            for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                # The row's columns past its diagonal (and past seqlen_k) are masked.
                row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                col_limit_right = row_idx + causal_row_offset

                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)

                for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                    acc_S_mn[r, c] = -Float32.inf if t0ScS_mn[0, c][1] >= col_limit_right else acc_S_mn[r, c]

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        sO: cute.Tensor,
        seqlen_q: Int32,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        """Stores the CTA's output rows (in the input dtype, through shared `sO` for 16-byte stores) and their LSE,
        skipping rows past seqlen_q."""
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        # Every thread has finished reading V (and Q) before O overwrites Q's shared memory.
        cute.arch.barrier()
        smem_copy_atom_O = copy_utils.get_smem_store_atom(self.dtype)
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))

        if const_expr(mLSE is not None):
            gLSE = cute.local_tile(mLSE[None, head_idx, batch_idx], (self.tile_m,), (m_block,))
            # LSE broadcast along the head dim, so the accumulator's partition gives each thread its rows' entries.
            gLSE_expanded_layout = cute.append(gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,)))
            gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
            thr_mma = tiled_mma.get_slice(tidx)
            taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
            assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
            taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
            t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))

            # Only the threads holding column 0 write the LSE.
            if taccOcO[0][1] == 0:
                for m in cutlass.range(cute.size(taccOgLSE.shape[1]), unroll_full=True):
                    if t0accOcO[m, 0][0] < seqlen_q - m_block * self.tile_m - taccOcO[0][0]:
                        taccOgLSE[m, 0] = lse[m]

        mO_cur = mO[None, None, head_idx, batch_idx]
        # Every thread's O is in shared memory before any is read back.
        cute.arch.barrier()
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO, self.dtype)
        cute.autovec_copy(tOsO, tOrO)
        gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO = copy_utils.predicate_k(tOcO, limit=mO.shape[1])

        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            if t0OcO[0, rest_m, 0][0] < seqlen_q - m_block * self.tile_m - tOcO[0][0]:
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None] if const_expr(self.check_hdim_v_oob) else None,
                )

    @cute.jit
    def advance_pipeline(self, pipeline_index):
        """The pipeline stage after `pipeline_index`, wrapping at `num_stages`."""
        return pipeline_index + 1 if pipeline_index < self.num_stages - 1 else 0

    @cute.jit
    def load_Q(
        self,
        gmem_thr_copy: cute.TiledCopy,
        gQ: cute.Tensor,
        sQ: cute.Tensor,
        block: Int32,
        seqlen: Int32,
        headdim: Int32,
    ):
        """Starts loading query tile `gQ` (block `block`) into `sQ`, skipping rows past `seqlen` (left unwritten: their
        outputs are never stored)."""
        tQsQ, tQgQ = gmem_thr_copy.partition_D(sQ), gmem_thr_copy.partition_S(gQ)
        cQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = copy_utils.predicate_k(tQcQ, limit=headdim)

        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            # Thread 0's (compile-time) coordinates, against a limit shifted by this thread's offset.
            if t0QcQ[0, m, 0][0] < seqlen - block * self.tile_m - tQcQ[0][0]:
                cute.copy(
                    gmem_thr_copy,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None] if const_expr(self.check_hdim_oob) else None,
                )

    @cute.jit
    def load_K(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        tKcK: cute.Tensor,
        t0KcK: cute.Tensor,
        tKpK: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ):
        """Starts loading K block `block` into pipeline stage `smem_pipe_write`, skipping rows past `seqlen` when
        `need_predicates` (left stale: their scores are masked)."""
        # Whether the copy's rows tile tile_n evenly, or its last pass overshoots.
        is_even_n_smem_k = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0

        if const_expr(need_predicates or not is_even_n_smem_k):
            if const_expr(is_even_n_smem_k):
                seqlen_limit = seqlen - block * self.tile_n
            else:
                if const_expr(not need_predicates):
                    seqlen_limit = self.tile_n
                else:
                    seqlen_limit = cutlass.min(seqlen - block * self.tile_n, self.tile_n)

            seqlen_limit -= tKcK[0][0]

            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if t0KcK[0, n, 0][0] < seqlen_limit:
                    cute.copy(
                        gmem_tiled_copy,
                        tKgK[None, n, None, block],
                        tKsK[
                            None,
                            n,
                            None,
                            smem_pipe_write if const_expr(self.num_stages > 1) else 0,
                        ],
                        pred=tKpK[None, n, None] if const_expr(self.check_hdim_oob) else None,
                    )
        else:
            cute.copy(
                gmem_tiled_copy,
                tKgK[None, None, None, block],
                tKsK[
                    None,
                    None,
                    None,
                    smem_pipe_write if const_expr(self.num_stages > 1) else 0,
                ],
                pred=tKpK if const_expr(self.check_hdim_oob) else None,
            )

    @cute.jit
    def load_V(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        tVcV: cute.Tensor,
        t0VcV: cute.Tensor,
        tVpV: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ):
        """Starts loading V block `block` into pipeline stage `smem_pipe_write`, zero-filling rows past `seqlen` when
        `need_predicates` (P is 0 there, but stale V could hold NaN or inf)."""
        is_even_n_smem_v = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0

        if const_expr(need_predicates or not is_even_n_smem_v):
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                # When the copy overshoots tile_n, only its last pass checks it.
                if is_even_n_smem_v or n < cute.size(tVsV.shape[1]) - 1 or tVcV[0, n, 0][0] < self.tile_n:
                    predicate = tVpV[None, n, None] if const_expr(self.check_hdim_v_oob) else None

                    if const_expr(need_predicates):
                        seqlen_limit = seqlen - block * self.tile_n - tVcV[0][0]
                        predicate_n = t0VcV[0, n, 0][0] < seqlen_limit
                        predicate = cute.make_fragment_like(tVpV[None, 0, None])

                        for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                            for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                                predicate[i, k] = (
                                    tVpV[i, n, k] if const_expr(self.check_hdim_v_oob) else True
                                ) and predicate_n

                    cute.copy(
                        gmem_tiled_copy,
                        tVgV[None, n, None, block],
                        tVsV[
                            None,
                            n,
                            None,
                            smem_pipe_write if const_expr(self.num_stages > 1) else 0,
                        ],
                        pred=predicate,
                    )
        else:
            cute.copy(
                gmem_tiled_copy,
                tVgV[None, None, None, block],
                tVsV[
                    None,
                    None,
                    None,
                    smem_pipe_write if const_expr(self.num_stages > 1) else 0,
                ],
                pred=tVpV if const_expr(self.check_hdim_v_oob) else None,
            )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """One CTA: query rows `m_block` of one head and batch (the block index), attended over K/V blocks from the
        last to the first, then normalized and stored with their LSE."""
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()
        seqlen_q = mQ.shape[0]
        seqlen_k = mK.shape[0]
        n_block_min, n_block_max = self.get_n_block_min_max(seqlen_q, seqlen_k, m_block)
        # A CTA whose rows see no key (causal, seqlen_q > seqlen_k) still runs one fully masked block 0.
        n_block = cutlass.max(n_block_max - 1, 0)

        # ///////////////////////////////////////////////////////////////////////////////
        # This CTA's tiles of Q, K and V.
        # ///////////////////////////////////////////////////////////////////////////////
        blkQ_shape = (self.tile_m, self.tile_hdim)
        blkK_shape = (self.tile_n, self.tile_hdim)
        blkV_shape = (self.tile_n, self.tile_hdimv)
        head_idx_kv = head_idx // self.qhead_per_kvhead
        mQ_cur = mQ[None, None, head_idx, batch_idx]
        mK_cur = mK[None, None, head_idx_kv, batch_idx]
        mV_cur = mV[None, None, head_idx_kv, batch_idx]
        gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))
        gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))
        gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))

        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory.
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)

        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout)
        else:
            sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)

        # V viewed as (head_dim_v, tile_n), the B operand of P V.
        sVt = layout_utils.transpose_view(sV)

        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
        tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)

        # ///////////////////////////////////////////////////////////////////////////////
        # MMA partitions and the output accumulator.
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # ldmatrix copies from shared memory to the MMA's registers (V's transposed).
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_QK = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype)
        smem_copy_atom_V = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype)
        smem_thr_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_K = cute.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Bounds: head-dim predicates in registers, sequence bounds checked with `if` per row (fewer registers).
        # ///////////////////////////////////////////////////////////////////////////////
        cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
        tKcK = gmem_thr_copy_K.partition_S(cK)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)

        if const_expr(self.tile_hdim == self.tile_hdimv):
            tVcV = tKcK
            t0VcV = t0KcK
        else:
            cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
            tVcV = gmem_thr_copy_V.partition_S(cV)
            t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)

        tKpK = copy_utils.predicate_k(tKcK, limit=mK.shape[1])

        if const_expr(self.same_hdim_kv):
            tVpV = tKpK
        else:
            tVpV = copy_utils.predicate_k(tVcV, limit=mV.shape[1])

        # One row of state per accumulator row this thread holds.
        softmax = Softmax.create(
            Float32(self.softmax_scale * LOG2_E),
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
        )
        softmax.reset()

        mma_params = SimpleNamespace(
            thr_mma_qk=thr_mma_qk,
            thr_mma_pv=thr_mma_pv,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_Q=smem_thr_copy_Q,
            smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V,
            tSsQ=tSsQ,
            tSsK=tSsK,
            tOsVt=tOsVt,
        )
        load_K = partial(
            self.load_K,
            gmem_tiled_copy_K,
            tKgK,
            tKsK,
            tKcK,
            t0KcK,
            tKpK,
            seqlen=seqlen_k,
        )
        load_V = partial(
            self.load_V,
            gmem_tiled_copy_V,
            tVgV,
            tVsV,
            tVcV,
            t0VcV,
            tVpV,
            seqlen=seqlen_k,
        )
        mask_fn = partial(
            self.apply_mask,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            m_block=m_block,
            thr_mma=thr_mma_qk,
            mask_causal=self.is_causal,
        )
        compute_one_n_block = partial(
            self.compute_one_n_block,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            softmax=softmax,
            load_K=load_K,
            load_V=load_V,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Prologue: Q, then the last K/V blocks (the only ones that can be partial).
        # ///////////////////////////////////////////////////////////////////////////////
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        self.load_Q(gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen_q, headdim=mQ.shape[1])
        cute.arch.cp_async_commit_group()

        def preprocess_Q():
            """Waits for Q and, with `Q_in_regs`, loads it into the MMA's registers."""
            # Q's group is followed by every K and V stage's (2 * num_stages - 1), or with Q_in_regs by K's first
            # stage's alone (upstream waits for 2 * num_stages - 1 there too, reading Q early past one stage).
            cute.arch.cp_async_wait_group(1 if const_expr(self.Q_in_regs) else self.num_stages * 2 - 1)

            if const_expr(self.Q_in_regs):
                cute.arch.barrier()
                tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

        # With Q_in_regs: Q, one stage of K, Q into registers, then V (into Q's shared memory). Otherwise: Q, every
        # stage of K and V, then Q.
        if const_expr(self.Q_in_regs):
            load_K(n_block, smem_pipe_write=0, need_predicates=True)
            cute.arch.cp_async_commit_group()
            preprocess_Q()
            # Every thread has read Q before V overwrites it.
            cute.arch.barrier()

        for stage in cutlass.range_constexpr(self.num_stages):
            if const_expr(not self.Q_in_regs or stage > 0):
                if stage == 0 or n_block - stage >= 0:
                    load_K(
                        n_block - stage,
                        smem_pipe_write=stage,
                        need_predicates=stage == 0,
                    )

                cute.arch.cp_async_commit_group()

            if const_expr(stage < self.num_stages - 1):
                if stage == 0 or n_block - stage >= 0:
                    load_V(
                        n_block - stage,
                        smem_pipe_write=stage,
                        need_predicates=stage == 0,
                    )

                cute.arch.cp_async_commit_group()

        if const_expr(not self.Q_in_regs):
            preprocess_Q()

        # ///////////////////////////////////////////////////////////////////////////////
        # Mainloop: the last block masked to seqlen_k, then (causal) the blocks crossing the diagonal, then the rest
        # unmasked.
        # ///////////////////////////////////////////////////////////////////////////////
        smem_pipe_read = Int32(0)
        smem_pipe_write = Int32(self.num_stages - 1)
        compute_one_n_block(
            n_block,
            smem_pipe_read,
            smem_pipe_write,
            is_first_n_block=True,
            mask_fn=partial(mask_fn, mask_seqlen=True),
        )
        smem_pipe_read = self.advance_pipeline(smem_pipe_read)
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        if const_expr(self.is_causal):
            n_block_min_causal_mask = self.get_n_block_min_causal_mask(seqlen_q, seqlen_k, m_block, n_block_min)

            for n_tile in cutlass.range(n_block_max - 1 - n_block_min_causal_mask, unroll=1):
                n_block = n_block_max - 2 - n_tile
                compute_one_n_block(
                    n_block,
                    smem_pipe_read,
                    smem_pipe_write,
                    mask_fn=partial(mask_fn, mask_seqlen=True),
                )
                smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        for n_tile in cutlass.range(n_block, unroll=1):
            compute_one_n_block(
                n_block - n_tile - 1,
                smem_pipe_read,
                smem_pipe_write,
                is_first_n_block=False,
                mask_fn=partial(mask_fn, mask_seqlen=False),
            )
            smem_pipe_read = self.advance_pipeline(smem_pipe_read)
            smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        # Normalize the output by the row sums, which become the LSE.
        row_scale = softmax.finalize()
        softmax.rescale_O(acc_O, row_scale)

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue, staging O in Q's shared memory.
        # ///////////////////////////////////////////////////////////////////////////////
        sO = cute.make_tensor(sQ.iterator, sO_layout)
        self.epilogue(
            acc_O,
            softmax.row_sum,
            mO,
            mLSE,
            sO,
            seqlen_q,
            gmem_tiled_copy_O,
            tiled_mma_pv,
            tidx,
            m_block,
            head_idx,
            batch_idx,
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        """Launches the forward on `stream`: Q (batch, seqlen_q, heads, head_dim), K (batch, seqlen_k, kv_heads,
        head_dim), V (batch, seqlen_k, kv_heads, head_dim_v) and O (batch, seqlen_q, heads, head_dim_v), each with a
        contiguous last dimension, and float32 LSE (batch, heads, seqlen_q) or None."""
        if const_expr(not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type == self.dtype)):
            raise TypeError("Q, K, V and O must all be of the kernel's dtype")

        if const_expr(mLSE is not None and mLSE.element_type != Float32):
            raise TypeError("LSE must be float32")

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        self.shared_bytes = SharedStorage.size_in_bytes()

        # (batch, seqlen, heads, head_dim) viewed as (seqlen, head_dim, heads, batch), LSE as (seqlen_q, heads, batch).
        mQ, mK, mV, mO = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0])) for t in (mQ, mK, mV, mO)
        ]

        if const_expr(mLSE is not None):
            mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[2, 1, 0]))

        grid_dim = (
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
        )

        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )
