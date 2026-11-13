"""lumen's Ampere attention backward (FlashAttention-2): dQ, dK and dV of O = softmax(scores(Q K^T)) V from O, dO and
each query row's log-sum-exp, with warp-level tensor-core MMA (mma.sync), ldmatrix and cp.async.

Ported from FlashAttention's CuTe DSL SM80 backward (https://github.com/Dao-AILab/flash-attention, commit f63c472c,
BSD-3-Clause), the same three kernels: `FlashAttentionBackwardPreprocess` (`flash_bwd_preprocess.py`: dPsum = rowsum(dO
* O), the LSE in log2 units and a zeroed float32 dQ accumulator, each padded to whole query tiles),
`FlashAttentionBackwardSm80` (`flash_bwd.py`: one CTA per K/V block of a head computes its dK and dV over the query
blocks that see it and adds each block's dQ atomically into the accumulator, in register order) and the SM80 path of
`FlashAttentionBackwardPostprocess` (`flash_bwd_postprocess.py`: the accumulator, scaled, into dQ). The main kernel
runs with the SM120 config FlashAttention runs it with (64 x 64 tiles, 4 warps) unless given another. lumen's traced
score ops are applied to the scores (their scale `ds_scale` scales dK and dQ, as FlashAttention's softmax scale does)
and the causal diagonal may be shifted; tiles of rows past the sequences are zeroed in shared memory once, so a P of 0
there can't meet uninitialized NaNs. Dropped: grouped-query heads, score mods, local windows, variable lengths,
deterministic mode, swapped-operand MMAs, PDL and the tile scheduler.
"""

import math
from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import dsl_user_op

from lumen.codegen.attention.cuda.ampere.forward import LOG2_E, AmpereAttentionForward, gemm, smem_layout_atom
from lumen.kernels.cuda import copy_utils, layout_utils


@dsl_user_op
def atomic_add_fp32(a: Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Adds `a` to the float32 at `gmem_ptr` atomically."""
    nvvm.atomicrmw(op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=Float32(a).ir_value(loc=loc, ip=ip))


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    """The address of element `coord` of `x`."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


class AmpereAttentionBackward:
    """The FlashAttention-2 backward on SM80: one CTA per `tile_n` keys of a head, streaming query blocks through
    shared memory (Q and dO `stages_q` and `stages_do` deep) and adding each block's dQ into a float32 dQ."""

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        causal_shift: Optional[int] = None,
        scores: tuple = (),
        ds_scale: float = 1.0,
        tile_m: Optional[int] = None,
        tile_n: Optional[int] = None,
        stages_q: Optional[int] = None,
        stages_do: Optional[int] = None,
        num_threads: Optional[int] = None,
        atom_layout_m_sdp: Optional[int] = None,
        atom_layout_n_dkv: Optional[int] = None,
        atom_layout_m_dq: Optional[int] = None,
    ):
        """The backward for float16 or bfloat16 `dtype` Q, K, V, dO, dK and dV with `head_dim` (Q and K) and
        `head_dim_v` (V, default `head_dim`), multiples of 8, causal or not (key `k` seen by query `q` when `k <= q +
        causal_shift`, by default `seqlen_k - seqlen_q`), the traced score ops `scores` (as the forward's: `("round",
        dtype)` or `("mul", c, dtype)`) with `ds_scale` their scale (dK and dQ scale by it). Each CTA of `num_threads`
        threads takes `tile_n` keys and `tile_m` queries at a time; the warps are laid out `atom_layout_*` along the
        rows of S and dP, of dK and dV, and of dQ; Q and dO run `stages_q` and `stages_do` deep. A config left out is
        FlashAttention-2's for the head dim (`default_config`)."""
        self.dtype = dtype
        # Head dims padded to a multiple of 32 (FlashAttention's: dQ, dK and dV accumulate in 32-wide register
        # layouts).
        self.tile_hdim = int(math.ceil(head_dim / 32) * 32)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / 32) * 32)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.is_causal = is_causal
        self.causal_shift = causal_shift
        self.scores = tuple(scores)
        self.ds_scale = ds_scale
        given = dict(
            tile_m=tile_m,
            tile_n=tile_n,
            stages_q=stages_q,
            stages_do=stages_do,
            num_threads=num_threads,
            atom_layout_m_sdp=atom_layout_m_sdp,
            atom_layout_n_dkv=atom_layout_n_dkv,
            atom_layout_m_dq=atom_layout_m_dq,
        )
        config = {
            **self.default_config(max(self.tile_hdim, self.tile_hdimv)),
            **{k: v for k, v in given.items() if v is not None},
        }
        self.tile_m = config["tile_m"]
        self.tile_n = config["tile_n"]
        self.stages_q = config["stages_q"]
        self.stages_do = config["stages_do"]
        assert self.stages_q >= self.stages_do, "Q is loaded at least as far ahead as dO"
        self.num_threads = config["num_threads"]
        self.atom_layout_m_sdp = config["atom_layout_m_sdp"]
        self.atom_layout_n_dkv = config["atom_layout_n_dkv"]
        self.atom_layout_m_dq = config["atom_layout_m_dq"]

    @staticmethod
    def default_config(head_dim: int) -> dict:
        """FlashAttention-2's SM80 backward config for padded head dim `head_dim` (a multiple of 32) on a GPU of 144
        KB or more of shared memory (`run_mha_bwd_hdim*` in `csrc/flash_attn/src/flash_bwd_launch_template.h`, its
        head dims bucketed as there): 8 warps, Q double- and dO single-buffered; 128 x 128 tiles and warp layouts
        4/4/4 up to head dim 64, 64 x 128 and 2/4/4 up to 96, 64 x 128 and 2/4/2 up to 128, 64 x 64 and 4/2/2 above
        (4/4/4 where 64 doesn't divide the head dim; Q single-buffered above 192, where two stages don't fit an A100).
        """
        if head_dim <= 64:
            tiles = (128, 128, 4, 4, 4)
        elif head_dim <= 96:
            tiles = (64, 128, 2, 4, 4)
        elif head_dim <= 128:
            tiles = (64, 128, 2, 4, 2)
        else:
            tiles = (64, 64, 4, 2, 2) if head_dim % 64 == 0 else (64, 64, 4, 4, 4)

        tile_m, tile_n, sdp, dkv, dq = tiles

        return dict(
            tile_m=tile_m,
            tile_n=tile_n,
            stages_q=1 if head_dim > 192 else 2,
            stages_do=1,
            num_threads=256,
            atom_layout_m_sdp=sdp,
            atom_layout_n_dkv=dkv,
            atom_layout_m_dq=dq,
        )

    def shift(self, seqlen_q, seqlen_k):
        """The causal diagonal's shift: key `k` is seen by query `q` when `k <= q + shift`."""
        return self.causal_shift if const_expr(self.causal_shift is not None) else seqlen_k - seqlen_q

    # The forward's score ops and mask, which see the same S tiles.
    apply_scores = AmpereAttentionForward.apply_scores
    apply_mask = AmpereAttentionForward.apply_mask

    def _setup_attributes(self):
        """Sets the shared-memory layouts (Q and dO staged, K, V, P and dS one tile each, the LSE and dPsum a column
        per stage) and the global-memory tiled copies: cp.async 128-bit loads of Q, K, V, dO, the LSE and dPsum,
        128-bit stores of dK and dV, and the dQ accumulator's one-float atomics."""
        sQ_layout_atom = smem_layout_atom(self.dtype, self.tile_hdim)
        sV_layout_atom = smem_layout_atom(self.dtype, self.tile_hdimv)
        self.sQ_layout = cute.tile_to_shape(sQ_layout_atom, (self.tile_m, self.tile_hdim, self.stages_q), (0, 1, 2))
        self.sK_layout = cute.tile_to_shape(sQ_layout_atom, (self.tile_n, self.tile_hdim), (0, 1))
        self.sV_layout = cute.tile_to_shape(sV_layout_atom, (self.tile_n, self.tile_hdimv), (0, 1))
        self.sdO_layout = cute.tile_to_shape(sV_layout_atom, (self.tile_m, self.tile_hdimv, self.stages_do), (0, 1, 2))
        self.sPdS_layout = cute.tile_to_shape(
            smem_layout_atom(self.dtype, self.tile_n), (self.tile_m, self.tile_n), (0, 1)
        )
        # The LSE and delta, a column per stage, read as each row of S's accumulator (broadcast over its columns).
        self.sLSE_layout = cute.make_layout((self.tile_m, self.stages_q), stride=(1, cute.round_up(self.tile_m, 64)))
        self.sLSEMma_layout = cute.make_layout(
            (self.tile_m, self.tile_n, self.stages_q), stride=(1, 0, cute.round_up(self.tile_m, 64))
        )

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=universal_copy_bits
        )
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQK_layout = cute.make_ordered_layout((self.num_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0))
        # Whether each copy's rows tile the block's rows evenly (else only its last pass checks them).
        self.is_even_m_smem_q = self.tile_m % tQK_layout.shape[0] == 0
        self.is_even_n_smem_k = self.tile_n % tQK_layout.shape[0] == 0
        tVdO_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tVdO_layout = cute.make_ordered_layout((self.num_threads // tVdO_shape_dim_1, tVdO_shape_dim_1), order=(1, 0))
        self.is_even_n_smem_v = self.tile_n % tVdO_layout.shape[0] == 0
        self.is_even_m_smem_do = self.tile_m % tVdO_layout.shape[0] == 0
        vQKVdO_layout = cute.make_layout((1, async_copy_elems))

        self.gmem_tiled_copy_QK = cute.make_tiled_copy_tv(atom_async_copy, tQK_layout, vQKVdO_layout)
        self.gmem_tiled_copy_VdO = cute.make_tiled_copy_tv(atom_async_copy, tVdO_layout, vQKVdO_layout)
        self.gmem_tiled_copy_dK = cute.make_tiled_copy_tv(atom_universal_copy, tQK_layout, vQKVdO_layout)
        self.gmem_tiled_copy_dV = cute.make_tiled_copy_tv(atom_universal_copy, tVdO_layout, vQKVdO_layout)
        # The padded LSE and dPsum load 16 bytes a thread; dQ is added one float32 a thread, in register order.
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), Float32, num_bits_per_copy=128
            ),
            cute.make_layout(self.num_threads),
            cute.make_layout(128 // Float32.width),
        )
        self.gmem_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=Float32.width),
            cute.make_layout(self.num_threads),
            cute.make_layout(1),
        )

    def _get_tiled_mma(self):
        """The m16n8k16 MMAs of S = Q K^T and dP = dO V^T, of dV = P^T dO and dK = dS^T Q, and of dQ = dS K, each with
        the CTA's warps laid out by its `atom_layout_*`."""
        warps = self.num_threads // 32

        def tiled(rows):
            """The MMA with `rows` warps along M and the rest along N."""
            layout = (rows, warps // rows, 1)
            return cute.make_tiled_mma(
                warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
                layout,
                permutation_mnk=(layout[0] * 16, layout[1] * 16, 16),
            )

        return tiled(self.atom_layout_m_sdp), tiled(self.atom_layout_n_dkv), tiled(self.atom_layout_m_dq)

    def _get_shared_storage_cls(self):
        """The CTA's shared memory: K, V, the Q and dO stages (1024-byte aligned), the LSE and delta columns, and P
        and dS."""
        sK_struct, sV_struct, sQ_struct, sdO_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sK_layout, self.sV_layout, self.sQ_layout, self.sdO_layout)
        ]
        sLSE_struct = cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.sLSE_layout)], 128]
        sPdS_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sPdS_layout)], 128]

        @cute.struct
        class SharedStorage:
            """K, V, the staged Q and dO, the LSE and delta, P and dS."""

            sK: sK_struct
            sV: sV_struct
            sQ: sQ_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sLSE_struct
            sP: sPdS_struct
            sdS: sPdS_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Launches the backward on `stream`: Q (batch, seqlen_q, heads, head_dim), K (batch, seqlen_k, heads,
        head_dim), V (batch, seqlen_k, heads, head_dim_v) and dO (batch, seqlen_q, heads, head_dim_v); the
        preprocess's float32 LSE (log2) and dPsum (batch, heads, seqlen_q padded to tile_m); it adds into its float32
        dQ accumulator (batch, heads, padded seqlen_q * head_dim padded to 32) and writes dK and dV (as K and V)."""
        if const_expr(not (mQ.element_type == mK.element_type == mV.element_type == mdO.element_type == self.dtype)):
            raise TypeError("Q, K, V and dO must all be of the kernel's dtype")

        if const_expr(not (mLSElog2.element_type == mdPsum.element_type == mdQaccum.element_type == Float32)):
            raise TypeError("the LSE, dPsum and dQ accumulator must be float32")

        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        self.shared_bytes = SharedStorage.size_in_bytes()
        tiled_mma_sdp, tiled_mma_dkv, tiled_mma_dq = self._get_tiled_mma()
        grid_dim = (cute.ceil_div(cute.size(mK.shape[1]), self.tile_n), cute.size(mQ.shape[2]), cute.size(mQ.shape[0]))

        self.kernel(
            mQ,
            mK,
            mV,
            mdO,
            mLSElog2,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sdO_layout,
            self.sPdS_layout,
            self.sLSE_layout,
            self.sLSEMma_layout,
            self.gmem_tiled_copy_QK,
            self.gmem_tiled_copy_VdO,
            self.gmem_tiled_copy_dK,
            self.gmem_tiled_copy_dV,
            self.gmem_tiled_copy_LSE,
            self.gmem_tiled_copy_dQaccum,
            tiled_mma_sdp,
            tiled_mma_dkv,
            tiled_mma_dq,
            SharedStorage,
        ).launch(grid=grid_dim, block=[self.num_threads, 1, 1], smem=self.shared_bytes, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sLSEMma_layout: cute.Layout,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_VdO: cute.TiledCopy,
        gmem_tiled_copy_dK: cute.TiledCopy,
        gmem_tiled_copy_dV: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """One CTA: keys `n_block` of one head and batch (the block index), over every query block that sees them:
        dV += P^T dO, dK += dS^T Q, and dQ += dS K added into the dQ accumulator; then dK and dV are stored."""
        tidx, _, _ = cute.arch.thread_idx()
        n_block, head_idx, batch_idx = cute.arch.block_idx()
        seqlen_q = mQ.shape[1]
        seqlen_k = mK.shape[1]

        # The query blocks that see this key block: all of them, or (causal) from the one holding the first query
        # on or past its first key's diagonal.
        m_block_max = cute.ceil_div(seqlen_q, self.tile_m)
        m_block_min = 0

        if const_expr(self.is_causal):
            m_block_min = cutlass.max((n_block * self.tile_n - self.shift(seqlen_q, seqlen_k)) // self.tile_m, 0)

        mQ_cur = mQ[batch_idx, None, head_idx, None]
        mdO_cur = mdO[batch_idx, None, head_idx, None]
        mK_cur = mK[batch_idx, None, head_idx, None]
        mV_cur = mV[batch_idx, None, head_idx, None]
        # (tile_m, head_dim, m_block), (tile_n, head_dim) and so on.
        gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
        gdO = cute.local_tile(mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0))
        gLSE = cute.local_tile(mLSElog2[batch_idx, head_idx, None], (self.tile_m,), (None,))
        gdPsum = cute.local_tile(mdPsum[batch_idx, head_idx, None], (self.tile_m,), (None,))
        gdQaccum = cute.local_tile(mdQaccum[batch_idx, head_idx, None], (self.tile_m * self.tile_hdim,), (None,))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sdO = storage.sdO.get_tensor(sdO_layout)
        sP = storage.sP.get_tensor(sPdS_layout)
        sdS = storage.sdS.get_tensor(sPdS_layout)
        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sLSE_layout)
        sLSEMma = storage.sLSE.get_tensor(sLSEMma_layout)
        sdPsumMma = storage.sdPsum.get_tensor(sLSEMma_layout)
        # Transposed views for the dK, dV and dQ MMAs.
        sQt, sdOt, sKt, sPt, sdSt = [layout_utils.transpose_view(t) for t in (sQ, sdO, sK, sP, sdS)]

        gmem_thr_copy_QK = gmem_tiled_copy_QK.get_slice(tidx)
        gmem_thr_copy_VdO = gmem_tiled_copy_VdO.get_slice(tidx)
        tQgQ = gmem_thr_copy_QK.partition_S(gQ)
        tQsQ = gmem_thr_copy_QK.partition_D(sQ)
        tKgK = gmem_thr_copy_QK.partition_S(gK)
        tKsK = gmem_thr_copy_QK.partition_D(sK)
        tVgV = gmem_thr_copy_VdO.partition_S(gV)
        tVsV = gmem_thr_copy_VdO.partition_D(sV)
        tdOgdO = gmem_thr_copy_VdO.partition_S(gdO)
        tdOsdO = gmem_thr_copy_VdO.partition_D(sdO)
        gmem_thr_copy_lse = gmem_tiled_copy_LSE.get_slice(tidx)
        gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
        tLSEgLSE = gmem_thr_copy_lse.partition_S(gLSE)
        tLSEsLSE = gmem_thr_copy_lse.partition_D(sLSE)
        tLSEgdPsum = gmem_thr_copy_lse.partition_S(gdPsum)
        tLSEsdPsum = gmem_thr_copy_lse.partition_D(sdPsum)
        tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
        tLSEcLSE = gmem_thr_copy_lse.partition_S(cute.make_identity_tensor((self.tile_m,)))

        # Rows past the sequences and padded head-dim columns are never loaded: zeroed once, they stay finite, so a
        # P of 0 there keeps dK, dV and dQ clean.
        for t in (tQsQ, tKsK, tVsV, tdOsdO):
            zeros = cute.make_fragment_like(t, self.dtype)
            zeros.fill(0.0)
            cute.autovec_copy(zeros, t)

        thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
        thr_mma_dkv = tiled_mma_dkv.get_slice(tidx)
        thr_mma_dq = tiled_mma_dq.get_slice(tidx)
        acc_dK = cute.make_rmem_tensor(thr_mma_dkv.partition_shape_C((self.tile_n, self.tile_hdim)), Float32)
        acc_dV = cute.make_rmem_tensor(thr_mma_dkv.partition_shape_C((self.tile_n, self.tile_hdimv)), Float32)
        acc_dK.fill(0.0)
        acc_dV.fill(0.0)

        tSrQ = thr_mma_sdp.make_fragment_A(thr_mma_sdp.partition_A(sQ[None, None, 0]))
        tSrK = thr_mma_sdp.make_fragment_B(thr_mma_sdp.partition_B(sK))
        tdPrdO = thr_mma_sdp.make_fragment_A(thr_mma_sdp.partition_A(sdO[None, None, 0]))
        tdPrV = thr_mma_sdp.make_fragment_B(thr_mma_sdp.partition_B(sV))
        tdVrP = thr_mma_dkv.make_fragment_A(thr_mma_dkv.partition_A(sPt))
        tdVrdO = thr_mma_dkv.make_fragment_B(thr_mma_dkv.partition_B(sdOt[None, None, 0]))
        tdKrdS = thr_mma_dkv.make_fragment_A(thr_mma_dkv.partition_A(sdSt))
        tdKrQ = thr_mma_dkv.make_fragment_B(thr_mma_dkv.partition_B(sQt[None, None, 0]))
        tdQrdS = thr_mma_dq.make_fragment_A(thr_mma_dq.partition_A(sdS))
        tdQrK = thr_mma_dq.make_fragment_B(thr_mma_dq.partition_B(sKt))

        tSsLSEMma = layout_utils.reshape_acc_to_mn(thr_mma_sdp.partition_C(sLSEMma))[None, 0, None]
        tSsdPsumMma = layout_utils.reshape_acc_to_mn(thr_mma_sdp.partition_C(sdPsumMma))[None, 0, None]

        smem_copy_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype)
        smem_copy_atom_t = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype)
        smem_thr_copy_QdO = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma_sdp).get_slice(tidx)
        smem_thr_copy_KV = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma_sdp).get_slice(tidx)
        smem_thr_copy_PdSt = cute.make_tiled_copy_A(smem_copy_atom_t, tiled_mma_dkv).get_slice(tidx)
        smem_thr_copy_QdOt = cute.make_tiled_copy_B(smem_copy_atom_t, tiled_mma_dkv).get_slice(tidx)
        smem_thr_copy_dS = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma_dq).get_slice(tidx)
        smem_thr_copy_Kt = cute.make_tiled_copy_B(smem_copy_atom_t, tiled_mma_dq).get_slice(tidx)
        r2s_thr_copy_PdS = cute.make_tiled_copy_C(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width),
            tiled_mma_sdp,
        ).get_slice(tidx)

        cQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        tQcQ = gmem_thr_copy_QK.partition_S(cQ)
        t0QcQ = gmem_thr_copy_QK.get_slice(0).partition_S(cQ)

        if const_expr(self.tile_hdim == self.tile_hdimv):
            tdOcdO, t0dOcdO = tQcQ, t0QcQ
        else:
            cdO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
            tdOcdO = gmem_thr_copy_VdO.partition_S(cdO)
            t0dOcdO = gmem_thr_copy_VdO.get_slice(0).partition_S(cdO)

        d_head = mQ.shape[3]
        d_head_v = mdO.shape[3]
        tQpQ = copy_utils.predicate_k(tQcQ, limit=d_head)
        tdOpdO = tQpQ if const_expr(self.same_hdim_kv) else copy_utils.predicate_k(tdOcdO, limit=d_head_v)

        mma_params = SimpleNamespace(
            thr_mma_sdp=thr_mma_sdp,
            thr_mma_dkv=thr_mma_dkv,
            thr_mma_dq=thr_mma_dq,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tdPrdO=tdPrdO,
            tdPrV=tdPrV,
            tdVrP=tdVrP,
            tdVrdO=tdVrdO,
            tdKrdS=tdKrdS,
            tdKrQ=tdKrQ,
            tdQrdS=tdQrdS,
            tdQrK=tdQrK,
            acc_dK=acc_dK,
            acc_dV=acc_dV,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_QdO=smem_thr_copy_QdO,
            smem_thr_copy_KV=smem_thr_copy_KV,
            smem_thr_copy_PdSt=smem_thr_copy_PdSt,
            smem_thr_copy_QdOt=smem_thr_copy_QdOt,
            smem_thr_copy_dS=smem_thr_copy_dS,
            smem_thr_copy_Kt=smem_thr_copy_Kt,
            r2s_thr_copy_PdS=r2s_thr_copy_PdS,
            tSsQ=smem_thr_copy_QdO.partition_S(sQ),
            tSsK=smem_thr_copy_KV.partition_S(sK),
            tdPsdO=smem_thr_copy_QdO.partition_S(sdO),
            tdPsV=smem_thr_copy_KV.partition_S(sV),
            tSsLSEMma=tSsLSEMma,
            tSsdPsumMma=tSsdPsumMma,
            tPsP=r2s_thr_copy_PdS.partition_D(sP),
            tdSsdS=r2s_thr_copy_PdS.partition_D(sdS),
            tdVsPt=smem_thr_copy_PdSt.partition_S(sPt),
            tdVsdOt=smem_thr_copy_QdOt.partition_S(sdOt),
            tdKsdSt=smem_thr_copy_PdSt.partition_S(sdSt),
            tdKsQt=smem_thr_copy_QdOt.partition_S(sQt),
            tdQsdS=smem_thr_copy_dS.partition_S(sdS),
            tdQsKt=smem_thr_copy_Kt.partition_S(sKt),
        )
        load_Q_LSE = partial(
            self.load_rows,
            gmem_tiled_copy_QK,
            tQgQ,
            tQsQ,
            tQcQ,
            t0QcQ,
            tQpQ,
            self.check_hdim_oob,
            self.is_even_m_smem_q,
            self.stages_q,
            gmem_tiled_copy_LSE,
            tLSEgLSE,
            tLSEsLSE,
            tLSEcLSE,
            seqlen=seqlen_q,
        )
        load_dO_dPsum = partial(
            self.load_rows,
            gmem_tiled_copy_VdO,
            tdOgdO,
            tdOsdO,
            tdOcdO,
            t0dOcdO,
            tdOpdO,
            self.check_hdim_v_oob,
            self.is_even_m_smem_do,
            self.stages_do,
            gmem_tiled_copy_LSE,
            tLSEgdPsum,
            tLSEsdPsum,
            tLSEcLSE,
            seqlen=seqlen_q,
        )
        compute_one_m_block = partial(
            self.compute_one_m_block,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            gmem_thr_copy_dQaccum=gmem_thr_copy_dQaccum,
            tdQgdQaccum=tdQgdQaccum,
            load_Q_LSE=load_Q_LSE,
            load_dO_dPsum=load_dO_dPsum,
            m_block_max=m_block_max,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            d_head=d_head,
            n_block=n_block,
        )

        if m_block_min < m_block_max:
            # Prologue: this CTA's V and K, then the first stages of Q (with its LSE) and dO (with its delta).
            self.load_kv(
                gmem_thr_copy_VdO,
                tVgV,
                tVsV,
                n_block,
                seqlen_k,
                d_head_v,
                self.tile_hdimv,
                self.check_hdim_v_oob,
                self.is_even_n_smem_v,
            )
            self.load_kv(
                gmem_thr_copy_QK,
                tKgK,
                tKsK,
                n_block,
                seqlen_k,
                d_head,
                self.tile_hdim,
                self.check_hdim_oob,
                self.is_even_n_smem_k,
            )
            cute.arch.cp_async_commit_group()

            m_block = m_block_min

            for stage in cutlass.range_constexpr(self.stages_q):
                if const_expr(self.stages_q == 1 or stage < self.stages_q - 1):
                    if stage == 0 or m_block + stage < m_block_max:
                        load_Q_LSE(m_block + stage, stage)

                    cute.arch.cp_async_commit_group()

                if const_expr(stage < self.stages_do):
                    if stage == 0 or m_block + stage < m_block_max:
                        load_dO_dPsum(m_block + stage, stage)

                    cute.arch.cp_async_commit_group()

            # Mainloop over the query blocks that see this key block.
            smem_pipe_read_q = Int32(0)
            smem_pipe_read_do = Int32(0)
            smem_pipe_write_q = Int32(self.stages_q - 1)
            smem_pipe_write_do = Int32(0)

            for m_tile in cutlass.range(m_block_min, m_block_max, unroll=1):
                compute_one_m_block(m_tile, smem_pipe_read_q, smem_pipe_read_do, smem_pipe_write_q, smem_pipe_write_do)
                smem_pipe_read_q = self.advance_pipeline(smem_pipe_read_q, self.stages_q)
                smem_pipe_read_do = self.advance_pipeline(smem_pipe_read_do, self.stages_do)
                smem_pipe_write_q = self.advance_pipeline(smem_pipe_write_q, self.stages_q)
                smem_pipe_write_do = self.advance_pipeline(smem_pipe_write_do, self.stages_do)

        # dK carries the score ops' scale (the postprocess scales dQ).
        acc_dK.store(acc_dK.load() * Float32(self.ds_scale))

        # Epilogue, staging dK and dV in K's and V's shared memory.
        sdK = cute.make_tensor(sK.iterator, sK_layout)
        sdV = cute.make_tensor(sV.iterator, sV_layout)
        self.epilogue(
            acc_dK,
            acc_dV,
            mdK,
            mdV,
            sdK,
            sdV,
            gmem_tiled_copy_dK,
            gmem_tiled_copy_dV,
            tiled_mma_dkv,
            tidx,
            n_block,
            head_idx,
            batch_idx,
            seqlen_k,
            d_head,
            d_head_v,
        )

    @cute.jit
    def compute_one_m_block(
        self,
        m_block: Int32,
        smem_pipe_read_q: Int32,
        smem_pipe_read_do: Int32,
        smem_pipe_write_q: Int32,
        smem_pipe_write_do: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        gmem_thr_copy_dQaccum: cute.TiledCopy,
        tdQgdQaccum: cute.Tensor,
        load_Q_LSE: Callable,
        load_dO_dPsum: Callable,
        m_block_max: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        d_head: Int32,
        n_block: Int32,
    ):
        """One query block `m_block`: S = Q K^T (score ops, mask), P = exp(S - LSE), dP = dO V^T, dS = P (dP - dPsum);
        dV += P^T dO, dK += dS^T Q, and dQ = dS K added into the dQ accumulator; meanwhile the next Q and dO stages
        load."""
        stage_q = smem_pipe_read_q if const_expr(self.stages_q > 1) else 0
        stage_do = smem_pipe_read_do if const_expr(self.stages_do > 1) else 0

        def load_Q_next():
            """Starts loading the Q block (and its LSE) `stages_q` - 1 ahead (the next one when single-staged)."""
            m_block_next = m_block + (self.stages_q - 1 if const_expr(self.stages_q > 1) else 1)

            if m_block_next < m_block_max:
                load_Q_LSE(m_block_next, smem_pipe_write_q)

            cute.arch.cp_async_commit_group()

        def load_dO_next():
            """Starts loading the dO block (and its delta) `stages_do` ahead."""
            if m_block + self.stages_do < m_block_max:
                load_dO_dPsum(m_block + self.stages_do, smem_pipe_write_do)

            cute.arch.cp_async_commit_group()

        # S = Q K^T, the traced score ops, the mask, then P = exp(S - LSE) (the LSE in log2 units, from the
        # preprocess).
        acc_shape_SdP = mma_params.thr_mma_sdp.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_rmem_tensor(acc_shape_SdP, Float32)
        acc_S.fill(0.0)
        cute.arch.cp_async_wait_group(1 if const_expr(self.stages_q > 1) else 0)
        cute.arch.barrier()
        gemm(
            mma_params.thr_mma_sdp,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ[None, None, None, stage_q],
            smem_copy_params.tSsK,
            smem_copy_params.smem_thr_copy_QdO,
            smem_copy_params.smem_thr_copy_KV,
        )
        self.apply_scores(acc_S)
        tLSErLSE = cute.make_fragment_like(smem_copy_params.tSsLSEMma[None, 0])
        cute.autovec_copy(smem_copy_params.tSsLSEMma[None, stage_q], tLSErLSE)
        self.apply_mask(
            acc_S,
            seqlen_q,
            seqlen_k,
            m_block,
            n_block,
            mma_params.thr_mma_sdp,
            mask_seqlen=True,
            mask_causal=self.is_causal,
        )
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)

        for r in cutlass.range(cute.size(acc_S_mn, mode=[0]), unroll_full=True):
            acc_S_mn[r, None].store(cute.math.exp2(acc_S_mn[r, None].load() * LOG2_E - tLSErLSE[r], fastmath=True))

        # dP = dO V^T, then dS = P (dP - dPsum); the score ops' scale goes on dK and dQ.
        acc_dP = cute.make_rmem_tensor(acc_shape_SdP, Float32)
        acc_dP.fill(0.0)
        cute.arch.cp_async_wait_group(1 if const_expr(self.stages_do > 1) else 0)
        cute.arch.barrier()
        gemm(
            mma_params.thr_mma_sdp,
            acc_dP,
            mma_params.tdPrdO,
            mma_params.tdPrV,
            smem_copy_params.tdPsdO[None, None, None, stage_do],
            smem_copy_params.tdPsV,
            smem_copy_params.smem_thr_copy_QdO,
            smem_copy_params.smem_thr_copy_KV,
            hook_fn=load_Q_next if const_expr(self.stages_q > 1) else None,
        )
        tLSErdPsum = cute.make_fragment_like(smem_copy_params.tSsdPsumMma[None, 0])
        cute.autovec_copy(smem_copy_params.tSsdPsumMma[None, stage_do], tLSErdPsum)
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP)

        for r in cutlass.range(cute.size(acc_dP_mn, mode=[0]), unroll_full=True):
            acc_dP_mn[r, None].store(acc_S_mn[r, None].load() * (acc_dP_mn[r, None].load() - tLSErdPsum[r]))

        # P and dS in the input dtype, through shared memory for the dV, dK and dQ MMAs.
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        cute.copy(
            smem_copy_params.r2s_thr_copy_PdS, smem_copy_params.r2s_thr_copy_PdS.retile(rP), smem_copy_params.tPsP
        )
        rdS = cute.make_fragment_like(acc_dP, self.dtype)
        rdS.store(acc_dP.load().to(self.dtype))
        # Every thread's P is written before any reads it back.
        cute.arch.barrier()
        cute.copy(
            smem_copy_params.r2s_thr_copy_PdS, smem_copy_params.r2s_thr_copy_PdS.retile(rdS), smem_copy_params.tdSsdS
        )

        # dV += P^T dO.
        gemm(
            mma_params.thr_mma_dkv,
            mma_params.acc_dV,
            mma_params.tdVrP,
            mma_params.tdVrdO,
            smem_copy_params.tdVsPt,
            smem_copy_params.tdVsdOt[None, None, None, stage_do],
            smem_copy_params.smem_thr_copy_PdSt,
            smem_copy_params.smem_thr_copy_QdOt,
        )
        # Every thread's dS is written before any reads it back.
        cute.arch.barrier()

        def dQ_mma(hook_fn):
            """dQ = dS K for this query block, added into its tile of the dQ accumulator in register order (the
            postprocess reads it back the same way)."""
            acc_dQ = cute.make_rmem_tensor(
                mma_params.thr_mma_dq.partition_shape_C((self.tile_m, self.tile_hdim)), Float32
            )
            acc_dQ.fill(0.0)
            gemm(
                mma_params.thr_mma_dq,
                acc_dQ,
                mma_params.tdQrdS,
                mma_params.tdQrK,
                smem_copy_params.tdQsdS,
                smem_copy_params.tdQsKt,
                smem_copy_params.smem_thr_copy_dS,
                smem_copy_params.smem_thr_copy_Kt,
                hook_fn=hook_fn,
            )
            acc_dQ_atomic = gmem_thr_copy_dQaccum.retile(acc_dQ)
            tdQgdQaccum_atomic = tdQgdQaccum[None, None, m_block]

            for i in cutlass.range(cute.size(acc_dQ_atomic), unroll_full=True):
                atomic_add_fp32(acc_dQ_atomic[i], elem_pointer(tdQgdQaccum_atomic, i))

        # With one Q stage, dK goes first so the next Q block can start loading.
        if const_expr(self.stages_q > 1):
            dQ_mma(load_dO_next)

        # dK += dS^T Q.
        gemm(
            mma_params.thr_mma_dkv,
            mma_params.acc_dK,
            mma_params.tdKrdS,
            mma_params.tdKrQ,
            smem_copy_params.tdKsdSt,
            smem_copy_params.tdKsQt[None, None, None, stage_q],
            smem_copy_params.smem_thr_copy_PdSt,
            smem_copy_params.smem_thr_copy_QdOt,
            hook_fn=load_dO_next if const_expr(self.stages_q == 1) else None,
        )

        if const_expr(self.stages_q == 1):
            cute.arch.barrier()
            dQ_mma(load_Q_next)

    @cute.jit
    def epilogue(
        self,
        acc_dK: cute.Tensor,
        acc_dV: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sdK: cute.Tensor,
        sdV: cute.Tensor,
        gmem_tiled_copy_dK: cute.TiledCopy,
        gmem_tiled_copy_dV: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        seqlen_k: Int32,
        d_head: Int32,
        d_head_v: Int32,
    ):
        """Stores the CTA's dK and dV rows (in the input dtype, through shared `sdK` and `sdV` for 16-byte stores),
        skipping rows past seqlen_k."""
        rdV = cute.make_fragment_like(acc_dV, self.dtype)
        rdV.store(acc_dV.load().to(self.dtype))
        rdK = cute.make_fragment_like(acc_dK, self.dtype)
        rdK.store(acc_dK.load().to(self.dtype))
        gmem_thr_copy_dK = gmem_tiled_copy_dK.get_slice(tidx)
        gmem_thr_copy_dV = gmem_tiled_copy_dV.get_slice(tidx)

        # Every thread has finished reading K and V before dK and dV overwrite them.
        cute.arch.barrier()
        smem_copy_atom_dKV = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width
        )
        smem_thr_copy_dKV = cute.make_tiled_copy_C(smem_copy_atom_dKV, tiled_mma).get_slice(tidx)
        cute.copy(smem_copy_atom_dKV, smem_thr_copy_dKV.retile(rdV), smem_thr_copy_dKV.partition_D(sdV))
        cute.copy(smem_copy_atom_dKV, smem_thr_copy_dKV.retile(rdK), smem_thr_copy_dKV.partition_D(sdK))

        mdK_cur = mdK[batch_idx, None, head_idx, None]
        mdV_cur = mdV[batch_idx, None, head_idx, None]
        gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
        gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
        tdKsdK = gmem_thr_copy_dK.partition_S(sdK)
        tdKgdK = gmem_thr_copy_dK.partition_D(gdK)
        tdVsdV = gmem_thr_copy_dV.partition_S(sdV)
        tdVgdV = gmem_thr_copy_dV.partition_D(gdV)
        tdKrdK = cute.make_fragment_like(tdKgdK, self.dtype)
        tdVrdV = cute.make_fragment_like(tdVgdV, self.dtype)
        # Every thread's dK and dV are in shared memory before any are read back.
        cute.arch.barrier()
        cute.autovec_copy(tdKsdK, tdKrdK)
        cute.autovec_copy(tdVsdV, tdVrdV)

        cdK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
        tdKcdK = gmem_thr_copy_dK.partition_S(cdK)
        t0dKcdK = gmem_tiled_copy_dK.get_slice(0).partition_S(cdK)

        if const_expr(self.tile_hdim == self.tile_hdimv):
            tdVcdV, t0dVcdV = tdKcdK, t0dKcdK
        else:
            cdV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
            tdVcdV = gmem_thr_copy_dV.partition_S(cdV)
            t0dVcdV = gmem_tiled_copy_dV.get_slice(0).partition_S(cdV)

        tdKpdK = copy_utils.predicate_k(tdKcdK, limit=d_head)
        tdVpdV = tdKpdK if const_expr(self.same_hdim_kv) else copy_utils.predicate_k(tdVcdV, limit=d_head_v)

        for rest_m in cutlass.range_constexpr(cute.size(tdKrdK.shape[1])):
            if t0dKcdK[0, rest_m, 0][0] < seqlen_k - n_block * self.tile_n - tdKcdK[0][0]:
                cute.copy(
                    gmem_tiled_copy_dK,
                    tdKrdK[None, rest_m, None],
                    tdKgdK[None, rest_m, None],
                    pred=tdKpdK[None, rest_m, None] if const_expr(self.check_hdim_oob) else None,
                )

        for rest_m in cutlass.range_constexpr(cute.size(tdVrdV.shape[1])):
            if t0dVcdV[0, rest_m, 0][0] < seqlen_k - n_block * self.tile_n - tdVcdV[0][0]:
                cute.copy(
                    gmem_tiled_copy_dV,
                    tdVrdV[None, rest_m, None],
                    tdVgdV[None, rest_m, None],
                    pred=tdVpdV[None, rest_m, None] if const_expr(self.check_hdim_v_oob) else None,
                )

    @cute.jit
    def advance_pipeline(self, pipeline_index, stages: cutlass.Constexpr):
        """The pipeline stage after `pipeline_index`, wrapping at `stages`."""
        return pipeline_index + 1 if pipeline_index < stages - 1 else 0

    @cute.jit
    def load_kv(
        self,
        gmem_thr_copy: cute.TiledCopy,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        block: Int32,
        seqlen: Int32,
        headdim: Int32,
        tile_hdim: cutlass.Constexpr,
        check_hdim_oob: cutlass.Constexpr,
        is_even: cutlass.Constexpr,
    ):
        """Starts loading K or V block `block` (`tXgX` into `tXsX`), skipping rows past `seqlen` and columns past
        `headdim` (left zero)."""
        cX = cute.make_identity_tensor((self.tile_n, tile_hdim))
        tXcX = gmem_thr_copy.partition_S(cX)
        t0XcX = gmem_thr_copy.get_slice(0).partition_S(cX)
        tXpX = copy_utils.predicate_k(tXcX, limit=headdim)

        for n in cutlass.range_constexpr(cute.size(tXsX.shape[1])):
            # When the copy's rows overshoot the block, only its last pass checks them.
            if is_even or n < cute.size(tXsX.shape[1]) - 1 or tXcX[0, n, 0][0] < self.tile_n:
                predicate_n = t0XcX[0, n, 0][0] < seqlen - block * self.tile_n - tXcX[0][0]
                predicate = cute.make_fragment_like(tXpX[None, 0, None])

                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (tXpX[i, n, k] if const_expr(check_hdim_oob) else True) and predicate_n

                cute.copy(gmem_thr_copy, tXgX[None, n, None], tXsX[None, n, None], pred=predicate)

    @cute.jit
    def load_rows(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        tXcX: cute.Tensor,
        t0XcX: cute.Tensor,
        tXpX: cute.Tensor,
        check_hdim_oob: cutlass.Constexpr,
        is_even: cutlass.Constexpr,
        stages: cutlass.Constexpr,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        tLSEgX: cute.Tensor,
        tLSEsX: cute.Tensor,
        tLSEcLSE: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
    ):
        """Starts loading Q (or dO) block `block` into stage `smem_pipe_write`, skipping rows past `seqlen`, and its
        rows' LSE (or dPsum) beside it, the whole tile (the preprocess padded them to whole tiles)."""
        stage = smem_pipe_write if const_expr(stages > 1) else 0

        for m in cutlass.range_constexpr(cute.size(tXsX.shape[1])):
            if is_even or m < cute.size(tXsX.shape[1]) - 1 or tXcX[0, m, 0][0] < self.tile_m:
                predicate_m = t0XcX[0, m, 0][0] < seqlen - block * self.tile_m - tXcX[0][0]
                predicate = cute.make_fragment_like(tXpX[None, 0, None])

                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (tXpX[i, m, k] if const_expr(check_hdim_oob) else True) and predicate_m

                cute.copy(gmem_tiled_copy, tXgX[None, m, None, block], tXsX[None, m, None, stage], pred=predicate)

        for m in cutlass.range_constexpr(cute.size(tLSEsX.shape[1])):
            if tLSEcLSE[0, m][0] < self.tile_m:
                cute.copy(gmem_tiled_copy_LSE, tLSEgX[None, m, block], tLSEsX[None, m, stage])


class AmpereAttentionBackwardPreprocess:
    """FlashAttention's backward preprocess on SM80: per query tile of a head, dPsum = rowsum(dO * O), the LSE in log2
    units, and its tile of the dQ accumulator zeroed, each padded to whole tiles."""

    def __init__(self, dtype: Type[cutlass.Numeric], head_dim: int, head_dim_v: int, tile_m: int, num_threads: int):
        """The preprocess for float16 or bfloat16 `dtype` O and dO of head dim `head_dim_v`, and a dQ accumulator of
        `head_dim` (both padded to 32, as the main kernel's), `tile_m` query rows (the main kernel's) a CTA of
        `num_threads` (at least `tile_m`) threads."""
        self.dtype = dtype
        self.tile_m = tile_m
        self.tile_hdim = int(math.ceil(head_dim / 32) * 32)
        self.tile_hdimv = int(math.ceil(head_dim_v / 32) * 32)
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.num_threads = num_threads
        assert num_threads >= tile_m, "a thread a row of the LSE"

    def _setup_attributes(self):
        """The O and dO loads (128-bit vectors, a power of 2 threads a row so the row sums stay in a warp) and the dQ
        accumulator's 128-bit zeroing stores."""
        gmem_k_block_size = (
            128
            if self.tile_hdimv % 128 == 0
            else (64 if self.tile_hdimv % 64 == 0 else (32 if self.tile_hdimv % 32 == 0 else 16))
        )
        num_copy_elems = 128 // self.dtype.width
        threads_per_row = gmem_k_block_size // num_copy_elems
        self.gmem_tiled_copy_O = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, self.num_threads, num_copy_elems
        )
        num_copy_elems_dQaccum = 128 // Float32.width
        assert (self.tile_m * self.tile_hdim // num_copy_elems_dQaccum) % self.num_threads == 0
        self.gmem_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(Float32, self.num_threads, num_copy_elems_dQaccum)

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdQaccum: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Launches the preprocess on `stream`: O and dO (batch, seqlen_q, heads, head_dim_v), the forward's float32
        LSE (batch, heads, seqlen_q); writes float32 dPsum and LSE log2 (batch, heads, seqlen_q padded to tile_m) and
        zeroes the float32 dQ accumulator (batch, heads, padded seqlen_q * padded head_dim)."""
        self._setup_attributes()
        grid_dim = (cute.ceil_div(cute.size(mO.shape[1]), self.tile_m), cute.size(mO.shape[2]), cute.size(mO.shape[0]))
        self.kernel(
            mO, mdO, mLSE, mdPsum, mLSElog2, mdQaccum, self.gmem_tiled_copy_O, self.gmem_tiled_copy_dQaccum
        ).launch(grid=grid_dim, block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdQaccum: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
    ):
        """One CTA: query tile `m_block` of one head and batch: its rows' dPsum (0 past the queries), LSE log2 (+inf
        past the queries, 0 for a row that sees no key, so P is 0) and dQ accumulator tile (0)."""
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()
        seqlen_q = mO.shape[1]
        seqlen_limit = seqlen_q - m_block * self.tile_m
        headdim_v = mO.shape[3]

        gLSE = cute.local_tile(mLSE[batch_idx, head_idx, None], (self.tile_m,), (m_block,))
        gLSElog2 = cute.local_tile(mLSElog2[batch_idx, head_idx, None], (self.tile_m,), (m_block,))
        lse = Float32.inf

        if tidx < seqlen_limit and tidx < self.tile_m:
            lse = gLSE[tidx]

        blk_shape = (self.tile_m, self.tile_hdimv)
        gO = cute.local_tile(mO[batch_idx, None, head_idx, None], blk_shape, (m_block, 0))
        gdO = cute.local_tile(mdO[batch_idx, None, head_idx, None], blk_shape, (m_block, 0))
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOgO = gmem_thr_copy_O.partition_S(gO)
        tOgdO = gmem_thr_copy_O.partition_S(gdO)
        cO = cute.make_identity_tensor(blk_shape)
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cO)
        tOpO = copy_utils.predicate_k(tOcO, limit=headdim_v) if const_expr(self.check_hdim_v_oob) else None
        tOrO = cute.make_rmem_tensor_like(tOgO)
        tOrdO = cute.make_rmem_tensor_like(tOgdO)

        if const_expr(self.check_hdim_v_oob):
            tOrO.fill(0.0)
            tOrdO.fill(0.0)

        for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
            # Thread 0's (compile-time) row coordinates, against a limit shifted by this thread's.
            if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                tOpO_cur = tOpO[None, m, None] if const_expr(tOpO is not None) else None
                copy_utils.copy(tOgO[None, m, None], tOrO[None, m, None], pred=tOpO_cur)
                copy_utils.copy(tOgdO[None, m, None], tOrdO[None, m, None], pred=tOpO_cur)

        # Each thread's part of each row's sum over the head dim, then across the threads sharing the row.
        pdpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
            cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)
        )
        threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
        PdP_sum = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
        PdP_sum.store(pdpsum)

        for m in cutlass.range_constexpr(cute.size(PdP_sum)):
            PdP_sum[m] = warp_reduce_width(PdP_sum[m], threads_per_row)

        # The thread holding each row's column 0 writes its dPsum, 0 past the queries.
        gPdPsum = cute.local_tile(mdPsum[batch_idx, head_idx, None], (self.tile_m,), (m_block,))

        if tOcO[0, 0, 0][1] == 0:
            for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                row = tOcO[0, m, 0][0]
                value = Float32(0.0)

                if row < seqlen_limit:
                    value = PdP_sum[m]

                gPdPsum[row] = value

        # This tile of the dQ accumulator starts at 0.
        gdQaccum = cute.local_tile(mdQaccum[batch_idx, head_idx, None], (self.tile_m * self.tile_hdim,), (m_block,))
        tdQgdQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx).partition_S(gdQaccum)
        zero = cute.make_rmem_tensor_like(tdQgdQaccum)
        zero.fill(0.0)
        cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)

        # The LSE in log2 units: a row that sees no key (-inf) becomes 0, so its masked scores give P = 0.
        lse_log2 = lse * LOG2_E if lse != -Float32.inf else Float32(0.0)

        if tidx < self.tile_m:
            gLSElog2[tidx] = lse_log2


@cute.jit
def warp_reduce_width(val: Float32, width: cutlass.Constexpr[int]) -> Float32:
    """The sum of `val` over each aligned group of `width` lanes (a butterfly), every lane getting it."""
    for i in cutlass.range_constexpr(int(math.log2(width))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)

    return val


class AmpereAttentionBackwardPostprocess:
    """FlashAttention's backward postprocess on SM80: per query tile of a head, its dQ accumulator tile (in the main
    kernel's register order) scaled into dQ in the input dtype, through shared memory for 16-byte stores."""

    def __init__(
        self, dtype: Type[cutlass.Numeric], head_dim: int, tile_m: int, num_threads: int, atom_layout_m_dq: int
    ):
        """The postprocess writing float16 or bfloat16 `dtype` dQ of `head_dim`, `tile_m` query rows a CTA of
        `num_threads` threads with dQ's warp layout `atom_layout_m_dq`: all three the main kernel's, whose register
        order the accumulator holds."""
        self.dtype = dtype
        self.tile_m = tile_m
        self.tile_hdim = int(math.ceil(head_dim / 32) * 32)
        self.num_threads = num_threads
        self.atom_layout_m_dq = atom_layout_m_dq

    def _setup_attributes(self):
        """The dQ MMA (only its accumulator layout is used), the accumulator's loads (16-byte cp.async into shared
        memory, then one float a thread, as the main kernel added them), dQ's shared layout and its 16-byte stores."""
        warps = self.num_threads // 32
        layout = (self.atom_layout_m_dq, warps // self.atom_layout_m_dq, 1)
        self.tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            layout,
            permutation_mnk=(layout[0] * 16, layout[1] * 16, 16),
        )
        async_copy_elems_accum = 128 // Float32.width
        assert (self.tile_m * self.tile_hdim // async_copy_elems_accum) % self.num_threads == 0
        self.g2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), Float32, num_bits_per_copy=128
            ),
            cute.make_layout(self.num_threads),
            cute.make_layout(async_copy_elems_accum),
        )
        self.s2r_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(Float32, self.num_threads, 1)
        self.sdQaccum_layout = cute.make_layout(self.tile_m * self.tile_hdim)
        self.sdQ_layout = cute.tile_to_shape(
            smem_layout_atom(self.dtype, self.tiled_mma.get_tile_size(1)), (self.tile_m, self.tile_hdim), (0, 1)
        )
        num_copy_elems = 128 // self.dtype.width
        threads_per_row = math.gcd(128, self.tile_hdim) // num_copy_elems
        self.gmem_tiled_copy_dQ = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, self.num_threads, num_copy_elems
        )

    @cute.jit
    def __call__(self, mdQaccum: cute.Tensor, mdQ: cute.Tensor, scale: Float32, stream: cuda.CUstream):
        """Launches the postprocess on `stream`: the float32 dQ accumulator (batch, heads, padded seqlen_q * padded
        head_dim) times `scale` into dQ (batch, seqlen_q, heads, head_dim)."""
        self._setup_attributes()
        smem_size = max(
            cute.size_in_bytes(Float32, self.sdQaccum_layout), cute.size_in_bytes(self.dtype, self.sdQ_layout)
        )
        grid_dim = (
            cute.ceil_div(cute.size(mdQ.shape[1]), self.tile_m),
            cute.size(mdQ.shape[2]),
            cute.size(mdQ.shape[0]),
        )
        self.kernel(
            mdQaccum,
            mdQ,
            scale,
            self.tiled_mma,
            self.sdQaccum_layout,
            self.sdQ_layout,
            self.g2s_tiled_copy_dQaccum,
            self.s2r_tiled_copy_dQaccum,
            self.gmem_tiled_copy_dQ,
        ).launch(grid=grid_dim, block=[self.num_threads, 1, 1], smem=smem_size, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: Float32,
        tiled_mma: cute.TiledMma,
        sdQaccum_layout: cute.Layout,
        sdQ_layout: cute.ComposedLayout,
        g2s_tiled_copy_dQaccum: cute.TiledCopy,
        s2r_tiled_copy_dQaccum: cute.TiledCopy,
        gmem_tiled_copy_dQ: cute.TiledCopy,
    ):
        """One CTA: query tile `m_block` of one head and batch: its accumulator tile into shared memory, back into
        registers in the dQ MMA's accumulator layout, scaled and rounded, through shared memory into dQ's rows (those
        before seqlen_q)."""
        smem = cutlass.utils.SmemAllocator()
        sdQaccum = smem.allocate_tensor(Float32, sdQaccum_layout, byte_alignment=1024)
        sdQ = cute.make_tensor(cute.recast_ptr(sdQaccum.iterator, dtype=self.dtype), sdQ_layout)
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()
        seqlen_q = mdQ.shape[1]
        head_dim = mdQ.shape[3]

        gdQaccum = cute.local_tile(mdQaccum[batch_idx, head_idx, None], (self.tile_m * self.tile_hdim,), (m_block,))
        gdQ = cute.local_tile(mdQ[batch_idx, None, head_idx, None], (self.tile_m, self.tile_hdim), (m_block, 0))

        # The accumulator tile into shared memory.
        g2s_thr_copy_dQaccum = g2s_tiled_copy_dQaccum.get_slice(tidx)
        cute.copy(
            g2s_tiled_copy_dQaccum,
            g2s_thr_copy_dQaccum.partition_S(gdQaccum),
            g2s_thr_copy_dQaccum.partition_D(sdQaccum),
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # Back into the dQ MMA's accumulator layout (the order the main kernel added it in), scaled and rounded.
        tdQsdQaccum = s2r_tiled_copy_dQaccum.get_slice(tidx).partition_S(sdQaccum)
        acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C((self.tile_m, self.tile_hdim)), Float32)
        assert cute.size(acc) == cute.size(tdQsdQaccum)
        cute.autovec_copy(tdQsdQaccum, cute.make_tensor(acc.iterator, cute.make_layout(tdQsdQaccum.shape)))
        rdQ = cute.make_fragment_like(acc, self.dtype)
        rdQ.store((acc.load() * scale).to(self.dtype))

        # Every thread has read the accumulator before dQ overwrites the same shared memory.
        cute.arch.barrier()
        copy_atom_r2s_dQ = copy_utils.get_smem_store_atom(self.dtype)
        thr_copy_r2s_dQ = cute.make_tiled_copy_C(copy_atom_r2s_dQ, tiled_mma).get_slice(tidx)
        cute.copy(thr_copy_r2s_dQ, thr_copy_r2s_dQ.retile(rdQ), thr_copy_r2s_dQ.partition_D(sdQ))

        # Every thread's dQ is in shared memory before it is read back for 16-byte stores.
        cute.arch.barrier()
        gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_slice(tidx)
        tdQgdQ = gmem_thr_copy_dQ.partition_S(gdQ)
        tdQsdQ = gmem_thr_copy_dQ.partition_D(sdQ)
        tdQrdQ = cute.make_fragment_like(tdQsdQ, self.dtype)
        cute.autovec_copy(tdQsdQ, tdQrdQ)

        cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        tdQcdQ = gmem_thr_copy_dQ.partition_S(cdQ)
        tdQpdQ = copy_utils.predicate_k(tdQcdQ, limit=head_dim)

        for rest_m in cutlass.range(cute.size(tdQrdQ.shape[1]), unroll_full=True):
            if tdQcdQ[0, rest_m, 0][0] < seqlen_q - m_block * self.tile_m:
                cute.copy(
                    gmem_tiled_copy_dQ,
                    tdQrdQ[None, rest_m, None],
                    tdQgdQ[None, rest_m, None],
                    pred=tdQpdQ[None, rest_m, None],
                )
