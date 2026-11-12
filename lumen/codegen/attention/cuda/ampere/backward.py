"""lumen's Ampere attention backward (FlashAttention-2): dQ, dK and dV of O = softmax(scores(Q K^T)) V from dO, each
query row's log-sum-exp and delta = rowsum(dO * O), with warp-level tensor-core MMA (mma.sync), ldmatrix and cp.async.

Ported from FlashAttention's CuTe DSL SM80 backward (https://github.com/Dao-AILab/flash-attention, commit f63c472c,
BSD-3-Clause): `FlashAttentionBackwardSm80` in `flash_attn/cute/flash_bwd.py`, with the SM120 config FlashAttention
runs it with (64 x 64 tiles, 4 warps). One CTA per K/V block of a head computes its dK and dV over the query blocks
that see it and adds each block's dQ into a float32 dQ. lumen's attention supplies the LSE (natural log) and delta
itself, so FlashAttention's preprocess kernel isn't needed, and dQ is added at its own coordinates, so neither is its
postprocess: the LSE and delta are read per row (rows past the queries see LSE +inf, so P = 0), the traced score ops
are applied to the scores and dS is scaled by `ds_scale` as traced. Kept: causal masking (bottom-right aligned by
default, any shift), lengths that aren't tile multiples, head dims to 256. Dropped: grouped-query heads, score mods,
local windows, variable lengths, deterministic mode, swapped-operand MMAs and the tile scheduler.
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
        tile_m: int = 64,
        tile_n: int = 64,
        stages_q: Optional[int] = None,
        stages_do: Optional[int] = None,
        num_threads: int = 128,
        atom_layout_m_sdp: int = 4,
        atom_layout_n_dkv: int = 4,
        atom_layout_m_dq: int = 4,
    ):
        """The backward for float16 or bfloat16 `dtype` Q, K, V, dO, dK and dV with `head_dim` (Q and K) and
        `head_dim_v` (V, default `head_dim`), multiples of 8, causal or not (key `k` seen by query `q` when
        `k <= q + causal_shift`, by default `seqlen_k - seqlen_q`), the traced score ops `scores` (as the forward's:
        `("round", dtype)` or `("mul", c, dtype)`) and dS scaled by `ds_scale`. Each CTA of `num_threads` threads
        takes `tile_n` keys and `tile_m` queries at a time; the warps are laid out `atom_layout_*` along the rows of
        S and dP, of dK and dV, and of dQ. Q and dO run `stages_q` and `stages_do` deep, by default 2 up to head dim
        64 and 1 above, as FlashAttention runs it."""
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
        self.tile_m = tile_m
        self.tile_n = tile_n
        stages = 2 if max(head_dim, head_dim_v) <= 64 else 1
        self.stages_q = stages_q if stages_q is not None else stages
        self.stages_do = stages_do if stages_do is not None else stages
        assert self.stages_q >= self.stages_do, "Q is loaded at least as far ahead as dO"
        self.num_threads = num_threads
        self.atom_layout_m_sdp = atom_layout_m_sdp
        self.atom_layout_n_dkv = atom_layout_n_dkv
        self.atom_layout_m_dq = atom_layout_m_dq

    def shift(self, seqlen_q, seqlen_k):
        """The causal diagonal's shift: key `k` is seen by query `q` when `k <= q + shift`."""
        return self.causal_shift if const_expr(self.causal_shift is not None) else seqlen_k - seqlen_q

    # The forward's score ops and mask, which see the same S tiles.
    apply_scores = AmpereAttentionForward.apply_scores
    apply_mask = AmpereAttentionForward.apply_mask

    def _setup_attributes(self):
        """Sets the shared-memory layouts (Q and dO staged, K, V, P and dS one tile each, the LSE and delta a column
        per stage) and the global-memory tiled copies: cp.async 128-bit loads of Q, K, V and dO and 128-bit stores of
        dK and dV."""
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
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mdQ: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Launches the backward on `stream`: Q (batch, seqlen_q, heads, head_dim), K (batch, seqlen_k, heads,
        head_dim), V (batch, seqlen_k, heads, head_dim_v) and dO (batch, seqlen_q, heads, head_dim_v); float32 LSE
        and delta (batch, heads, seqlen_q); it adds into float32 dQ (batch, seqlen_q, heads, head_dim), zeroed by
        the caller, and writes dK and dV (as K and V). Each tensor's last dimension is contiguous."""
        if const_expr(not (mQ.element_type == mK.element_type == mV.element_type == mdO.element_type == self.dtype)):
            raise TypeError("Q, K, V and dO must all be of the kernel's dtype")

        if const_expr(not (mLSE.element_type == mDelta.element_type == mdQ.element_type == Float32)):
            raise TypeError("the LSE, delta and dQ must be float32")

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
            mLSE,
            mDelta,
            mdQ,
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
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mdQ: cute.Tensor,
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
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """One CTA: keys `n_block` of one head and batch (the block index), over every query block that sees them:
        dV += P^T dO, dK += dS^T Q, and dQ += dS K added into the float32 dQ; then dK and dV are stored."""
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
        mLSE_cur = mLSE[batch_idx, head_idx, None]
        mDelta_cur = mDelta[batch_idx, head_idx, None]
        mdQ_cur = mdQ[batch_idx, None, head_idx, None]
        mK_cur = mK[batch_idx, None, head_idx, None]
        mV_cur = mV[batch_idx, None, head_idx, None]
        # (tile_m, head_dim, m_block), (tile_n, head_dim) and so on.
        gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
        gdO = cute.local_tile(mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0))

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
            mLSE_cur,
            sLSE,
            Float32.inf,
            LOG2_E,
            tidx,
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
            mDelta_cur,
            sdPsum,
            0.0,
            1.0,
            tidx,
            seqlen=seqlen_q,
        )
        compute_one_m_block = partial(
            self.compute_one_m_block,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            mdQ_cur=mdQ_cur,
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
        mdQ_cur: cute.Tensor,
        load_Q_LSE: Callable,
        load_dO_dPsum: Callable,
        m_block_max: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        d_head: Int32,
        n_block: Int32,
    ):
        """One query block `m_block`: S = Q K^T (score ops, mask), P = exp(S - LSE), dP = dO V^T, dS = P (dP - delta)
        ds_scale; dV += P^T dO, dK += dS^T Q, and dQ = dS K added into dQ; meanwhile the next Q and dO stages load."""
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

        # S = Q K^T, the traced score ops, the mask, then P = exp(S - LSE) (the LSE stored in log2 units).
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

        # dP = dO V^T, then dS = P (dP - delta) ds_scale.
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
            acc_dP_mn[r, None].store(
                acc_S_mn[r, None].load() * (acc_dP_mn[r, None].load() - tLSErdPsum[r]) * Float32(self.ds_scale)
            )

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
            """dQ = dS K for this query block, added into dQ at its rows and columns (those inside the tensor)."""
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
            gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
            cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
            tdQgdQ = mma_params.thr_mma_dq.partition_C(gdQ)
            tdQcdQ = mma_params.thr_mma_dq.partition_C(cdQ)
            rows = seqlen_q - m_block * self.tile_m

            for i in cutlass.range(cute.size(acc_dQ), unroll_full=True):
                row, col = tdQcdQ[i]

                if row < rows and col < d_head:
                    atomic_add_fp32(acc_dQ[i], elem_pointer(tdQgdQ, i))

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
        mRow: cute.Tensor,
        sRow: cute.Tensor,
        fill: cutlass.Constexpr,
        scale: cutlass.Constexpr,
        tidx: Int32,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
    ):
        """Starts loading Q (or dO) block `block` into stage `smem_pipe_write`, skipping rows past `seqlen`, and
        stores its rows' LSE (or delta) times `scale` beside it, `fill` past `seqlen` or for -inf (+inf for the LSE:
        P = 0)."""
        stage = smem_pipe_write if const_expr(stages > 1) else 0

        for m in cutlass.range_constexpr(cute.size(tXsX.shape[1])):
            if is_even or m < cute.size(tXsX.shape[1]) - 1 or tXcX[0, m, 0][0] < self.tile_m:
                predicate_m = t0XcX[0, m, 0][0] < seqlen - block * self.tile_m - tXcX[0][0]
                predicate = cute.make_fragment_like(tXpX[None, 0, None])

                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (tXpX[i, m, k] if const_expr(check_hdim_oob) else True) and predicate_m

                cute.copy(gmem_tiled_copy, tXgX[None, m, None, block], tXsX[None, m, None, stage], pred=predicate)

        # One thread a row: the row's value, or `fill` past the sequence.
        for r in cutlass.range_constexpr(cute.ceil_div(self.tile_m, self.num_threads)):
            row = tidx + r * self.num_threads

            if row < self.tile_m:
                at = block * self.tile_m + row
                value = Float32(fill)

                if at < seqlen:
                    value = mRow[at] * Float32(scale)

                # A query that sees no key has LSE -inf: as +inf its P is 0 (rather than NaN), as on Triton.
                if value == -Float32.inf:
                    value = Float32(fill)

                sRow[row, stage] = value
