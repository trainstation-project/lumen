# Copyright (c) 2026, Tri Dao.
"""lumen's Ampere GEMM, D = A @ B with warp-level tensor-core MMA (mma.sync) and cp.async loads.

Written after QuACK's GEMM classes (https://github.com/Dao-AILab/quack, commit bb0e6b2, Apache-2.0): the constructor,
`_setup_tiled_mma`, `epi_retile_acc` and the scheduler setup are QuACK's (its `GemmSm80`,
`GemmSm90`, `GemmSm120` and `GemmBase`); QuACK's own SM80 mainloop and epilogue are unimplemented, so the kernel body
follows its SM120 warp-MMA consumer with the TMA pipelines swapped for SM80's: every CTA thread issues cp.async,
mma.sync and the epilogue's copies, with no warp specialization.
"""

from __future__ import annotations

import math
from typing import Callable, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.experimental import primitives as prims
from cutlass.pipeline import NamedBarrier
from cutlass.utils import LayoutEnum, get_smem_capacity_in_bytes

from lumen.kernels.cuda import copy_utils, layout_utils
from lumen.kernels.cuda.tile_scheduler import (
    PersistenceMode,
    RasterOrderOption,
    TileScheduler,
    TileSchedulerArguments,
)


class AmpereGemm:
    def __init__(
        self,
        accum_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: tuple[int, int] | tuple[int, int, int],
        is_persistent: bool = False,
        use_pdl: bool = False,
        NUM_WARPS: int | None = None,
        occupancy: int | None = None,
        arch: int = 80,
        aligned_mn: bool = False,
        aligned_k: bool = False,
    ):
        """The GEMM for `a_dtype` operands accumulating in `accum_dtype`, one CTA per (M, N) tile of `tile_shape_mnk`
        (K defaulting to four MMA instructions), with `NUM_WARPS` warps (4, or 8 for the widest tiles) and
        `occupancy` CTAs per SM. `aligned_mn` compiles it for M and N that are multiples of the tile's (no bounds
        checks on the rows and columns it loads and stores), `aligned_k` for such a K (no partial last k-tile)."""
        if arch not in (80, 86, 87, 89):
            raise ValueError(f"SM80-family GEMM supports arch in {self._supported_archs}, got {arch}")

        self.arch = arch
        self.aligned_mn = aligned_mn
        self.aligned_k = aligned_k
        self.accum_dtype = accum_dtype
        self.is_persistent = is_persistent
        self.use_pdl = use_pdl
        self.mma_inst_mnk = (16, 8, 16)

        self._set_cta_tile_shape(tile_shape_mnk)
        self._set_warps(NUM_WARPS)
        self.NUM_THREADS = self.NUM_WARPS * cute.arch.WARP_SIZE

        self.atom_layout_mnk = (2, self.NUM_WARPS >> 1, 1)

        self.smem_capacity = self._smem_capacity_for_arch(self.arch)
        self.buffer_align_bytes = 1024
        self.occupancy = self._default_occupancy(a_dtype) if occupancy is None else occupancy

        self.epilogue_barrier = NamedBarrier(barrier_id=1, num_threads=self.NUM_WARPS * cute.arch.WARP_SIZE)
        self.epi_m_major = True

    def _set_cta_tile_shape(self, tile_shape_mnk: tuple[int, int] | tuple[int, int, int]) -> None:
        if len(tile_shape_mnk) == 3:
            tile_m, tile_n, tile_k = tile_shape_mnk
        elif len(tile_shape_mnk) == 2:
            tile_m, tile_n = tile_shape_mnk
            tile_k = self.mma_inst_mnk[2] << 2
        else:
            raise ValueError("SM80 tile shape must be (M, N) or (M, N, K)")

        if tile_m % 16 != 0 or tile_n % 16 != 0:
            raise ValueError("SM80 tile shape M/N must be divisible by 16")

        self.cta_tile_shape_mnk = (tile_m, tile_n, tile_k)

    def _set_warps(self, NUM_WARPS: int | None) -> None:
        if NUM_WARPS is None:
            tile_m, tile_n, _ = self.cta_tile_shape_mnk

            if (tile_m, tile_n) in [(128, 256), (128, 192)]:
                NUM_WARPS = 8
            else:
                NUM_WARPS = 4

        if NUM_WARPS not in (4, 8):
            raise ValueError("SM80 GEMM supports NUM_WARPS=4 or 8")

        self.NUM_WARPS = NUM_WARPS

    def _default_occupancy(self, a_dtype) -> int:
        """CTAs per SM: two with 4 warps, one with 8, or one when two would not fit three stages of A and B tiles of
        dtype `a_dtype` in shared memory."""
        default_occupancy = 1 if self.NUM_WARPS == 8 else 2
        tile_m, tile_n, tile_k = self.cta_tile_shape_mnk
        ab_bytes_per_stage = (tile_m + tile_n) * tile_k * a_dtype.width // 8

        if 3 * ab_bytes_per_stage > self.smem_capacity // default_occupancy - self.buffer_align_bytes:
            default_occupancy = 1

        return default_occupancy

    def _smem_capacity_for_arch(self, arch: int) -> int:
        # CUDA documents CC 8.7 with the same shared-memory capacity as CC 8.0,
        # but this CUTLASS helper build does not accept "sm_87".
        if arch == 87:
            arch = 80

        return get_smem_capacity_in_bytes(f"sm_{arch}")

    def _setup_tiled_mma(self) -> None:
        """The CTA's tiled mma.sync: `atom_layout_mnk` warps of 16x8x16 instructions, each warp two instructions wide
        in N so its B fragments load with 4-matrix ldmatrix."""
        op = warp.MmaF16BF16Op(self.a_dtype, self.accum_dtype, self.mma_inst_mnk)
        atom_m, atom_n, atom_k = self.atom_layout_mnk
        tC = cute.make_layout(self.atom_layout_mnk)
        permutation_mnk = (
            atom_m * self.mma_inst_mnk[0],
            atom_n * self.mma_inst_mnk[1] * 2,
            atom_k * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)
        self.tiled_mma_gated_postact = self.tiled_mma

    def _setup_attributes(self) -> None:
        """The tiled MMA, epilogue subtile, stage counts and shared memory layouts for the traced operands."""
        assert not self.is_persistent, "lumen's SM80 GEMM launches one CTA per tile"
        self._setup_tiled_mma()

        self.epi_tile = self._compute_tile_shape_or_override(self.cta_tile_shape_mnk, self.atom_layout_mnk)
        self.epi_tile_shape = cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile)

        self.ab_stage, self.epi_stage = self._compute_stages(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.d_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        # The mainloop keeps ab_stage - 1 k-tiles in flight while it reads one.
        assert self.ab_stage >= 2, f"SM80 GEMM needs >= 2 A/B stages, shared memory fits {self.ab_stage}"

        tile_m, tile_n, tile_k = self.cta_tile_shape_mnk
        self.a_smem_layout_staged = self._make_smem_layout(
            self.a_dtype, self.a_layout, (tile_m, tile_k), self.ab_stage
        )
        self.b_smem_layout_staged = self._make_smem_layout(
            self.b_dtype, self.b_layout, (tile_n, tile_k), self.ab_stage
        )
        self.epi_smem_layout_staged = self._make_smem_layout(
            self.d_dtype, self.d_layout, self.epi_tile, self.epi_stage
        )

    @cute.jit
    def mma(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        k_tile_cnt: Int32,
        copy_A: Callable,
        copy_B: Callable,
        copy_block_A: Callable,
        copy_block_B: Callable,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
    ):
        """Multistage cp.async mainloop accumulating the tile's `k_tile_cnt` k-tiles into `acc`.

        `copy_A` / `copy_B` (k_tile, stage, pred_k) start a k-tile's gmem -> smem copy; `copy_block_A` / `copy_block_B`
        (stage, k_block) load one k16 block of a stage into the fragments. ab_stage - 1 k-tiles are kept in flight:
        k-tile t + ab_stage - 1 is issued into the stage k-tile t - 1 vacated, once every thread has moved past it.
        The register side has the produce rhythm of QuACK's SM120 warp-MMA mainloop: load block k + 1 (slot 0 of
        the next stage at the tile's last block), then MMA block k."""
        num_k_blocks = cute.size(tCrA, mode=[2])

        # Prologue: issue the first ab_stage - 1 k-tiles, one commit group each (empty past the last k-tile, so
        # the wait_group counts below hold for any k_tile_cnt). Only the last k-tile can run past K.
        for k_tile in cutlass.range_constexpr(self.ab_stage - 1):
            if const_expr(self.aligned_k):
                if k_tile < k_tile_cnt:
                    copy_A(k_tile, k_tile)
                    copy_B(k_tile, k_tile)
            elif k_tile < k_tile_cnt - 1:
                copy_A(k_tile, k_tile)
                copy_B(k_tile, k_tile)
            elif k_tile == k_tile_cnt - 1:
                copy_A(k_tile, k_tile, pred_k=True)
                copy_B(k_tile, k_tile, pred_k=True)

            cute.arch.cp_async_commit_group()

        read_stage = Int32(0)
        write_stage = Int32(self.ab_stage - 1)

        # Load first k-block
        cute.arch.cp_async_wait_group(self.ab_stage - 2)
        cute.arch.sync_threads()

        copy_block_A(read_stage, 0)
        copy_block_B(read_stage, 0)

        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            for k in cutlass.range_constexpr(num_k_blocks):
                if const_expr(k == num_k_blocks - 1):
                    # The next k-tile's stage: wait for its copy, and for every thread to finish reading the
                    # stage the copy below overwrites.
                    read_stage += 1
                    if read_stage == self.ab_stage:
                        read_stage = Int32(0)
                    cute.arch.cp_async_wait_group(self.ab_stage - 2)
                    cute.arch.sync_threads()

                # the wrap load is the NEXT tile's slot-0 preload
                k_next = 0 if k + 1 == num_k_blocks else k + 1
                copy_block_A(read_stage, k_next)
                copy_block_B(read_stage, k_next)

                if const_expr(k == 0):
                    next_k_tile = k_tile + self.ab_stage - 1
                    if const_expr(self.aligned_k):
                        if next_k_tile < k_tile_cnt:
                            copy_A(next_k_tile, write_stage)
                            copy_B(next_k_tile, write_stage)
                    else:
                        if next_k_tile < k_tile_cnt - 1:
                            copy_A(next_k_tile, write_stage)
                            copy_B(next_k_tile, write_stage)
                        if next_k_tile == k_tile_cnt - 1:
                            copy_A(next_k_tile, write_stage, pred_k=True)
                            copy_B(next_k_tile, write_stage, pred_k=True)
                    cute.arch.cp_async_commit_group()
                    write_stage += 1
                    if write_stage == self.ab_stage:
                        write_stage = Int32(0)

                cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)

        cute.arch.cp_async_wait_group(0)

    @cute.jit
    def epilogue(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        mD_mn: cute.Tensor,
        sD: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
        tidx: Int32,
    ):
        """Stores the tile's accumulator `acc` to D one epilogue subtile at a time: registers -> `sD` (converted to
        D's dtype), then `sD` -> gmem in 16-byte vectors along D's contiguous dimension."""
        # No stmatrix before SM90: each thread stores its C fragment of a subtile straight into `sD`, partitioned by
        # the tiled MMA as the accumulator is (MMA, MMA_M / EPI_M, MMA_N / EPI_N, STAGE).
        tCsD = tiled_mma.get_slice(tidx).partition_C(sD)
        # (MMA, MMA_M / EPI_M, MMA_N / EPI_N, (EPI_M, EPI_N)), and one subtile's fragment in D's dtype
        tRS_rAcc = self.epi_retile_acc(acc)
        tRS_rD = cute.make_rmem_tensor(tRS_rAcc[None, None, None, 0].shape, self.d_dtype)
        copy_D = self.epilog_gmem_copy_fn(mD_mn, sD, tile_coord_mnkl, tidx)

        epi_tile_shape = cute.zipped_divide(cute.make_layout(self.cta_tile_shape_mnk[:2]), self.epi_tile).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape, order=(0, 1) if const_expr(self.epi_m_major) else (1, 0)
        )
        epi_tile_num = cute.size(epi_tile_shape)

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)  # (epi_m, epi_n)
            tRS_rD.store(tRS_rAcc[None, None, None, epi_coord].load().to(self.d_dtype))
            epi_buffer = epi_idx % self.epi_stage

            # The buffer's previous subtile has left smem before it is overwritten.
            self.epilogue_barrier.arrive_and_wait()
            cute.autovec_copy(tRS_rD, tCsD[None, None, None, epi_buffer])
            self.epilogue_barrier.arrive_and_wait()
            copy_D(epi_buffer, epi_coord)

    @cute.jit
    def gmem_load_copy_fn(
        self,
        tiled_copy: cute.TiledCopy,
        mX: cute.Tensor,
        sX: cute.Tensor,
        tile_idx: Int32,
        tile_mn: int,
        tidx: Int32,
    ) -> Callable:
        """`copy(k_tile, stage, pred_k=False)`: the cp.async of k-tile `k_tile` of row tile `tile_idx` of `mX` (mn, k)
        into stage `stage` of `sX` (tile_mn, tile_k, STAGE). Vectors outside `mX` are zero in smem (extents are
        16-byte multiples, so a vector is wholly inside or outside): rows past MN are zeroed in every stage here, once
        per tile, and columns past K by `pred_k`, which only the last k-tile needs. An `aligned_mn` GEMM checks no
        rows."""
        tile_shape = (tile_mn, self.cta_tile_shape_mnk[2])
        # (tile_mn, tile_k, RestK)
        gX = cute.local_tile(mX, tile_shape, (tile_idx, None))
        cX = cute.local_tile(cute.make_identity_tensor(mX.shape), tile_shape, (tile_idx, None))
        thr_copy = tiled_copy.get_slice(tidx)
        # (CPY, CPY_MN, CPY_K, RestK) and (CPY, CPY_MN, CPY_K, STAGE)
        tXgX = thr_copy.partition_S(gX)
        tXcX = thr_copy.partition_S(cX)
        tXsX = thr_copy.partition_D(sX)
        zero = cute.make_rmem_tensor_like(tXsX[None, 0, 0, 0])
        zero.fill(0)

        limit_k = mX.shape[1]

        # MN bounds, one predicate per row of vectors, fixed for the tile (all true when `aligned_mn`).
        num_rows = cute.size(tXsX, mode=[1])
        tXpX = cute.make_rmem_tensor(num_rows, Boolean)
        if const_expr(self.aligned_mn):
            tXpX.fill(True)
        else:
            for i in cutlass.range_constexpr(num_rows):
                tXpX[i] = tXcX[0, i, 0, 0][0] < mX.shape[0]

        # cp.async never writes the rows past MN: zero them in every stage up front.
        if const_expr(not self.aligned_mn):
            for stage in cutlass.range_constexpr(cute.size(tXsX, mode=[3])):
                for i in cutlass.range_constexpr(num_rows):
                    if not tXpX[i]:
                        for j in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                            cute.autovec_copy(zero, tXsX[None, i, j, stage])

        def copy_fn(k_tile, stage, pred_k: cutlass.Constexpr[bool] = False):
            for i in cutlass.range_constexpr(num_rows):
                for j in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                    if tXpX[i]:
                        if const_expr(pred_k):
                            if tXcX[0, i, j, k_tile][1] < limit_k:
                                cute.copy(tiled_copy, tXgX[None, i, j, k_tile], tXsX[None, i, j, stage])
                            else:
                                cute.autovec_copy(zero, tXsX[None, i, j, stage])
                        else:
                            cute.copy(tiled_copy, tXgX[None, i, j, k_tile], tXsX[None, i, j, stage])

        return copy_fn

    @cute.jit
    def smem_load_copy_fn(
        self,
        make_tiled_copy: Callable,
        tiled_mma: cute.TiledMma,
        sX: cute.Tensor,
        tCrX: cute.Tensor,
        transpose: bool,
        tidx: Int32,
    ) -> Callable:
        """`copy_block(stage, k_block)`: ldmatrix (LDSM.T when `transpose`) of k16 block `k_block` of stage `stage` of
        `sX` into the MMA fragment `tCrX`; `make_tiled_copy` is `cute.make_tiled_copy_A` or `_B`."""
        atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose, 4), sX.element_type)
        smem_tiled_copy = make_tiled_copy(atom, tiled_mma)
        thr_copy = smem_tiled_copy.get_slice(tidx)
        # (CPY, CPY_MN, CPY_K, STAGE)
        tCsX_copy_view = thr_copy.partition_S(sX)
        tCrX_copy_view = thr_copy.retile(tCrX)

        def copy_block(stage, k_block):
            cute.copy(smem_tiled_copy, tCsX_copy_view[None, None, k_block, stage], tCrX_copy_view[None, None, k_block])

        return copy_block

    @cute.jit
    def epilog_gmem_copy_fn(
        self,
        mD_mn: cute.Tensor,
        sD: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
        tidx: Int32,
    ) -> Callable:
        """`copy_D(src_idx, dst_idx)`: stage `src_idx` of `sD` to epilogue subtile `dst_idx` (epi_m, epi_n) of D's
        tile `tile_coord_mnkl`, in 16-byte vectors along D's contiguous dimension, skipping those outside D (none when
        `aligned_mn`)."""
        tile_mn = self.cta_tile_shape_mnk[:2]
        gD = cute.local_tile(mD_mn, tile_mn, (tile_coord_mnkl[0], tile_coord_mnkl[1]))
        cD = cute.local_tile(cute.make_identity_tensor(mD_mn.shape), tile_mn, (tile_coord_mnkl[0], tile_coord_mnkl[1]))
        # (epi_M, epi_N, EPI_M, EPI_N)
        gD_epi = cute.flat_divide(gD, self.epi_tile)
        cD_epi = cute.flat_divide(cD, self.epi_tile)

        # Walk D along its contiguous dimension: an m-major D is copied through (n, m) views.
        major_size = self.epi_tile[1]
        if const_expr(self.d_layout.is_m_major_c()):
            gD_epi = layout_utils.select(gD_epi, [1, 0, 2, 3])
            cD_epi = layout_utils.select(cD_epi, [1, 0, 2, 3])
            sD = layout_utils.select(sD, [1, 0, 2])
            major_size = self.epi_tile[0]

        vec = 128 // self.d_dtype.width
        threads_per_row = math.gcd(major_size // vec, self.NUM_THREADS)
        tiled_copy = copy_utils.tiled_copy_2d(self.d_dtype, threads_per_row, self.NUM_THREADS, vec)
        thr_copy = tiled_copy.get_slice(tidx)
        # (CPY, CPY_R, CPY_C, STAGE) and (CPY, CPY_R, CPY_C, EPI_M, EPI_N)
        tDsD = thr_copy.partition_S(sD)
        tDgD = thr_copy.partition_D(gD_epi)
        tDcD = thr_copy.partition_S(cD_epi)

        def copy_fn(src_idx, dst_idx):
            for i in cutlass.range_constexpr(cute.size(tDsD, mode=[1])):
                for j in cutlass.range_constexpr(cute.size(tDsD, mode=[2])):
                    # The (n, m) views reorder the modes, not the coordinates: they stay (m, n).
                    if const_expr(self.aligned_mn):
                        cute.copy(tiled_copy, tDsD[None, i, j, src_idx], tDgD[None, i, j, dst_idx[0], dst_idx[1]])
                    elif cute.elem_less(tDcD[0, i, j, dst_idx[0], dst_idx[1]], mD_mn.shape):
                        cute.copy(tiled_copy, tDsD[None, i, j, src_idx], tDgD[None, i, j, dst_idx[0], dst_idx[1]])

        return copy_fn

    def epi_retile_acc(self, acc: cute.Tensor) -> cute.Tensor:
        """The accumulator (MMA, MMA_M, MMA_N) split by epilogue subtile, (MMA, MMA_M / EPI_M, MMA_N / EPI_N,
        (EPI_M, EPI_N)): each subtile's slice is the tiled MMA's C fragment of an epi_tile (QuACK's SM90 version
        reshapes for stmatrix's register order first; SM80 has no stmatrix)."""
        # (MMA, MMA_M / EPI_M, MMA_N / EPI_N)
        epi_acc_shape = (acc.shape[0], *cute.ceil_div(acc.shape[1:], self.epi_tile_shape))
        # (MMA, MMA_M / EPI_M, MMA_N / EPI_N, 1, EPI_M, EPI_N)
        acc_divide = cute.flat_divide(acc, epi_acc_shape)
        assert cute.size(acc_divide, mode=[3]) == 1
        return cute.group_modes(acc_divide[None, None, None, 0, None, None], 3, 5)

    def _compute_tile_shape_or_override(
        self,
        cta_tile_shape_mnk: tuple[int, int, int],
        atom_layout_mnk: tuple[int, int, int],
    ) -> tuple[int, int]:
        """The epilogue subtile: at most 64 x 32 of the CTA tile, covering the tiled MMA's warp spans (16 rows per
        M-warp, 16 columns per N-warp) so the r2s partition never wraps across warps."""
        m_span = atom_layout_mnk[0] << 4
        n_span = atom_layout_mnk[1] << 4
        tile_m = max(math.gcd(64, cta_tile_shape_mnk[0]), m_span)
        tile_n = max(math.gcd(32, cta_tile_shape_mnk[1]), n_span)
        return tile_m, tile_n

    @classmethod
    def _compute_stages(
        cls,
        cta_tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """(A/B stages, D epilogue stages) filling a CTA's share of shared memory: QuACK's split, with the A/B stages
        first and the leftover below one more A/B stage going to D."""
        epi_tiles = (cta_tile_shape_mnk[0] * cta_tile_shape_mnk[1]) // cute.size(cute.shape(epi_tile))
        epi_stage = min(epi_tiles, 4 if epi_tile[1] <= 16 else 2)
        epi_bytes_per_stage = cute.size(cute.shape(epi_tile)) * d_dtype.width // 8
        epi_bytes = epi_bytes_per_stage * epi_stage

        a_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = cute.size(a_shape) * a_dtype.width // 8 + cute.size(b_shape) * b_dtype.width // 8
        mbar_helpers_bytes = 1024

        remaining_bytes = smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ab_stage = remaining_bytes // ab_bytes_per_stage

        leftover = remaining_bytes - ab_bytes_per_stage * ab_stage
        epi_stage += min(leftover // epi_bytes_per_stage, epi_tiles - epi_stage)
        return ab_stage, epi_stage

    def _make_smem_layout(
        self,
        dtype: Type[cutlass.Numeric],
        layout: LayoutEnum,
        shape: tuple[int, int],
        stage: int,
    ) -> cute.ComposedLayout:
        """`stage` swizzled (rows, cols) tiles in shared memory, for A (tile_m, tile_k), B (tile_n, tile_k) or the
        epilogue's D (epi_m, epi_n): contiguous along cols for a row-major `layout`, along rows otherwise, as the
        tensor is in global memory. The atom is 8 lines of the widest of 128, 64, 32 or 16 bytes the tile's contiguous
        side holds, XOR-swizzled `Sw<B,4,3>` (B = 3, 2, 1, 0) so the 8 lines of an ldmatrix read hit distinct banks;
        the atoms tile the other side first, then the contiguous one, then the stages."""
        row_major = layout == LayoutEnum.ROW_MAJOR
        contiguous_bits = (shape[1] if row_major else shape[0]) * dtype.width
        swizzle_bits = next(b for b in (3, 2, 1, 0) if contiguous_bits % (128 << b) == 0)
        line = (128 << swizzle_bits) // dtype.width

        if row_major:
            atom = cute.make_layout((8, line), stride=(line, 1))
        else:
            atom = cute.make_layout((line, 8), stride=(1, line))

        atom = cute.make_composed_layout(cute.make_swizzle(swizzle_bits, 4, 3), 0, atom)
        return cute.tile_to_shape(atom, (*shape, stage), order=(0, 1, 2) if row_major else (1, 0, 2))

    def _make_gmem_tiled_copy_AB(self, dtype, major_mode, tile_mn, copy_bits=128):
        """cp.async tiled copy of a (tile_mn, tile_k) operand tile by all CTA threads in `copy_bits` vectors along its
        contiguous dimension, at most a 128-byte cache line per row of threads (QuACK's `_make_gmem_tiled_copy_A`
        with the tile's MN extent passed in, for A and B)."""
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=copy_bits,
        )
        copy_elems = copy_bits // dtype.width
        loads_per_cache_line = 128 * 8 // copy_bits  # 128 bytes per cache line
        shape_dim_1 = cute.size(self.cta_tile_shape_mnk[2]) // copy_elems
        if shape_dim_1 > loads_per_cache_line:
            shape_dim_1 = math.gcd(shape_dim_1, loads_per_cache_line)
        # thread layout for copy
        thread_layout = cute.make_layout((self.NUM_THREADS // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1))
        if major_mode != LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(tile_mn) // copy_elems
            if shape_dim_0 > loads_per_cache_line:
                shape_dim_0 = math.gcd(shape_dim_0, loads_per_cache_line)
            thread_layout = cute.make_layout((shape_dim_0, self.NUM_THREADS // shape_dim_0), stride=(1, shape_dim_0))
        # Value layout for copy
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_async_copy, thread_layout, value_layout)

    def _batch_last(self, t: cute.Tensor) -> cute.Tensor:
        """Batched `t` (l, x, y) as (x, y, l), or unbatched `t` (x, y) with a size-1 batch (lumen, after QuACK's
        GemmBase.permute_batch_last)."""
        if cutlass.const_expr(cute.rank(t) == 3):
            return layout_utils.select(t, [1, 2, 0])
        return layout_utils.expand(t, 2, 1)

    def _scheduler_params(self, mA: cute.Tensor, mB: cute.Tensor, mD: cute.Tensor) -> TileScheduler.Params:
        """The tile scheduler's parameters for A (m, k, l), B (n, k, l) and D (m, n, l): one CTA per tile, grouped by 8
        along the heuristic's raster order."""
        tile_sched_args = TileSchedulerArguments(
            problem_shape_ntile_mnl=(
                cute.ceil_div(cute.size(mA, mode=[0]), self.cta_tile_shape_mnk[0]),
                cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1]),
                mD.shape[2],
            ),
            raster_order=RasterOrderOption.Heuristic,
            group_size=Int32(8),
            # SM8x launches independent CTAs: single-CTA clusters.
            cluster_shape_mnk=(1, 1, 1),
            persistence_mode=PersistenceMode.NONE,
        )

        return TileScheduler.to_underlying_arguments(tile_sched_args)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        mA_mkl: cute.Tensor,
        mB_nkl: cute.Tensor,
        mD_mnl: cute.Tensor,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        TileSchedulerCls: cutlass.Constexpr[Callable],
    ):
        """One CTA's GEMM: per scheduled (m, n, batch) tile, the A/B mainloop into a float32 accumulator, then the
        epilogue storing it to D. Its runtime parameters are the three tensors alone (the rest is static), so Rust
        launches it with a plain `cuLaunchKernel`; the scheduler's parameters are built here from their shapes."""
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)

        # Make fragments
        thr_mma = tiled_mma.get_slice(tidx)

        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C(self.cta_tile_shape_mnk[:2]), self.accum_dtype)
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA)[None, None, None, 0])
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB)[None, None, None, 0])

        # ldmatrix s2r loads; the MMA's operands are K-major, the smem major only picks LDSM vs LDSM.T
        copy_block_A = self.smem_load_copy_fn(
            cute.make_tiled_copy_A, tiled_mma, sA, tCrA, self.a_layout.is_m_major_a(), tidx
        )

        copy_block_B = self.smem_load_copy_fn(
            cute.make_tiled_copy_B, tiled_mma, sB, tCrB, self.b_layout.is_n_major_b(), tidx
        )

        k_tile_cnt = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_tile_shape_mnk[2])

        # PDL: wait for the prior kernel before any gmem reads.
        if const_expr(self.use_pdl):
            prims.griddepcontrol(prims.GridDepAction.WAIT)

        tile_scheduler = TileSchedulerCls.create(self._scheduler_params(mA_mkl, mB_nkl, mD_mnl))
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
            tile_coord_mnkl = work_tile.tile_idx
            batch_idx = tile_coord_mnkl[3]

            copy_A = self.gmem_load_copy_fn(
                tiled_copy_A, mA_mkl[None, None, batch_idx], sA, tile_coord_mnkl[0], self.cta_tile_shape_mnk[0], tidx
            )

            copy_B = self.gmem_load_copy_fn(
                tiled_copy_B, mB_nkl[None, None, batch_idx], sB, tile_coord_mnkl[1], self.cta_tile_shape_mnk[1], tidx
            )

            acc.fill(0.0)
            self.mma(tiled_mma, acc, k_tile_cnt, copy_A, copy_B, copy_block_A, copy_block_B, tCrA, tCrB)

            self.epilogue(tiled_mma, acc, mD_mnl[None, None, batch_idx], sD, tile_coord_mnkl, tidx)

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        """Writes `mA @ mB` into `mD`, crossing as the caller holds them: A (l, m, k), B (l, k, n) and D (l, m, n), or
        unbatched (m, k), (k, n) and (m, n), each with any one dimension contiguous. lumen only compiles through this
        (`launch.compile`) and launches the kernel from Rust."""
        # Tensors arrive batch-first: rotate (l, x, y) -> (x, y, l) at trace time, a trivial batch appended to rank-2
        # operands, and B relabelled from the caller's (k, n) to the kernel's (n, k) (QuACK's b_transposed).
        mA, mB, mD = [self._batch_last(t) for t in (mA, mB, mD)]
        mB = layout_utils.select(mB, [1, 0, 2])

        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type

        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD)

        if const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        if const_expr(self.a_dtype.width != 16):
            raise TypeError("SM80 GEMM takes float16 or bfloat16 operands")

        self._setup_attributes()

        tiled_copy_A = self._make_gmem_tiled_copy_AB(self.a_dtype, self.a_layout, self.cta_tile_shape_mnk[0])
        tiled_copy_B = self._make_gmem_tiled_copy_AB(self.b_dtype, self.b_layout, self.cta_tile_shape_mnk[1])

        grid = TileScheduler.get_grid_shape(self._scheduler_params(mA, mB, mD), Int32(0))

        @cute.struct
        class SharedStorage:
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, cute.cosize(self.epi_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.shared_bytes = SharedStorage.size_in_bytes()

        self.kernel(
            self.tiled_mma,
            mA,
            mB,
            mD,
            tiled_copy_A,
            tiled_copy_B,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            TileScheduler,
        ).launch(
            grid=grid,
            block=[self.NUM_THREADS, 1, 1],
            smem=self.shared_bytes,
            stream=stream,
            # occupancy > 1 needs the launch bound so ptxas caps registers for 2 resident CTAs
            min_blocks_per_mp=self.occupancy,
            use_pdl=self.use_pdl,
        )
