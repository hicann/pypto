# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python API declarations for the PyPTO VF namespace (``pl.vf.xxx``).

These declarations exist so that:
- IDE "Go to Definition" works for every ``pl.vf.xxx`` call
- Python catches typos at import time
- Type checkers can validate argument types
- Docstrings document the user-facing calling convention

None of these functions are meant to be called at runtime.  Inside a PyPTO
kernel the AST parser intercepts every ``pl.vf.xxx`` call before Python executes
it.  Outside a kernel, calling a declaration raises ``RuntimeError``.
"""

from __future__ import annotations

from typing import Optional

from pypto.ir import (
    BinType,
    CastLayout,
    DataCopyMode,
    DuplicatePos,
    HistType,
    IndexOrder,
    LoadDist,
    MaskPattern,
    MaskWidth,
    MemBarMode,
    MergeMode,
    PackPart,
    SaturateMode,
    SqueezeMode,
    StoreDist,
    VFRoundMode,
)

from ._api import DType, _api_decl

# ===================================================================
# VF namespace (``pl.vf.*``)
# ===================================================================


class Vf:
    """Vector Function unit operations (A5 architecture).

    Used inside ``@pl.vector_function`` decorated functions.  Compute ops
    produce a result and must use the assignment form ``dst = vf.xxx(...)``;
    the destination register is declared implicitly.  Only store/side-effect
    ops are called as bare statements.
    """

    @staticmethod
    @_api_decl
    def create_mask(pattern: Optional[MaskPattern] = None,
                    dtype: Optional[DType] = None):
        """Create and initialize a VF predicate mask register.

        Both kwargs are optional and may be given independently:
            preg = vf.create_mask(dtype=pl.DT_FP16)          # pattern defaults to ALL
            preg = vf.create_mask(pattern=pl.MaskPattern.VL8) # dtype defaults to FP32
            preg = vf.create_mask()                           # both defaults

        Args:
            pattern: Mask pattern, ``pl.MaskPattern.ALL`` (default) or other
                ``MaskPattern`` enum value (VL1..VL128, M3, M4, H, Q, ALLF)
            dtype: Data type that determines mask granularity (default FP32;
                FP16/UINT8 etc.)

        Returns:
            mask_reg: Initialized mask register
        """

    @staticmethod
    @_api_decl
    def update_mask(scalar, dtype: Optional[DType] = None):
        """Update a VF mask register from a scalar value.

        Args:
            scalar: Scalar value whose bits define the new mask pattern

        Kwargs:
            dtype: Data type for mask width selection (default FP32 -> b32)
        """

    @staticmethod
    @_api_decl
    def full(scalar_or_src, mask=None, dtype: Optional[DType] = None,
             mode: Optional[MergeMode] = None,
             pos: Optional[DuplicatePos] = None):
        """Broadcast a scalar or vector-source element into all lanes of a VF register.

        Two modes:

        - **Scalar mode**: ``vf.full(2.5, preg, dtype=pl.DT_FP32)`` --- broadcasts a scalar
          value to all register lanes (vbr/vdup instruction).
        - **Tensor mode**: ``vf.full(src_reg, preg)`` --- broadcasts the lowest or highest
          element of a source register to all lanes of the destination register
          (vdup instruction). Mask is required in Tensor mode.

        Args:
            scalar_or_src: Scalar value (Scalar mode) or source register (Tensor mode)
            mask: Predicate mask register. Required for Tensor mode; optional for Scalar mode

        Kwargs:
            dtype: Required for Scalar mode (cannot infer from scalar). Auto-inferred
                for Tensor mode from the source register.
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
            pos: ``pl.DuplicatePos.LOWEST`` (default) or ``pl.DuplicatePos.HIGHEST`` ---
                selects which element to broadcast in Tensor mode
        """

    @staticmethod
    @_api_decl
    def load_align(src, offset=None, dtype: Optional[DType] = None,
                   dist: Optional[LoadDist] = None,
                   data_copy_mode: Optional[DataCopyMode] = None,
                   block_stride=None, post_update: bool = False):
        """Load aligned data from a UB Tile into a VF register (vlds instruction).

        Assignment form (the destination register is declared implicitly):

            dst = vf.load_align(src, offset)

        Args:
            src: Source UB Tile pointer
            offset: Element offset into the tile (or post-update stride)

        Kwargs:
            dtype: Data type for type-specific variants (e.g. ``pl.DT_UINT32``)
            dist: ``pl.LoadDist`` value selecting load distribution pattern
                (e.g. ``pl.LoadDist.BRC``, ``pl.LoadDist.US``, ``pl.LoadDist.BRC_B32``)
            data_copy_mode: ``pl.DataCopyMode.DATA_BLOCK_COPY`` (AscendC's name for
                the non-contiguous datablock load) selects the vsldb instruction.
                ``DATA_BLOCK_LOAD`` is accepted as an equivalent legacy alias.
            block_stride: Datablock stride in bytes for the datablock-copy mode
            post_update: ``True`` for post-increment addressing
        """

    @staticmethod
    @_api_decl
    def store_align(dst, src, *args, dist: Optional[StoreDist] = None,
                    data_copy_mode: Optional[DataCopyMode] = None,
                    block_stride=None, repeat_stride=None,
                    post_update: bool = False):
        """Store aligned data from a VF register to a UB Tile (vsts instruction).

        Statement form only (no assignment form --- dst is a UB tile, not a register):

            vf.store_align(dst_tile, src, mask)

        Args:
            dst: Destination UB Tile pointer
            src: Source register
            mask: Predicate mask register (omitted when src is a MaskReg)

        Kwargs:
            dist: ``pl.StoreDist.NORM`` (default), ``pl.StoreDist.NORM_B16``,
                  ``pl.StoreDist.FIRST_ELEMENT``, ``pl.StoreDist.PACK``,
                  ``pl.StoreDist.PACK4``, ``pl.StoreDist.INTLV`` / ``INTLV_B32``
                  (interleaved, requires two src registers),
                  ``pl.StoreDist.PACK`` for MaskReg src (psts PK mode)
            post_update: True to auto-advance destination address after store
            data_copy_mode: ``pl.DataCopyMode.DATA_BLOCK_COPY`` for vsstb instruction
            block_stride: DataBlock copy block stride
            repeat_stride: DataBlock copy repeat stride
        """

    @staticmethod
    @_api_decl
    def store_unalign(dst_ptr, src, align_reg, stride=None,
                      post_update: bool = False):
        """Store unaligned data from a VF register to UB (vstur/vstus instruction).

        Args:
            dst_ptr: Destination UB pointer
            src: Source register
            align_reg: Alignment register (from unalign_reg_for_store)
            stride: Optional stride for strided mode (vstus)

        Kwargs:
            post_update: ``True`` to auto-advance destination address after store

        When called with 4 args, the 4th arg is stride for strided mode (vstus).
        """

    @staticmethod
    @_api_decl
    def store_unalign_post(dst_ptr, align_reg, stride=None,
                           post_update: bool = False):
        """Complete an unaligned store sequence (vstar/vstas instruction).

        Args:
            dst_ptr: Destination UB pointer
            align_reg: Alignment register (from unalign_reg_for_store)
            stride: Optional stride for strided mode (vstas)

        Kwargs:
            post_update: ``True`` to auto-advance destination address after store

        When called with 3 args, the 3rd arg is stride for strided mode (vstas).
        """

    @staticmethod
    @_api_decl
    def unalign_reg_for_store():
        """Declare an unaligned register for store operations.

        Must be called before store_unalign/store_unalign_post to allocate the
        alignment state register.

        Returns:
            Unaligned register handle
        """

    @staticmethod
    @_api_decl
    def mem_bar(mode: Optional[MemBarMode] = None):
        """Insert a VF memory barrier (maps to AscendC ``LocalMemBar<src,dst>``).

        Orders memory ops of the ``src`` kind before those of the ``dst`` kind.
        Select the pair via the ``mode`` kwarg (default ``VST_VLD``)::

            vf.mem_bar(mode=pl.MemBarMode.VST_VLD)   # vector store -> vector load
            vf.mem_bar(mode=pl.MemBarMode.VST_VST)   # vector store -> vector store (WAW)
            vf.mem_bar(mode=pl.MemBarMode.VV_ALL)    # all vector <-> all vector

        Supported modes (12, matching AscendC's legal MemType combinations):
            VST_VLD, VLD_VST, VST_VST, VST_LD, VST_ST, VLD_ST,
            ST_VLD, ST_VST, LD_VST, VV_ALL, VS_ALL, SV_ALL
        where V*=vector, S*/*_LD/*_ST=scalar, *_ALL=full barrier of that unit.

        Kwargs:
            mode: ``pl.MemBarMode`` value selecting the src->dst ordering
        """

    @staticmethod
    @_api_decl
    def max(src0, src1, mask, mode: Optional[MergeMode] = None):
        """Element-wise maximum: ``dst[i] = max(src0[i], src1[i])``

        Args:
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def add(src0, src1, mask, mode: Optional[MergeMode] = None):
        """Element-wise addition: ``dst[i] = src0[i] + src1[i]``

        Assignment form (the destination register is declared implicitly):

            dst = vf.add(src0, src1, pred)

        Args:
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def sub(src0, src1, mask, mode: Optional[MergeMode] = None):
        """Element-wise subtraction: ``dst[i] = src0[i] - src1[i]``

        Args:
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def mul(src0, src1, mask, mode: Optional[MergeMode] = None):
        """Element-wise multiplication: ``dst[i] = src0[i] * src1[i]``

        Args:
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def div(src0, src1, mask, mode: Optional[MergeMode] = None):
        """Element-wise division: ``dst[i] = src0[i] / src1[i]``

        Args:
            src0: Numerator register
            src1: Denominator register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def muls(src, scalar, mask, mode: Optional[MergeMode] = None):
        """Multiply all elements by a scalar: ``dst[i] = src[i] * scalar``

        Args:
            src: Source register
            scalar: Scalar multiplier value
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``

        Note:
            Does not support UINT8/INT8 types directly. For 1-byte types,
            the backend automatically expands to vdup+vmul.
        """

    @staticmethod
    @_api_decl
    def mul_add_dst(src0, src1, mask, mode: Optional[MergeMode] = None):
        """Fused multiply-add into destination: ``dst = src0 * src1 + dst``

        The destination register is both read (as addend) and written.
        Maps to hardware ``vmula`` instruction.

        Args:
            src0: First multiplicand register
            src1: Second multiplicand register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def and_(src0, src1, mask, mode: Optional[MergeMode] = None):
        """Element-wise bitwise AND: ``dst[i] = src0[i] & src1[i]``

        Args:
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def or_(src0, src1, mask, mode: Optional[MergeMode] = None):
        """Element-wise bitwise OR: ``dst[i] = src0[i] | src1[i]``

        Args:
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def xor(src0, src1, mask, mode: Optional[MergeMode] = None,
            dtype: Optional[DType] = None):
        """Element-wise bitwise XOR: ``dst[i] = src0[i] ^ src1[i]``

        Args:
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
            dtype: Data type for type-specific variants (e.g. ``pl.DT_UINT16``)
        """

    @staticmethod
    @_api_decl
    def reduce_sum(src, mask, datablock: bool = False,
                   merge_mode: Optional[MergeMode] = None):
        """In-register sum reduction across all lanes (vcadd / vcgadd).

        Reduces all active lanes of the source register into the first element
        of the destination register.

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            datablock: ``True`` to use datablock-granularity reduction (vcgadd)
            merge_mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def reduce_max(src, mask, datablock: bool = False,
                   merge_mode: Optional[MergeMode] = None):
        """In-register max reduction across all lanes (vcmax / vcgmax).

        Reduces all active lanes of the source register into the first element
        of the destination register.

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            datablock: ``True`` to use datablock-granularity reduction (vcgmax)
            merge_mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def reduce_min(src, mask, datablock: bool = False,
                   merge_mode: Optional[MergeMode] = None):
        """In-register min reduction across all lanes (vcmin / vcgmin).

        Reduces all active lanes of the source register into the first element
        of the destination register.

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            datablock: ``True`` to use datablock-granularity reduction (vcgmin)
            merge_mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def ln(src, mask, mode: Optional[MergeMode] = None,
           precision: Optional[str] = None):
        """Natural logarithm: ``dst[i] = ln(src[i])``

        Maps to hardware ``vln`` instruction.

        Args:
            src: Source register (must be positive)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
            precision: ``"HIGH"`` to enable subnormal-input compensation
                       (prescale + vln + offset correction)
        """

    @staticmethod
    @_api_decl
    def exp_sub(src, max_val, mask, layout: Optional[CastLayout] = None):
        """Fused exp-subtract: ``dst[i] = exp(src[i] - max[i])``

        Commonly used in softmax computation. Subtracts max for numerical
        stability before applying exp.

        Args:
            src: Source register
            max_val: Max register (subtracted before exp)
            mask: Predicate mask register

        Kwargs:
            layout: ``pl.CastLayout.ZERO`` (even half, default) or
                ``pl.CastLayout.ONE`` (odd half) --- for half-width results
        """

    @staticmethod
    @_api_decl
    def astype(src, mask, dtype: Optional[DType] = None,
               layout: Optional[CastLayout] = None,
               round_mode: Optional[VFRoundMode] = None,
               saturate: Optional[SaturateMode] = None):
        """Type conversion between register types (vcvt instruction).

        Supports same-width and cross-width conversions: float<->int, int->int
        narrowing/widening, float precision changes.

        Args:
            src: Source register (source type)
            mask: Predicate mask register

        Kwargs:
            layout: ``pl.CastLayout.ZERO`` (default) / ``ONE`` / ``TWO`` / ``THREE``
            round_mode: ``pl.VFRoundMode.CAST_ROUND`` (default) / ``CAST_RINT`` /
                ``CAST_FLOOR`` / ``CAST_CEIL`` / ``CAST_TRUNC`` / ``CAST_RNA`` /
                ``CAST_ODD`` / ``CAST_HYBRID``
            saturate: ``pl.SaturateMode.OFF`` (default) or ``pl.SaturateMode.ON``
        """

    @staticmethod
    @_api_decl
    def de_interleave(src0, src1, dtype: Optional[DType] = None):
        """De-interleave: split even/odd elements into two registers.

        ``dst0 = src[0], src[2], src[4], ...`` (even elements)
        ``dst1 = src[1], src[3], src[5], ...`` (odd elements)

        Args:
            src0: First source register
            src1: Second source register

        Kwargs:
            dtype: When src operands are MaskReg, specifies the interleave bit-width
                (selects ``pdintlv_b8``/``b16``/``b32``). Inferred from src0 if omitted.
        """

    @staticmethod
    @_api_decl
    def select(src_true, src_false, mask, mode: Optional[MergeMode] = None):
        """Conditional select: ``dst[i] = src_true[i] if mask[i] else src_false[i]``

        Args:
            src_true: Register selected when mask bit is 1
            src_false: Register selected when mask bit is 0
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def shift_left(src, shift, mask, mode: Optional[MergeMode] = None):
        """Left shift: ``dst[i] = src[i] << shift``

        The shift amount may be a scalar (all lanes shifted by the same amount,
        emits ``vshls``) or a vector register (per-lane shift amount, emits
        ``vshl``). The form is selected automatically from the argument type::

            dst = vf.shift_left(src, 2, mask)          # scalar: uniform shift
            dst = vf.shift_left(src, shift_reg, mask)   # register: per-lane shift

        This unified op replaces the former separate ``vf.shift_lefts`` (scalar)
        and ``vf.shift_left`` (vector) entry points.

        Args:
            src: Source register
            shift: Shift amount --- scalar integer (uniform) or a vector register
                (per-lane). Negative values are undefined.
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def shift_right(src, shift, mask, mode: Optional[MergeMode] = None,
                    dtype: Optional[DType] = None):
        """Right shift: ``dst[i] = src[i] >> shift``

        The shift amount may be a scalar (all lanes shifted by the same amount,
        emits ``vshrs``) or a vector register (per-lane shift amount, emits
        ``vshr``). The form is selected automatically from the argument type::

            dst = vf.shift_right(src, 24, mask)         # scalar: uniform shift
            dst = vf.shift_right(src, shift_reg, mask)   # register: per-lane shift

        This unified op replaces the former separate ``vf.shift_rights`` (scalar)
        and ``vf.shift_right`` (vector) entry points. Unsigned src does a logical
        shift; signed src does an arithmetic shift.

        Args:
            src: Source register
            shift: Shift amount --- scalar integer (uniform) or a vector register
                (per-lane). Negative values are undefined.
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
            dtype: Data type for type-specific variants (e.g. ``pl.DT_UINT32``)
        """

    @staticmethod
    @_api_decl
    def histograms(src, mask, bin_type: Optional[BinType] = None,
                   hist_type: Optional[HistType] = None):
        """Histogram accumulation (chistv2/dhistv2 instruction).

        Computes histogram on UINT8 data. Supports both cumulative (chistv2)
        and frequency (dhistv2) modes.  The destination register is both read
        and written --- ``chistv2`` accumulates into the existing value, so the
        dst must be pre-initialized (e.g. via ``vf.full(0, ...)``) before the
        first call.  Subsequent ``dst = vf.histograms(...)`` calls reuse the
        same register and continue accumulating.

        Args:
            src: Source register (data to bin)
            mask: Predicate mask register

        Kwargs:
            bin_type: ``pl.BinType.BIN0`` (default) or ``pl.BinType.BIN1`` --- selects bin mapping
            hist_type: ``pl.HistType.ACCUMULATE`` (default, chistv2) or
                ``pl.HistType.FREQUENCY`` (dhistv2)

        Returns:
            Destination histogram register (same register passed as dst,
            with accumulated histogram values)
        """

    @staticmethod
    @_api_decl
    def eq(a, b, mask, cmp_dtype: Optional[DType] = None):
        """Element-wise equality comparison.

        Compares two source elements and writes the result to the
        destination mask register.  If the second argument is a scalar
        literal the vector-scalar compare path is used; otherwise the
        vector-vector compare path is used.

        Args:
            a: First source register
            b: Second source register or scalar value
            mask: Source predicate mask

        Returns:
            MaskReg with comparison result (True where a_i == b_i)
        """

    @staticmethod
    @_api_decl
    def ne(a, b, mask, cmp_dtype: Optional[DType] = None):
        """Element-wise not-equal comparison.

        Args:
            a: First source register
            b: Second source register or scalar value
            mask: Source predicate mask

        Returns:
            MaskReg with comparison result (True where a_i != b_i)
        """

    @staticmethod
    @_api_decl
    def lt(a, b, mask, cmp_dtype: Optional[DType] = None):
        """Element-wise less-than comparison.

        Args:
            a: First source register
            b: Second source register or scalar value
            mask: Source predicate mask

        Returns:
            MaskReg with comparison result (True where a_i < b_i)
        """

    @staticmethod
    @_api_decl
    def gt(a, b, mask, cmp_dtype: Optional[DType] = None):
        """Element-wise greater-than comparison.

        Args:
            a: First source register
            b: Second source register or scalar value
            mask: Source predicate mask

        Returns:
            MaskReg with comparison result (True where a_i > b_i)
        """

    @staticmethod
    @_api_decl
    def le(a, b, mask, cmp_dtype: Optional[DType] = None):
        """Element-wise less-or-equal comparison.

        Args:
            a: First source register
            b: Second source register or scalar value
            mask: Source predicate mask

        Returns:
            MaskReg with comparison result (True where a_i <= b_i)
        """

    @staticmethod
    @_api_decl
    def ge(a, b, mask, cmp_dtype: Optional[DType] = None):
        """Element-wise greater-or-equal comparison.

        Args:
            a: First source register
            b: Second source register or scalar value
            mask: Source predicate mask

        Returns:
            MaskReg with comparison result (True where a_i >= b_i)
        """

    @staticmethod
    @_api_decl
    def squeeze(src, mask, gather_mode: Optional[SqueezeMode] = None,
                dtype: Optional[DType] = None):
        """Squeeze mask to index register (vsqz instruction).

        Converts active mask bits into a packed index sequence in the
        destination register.

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            gather_mode: ``pl.SqueezeMode.STORE_REG`` or ``pl.SqueezeMode.NO_STORE_REG``
            dtype: Data type for the destination register (e.g. ``pl.DT_UINT32``)
        """

    @staticmethod
    @_api_decl
    def arange(start, dtype: Optional[DType] = None,
               index_order: Optional[IndexOrder] = None):
        """Generate an index sequence starting from ``start`` (vci instruction).

        Fills each lane with a sequential value. ``index_order`` selects the
        direction::

            dst = vf.arange(start)                                          # dst[i] = start + i (INC)
            dst = vf.arange(start, index_order=pl.IndexOrder.INCREASE_ORDER)  # same as default
            dst = vf.arange(start, index_order=pl.IndexOrder.DECREASE_ORDER)  # dst[i] = start - i (DEC)

        The per-lane step is fixed at +/-1 (a hardware property of vci); use a
        following ``vf.muls`` to scale the step if needed.

        Args:
            start: Starting value (scalar)

        Kwargs:
            index_order: ``pl.IndexOrder.INCREASE_ORDER`` (default, dst[i]=start+i)
                or ``pl.IndexOrder.DECREASE_ORDER`` (dst[i]=start-i)
        """

    @staticmethod
    @_api_decl
    def gather(src_ub, indices, mask,
               data_copy_mode: Optional[DataCopyMode] = None):
        """Gather elements by index from UB memory.

        Reads elements from non-contiguous UB locations specified by an index register.
        The ``data_copy_mode`` kwarg selects the gather granularity:

            dst = vf.gather(src_ub, indices, mask)                                    # per-element (vgather2)
            dst = vf.gather(src_ub, indices, mask,
                            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_LOAD)           # per 32B datablock (vgatherb)

        The DATA_BLOCK_LOAD form replaces the former standalone ``vf.gatherb`` op.

        Args:
            src_ub: Source UB pointer (base address)
            indices: Index register. NORM mode: element offsets; DATA_BLOCK_LOAD
                mode: byte offsets, must be 32-byte aligned (one index per datablock)
            mask: Predicate mask register

        Kwargs:
            data_copy_mode: ``pl.DataCopyMode.NORM`` (default, per-element via vgather2)
                or ``pl.DataCopyMode.DATA_BLOCK_LOAD`` (per 32B datablock via vgatherb)
        """

    @staticmethod
    @_api_decl
    def clear_spr():
        """Clear special purpose register (AR register).

        Resets the accumulator register used by certain VF instructions.
        """

    @staticmethod
    @_api_decl
    def log(src, mask, mode: Optional[MergeMode] = None,
            precision: Optional[str] = None):
        """Natural logarithm (alias for ln): ``dst[i] = ln(src[i])``

        Identical to ``vf.ln()``. Both map to the hardware ``vln`` instruction.

        Args:
            src: Source register (must be positive)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
            precision: ``"HIGH"`` for subnormal compensation
        """

    @staticmethod
    @_api_decl
    def min(src0, src1, mask, mode: Optional[MergeMode] = None):
        """Element-wise minimum: ``dst[i] = min(src0[i], src1[i])``

        Args:
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def exp(src, mask, mode: Optional[MergeMode] = None,
            precision: Optional[str] = None):
        """Exponential function: ``dst[i] = exp(src[i])``

        Maps to hardware ``vexp`` instruction.

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
            precision: ``"HIGH"`` to detect subnormal results and apply
                       exp(x+ln2)*0.5 correction
        """

    @staticmethod
    @_api_decl
    def abs(src, mask, mode: Optional[MergeMode] = None):
        """Element-wise absolute value: ``dst[i] = |src[i]|``

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def not_(src, mask, mode: Optional[MergeMode] = None):
        """Element-wise bitwise NOT: ``dst[i] = ~src[i]``

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def sqrt(src, mask, mode: Optional[MergeMode] = None,
             precision: Optional[str] = None):
        """Square root: ``dst[i] = sqrt(src[i])``

        Maps to hardware ``vsqrt`` instruction.

        Args:
            src: Source register (must be non-negative)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
            precision: ``"HIGH"`` for FP16 subnormal precision path
                       (prescale x2^10, vsqrt, then x2^-5)
        """

    @staticmethod
    @_api_decl
    def relu(src, mask, mode: Optional[MergeMode] = None):
        """ReLU activation: ``dst[i] = max(0, src[i])``

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def neg(src, mask, mode: Optional[MergeMode] = None):
        """Negation: ``dst[i] = -src[i]``

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def adds(src, scalar, mask, mode: Optional[MergeMode] = None):
        """Add scalar to each element: ``dst[i] = src[i] + scalar``

        Args:
            src: Source register
            scalar: Scalar addend value
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def subs(src, scalar, mask, mode: Optional[MergeMode] = None):
        """Subtract scalar from each element: ``dst[i] = src[i] - scalar``

        Implemented as ``vadds(dst, src, -scalar, ...)``.

        Args:
            src: Source register
            scalar: Scalar subtrahend value
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def mins(src, scalar, mask, mode: Optional[MergeMode] = None):
        """Element-wise minimum with scalar: ``dst[i] = min(src[i], scalar)``

        Args:
            src: Source register
            scalar: Scalar value to compare against
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def maxs(src, scalar, mask, mode: Optional[MergeMode] = None):
        """Element-wise maximum with scalar: ``dst[i] = max(src[i], scalar)``

        Args:
            src: Source register
            scalar: Scalar value to compare against
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def leaky_relu(src, alpha, mask, mode: Optional[MergeMode] = None):
        """Leaky ReLU: ``dst[i] = src[i] if src[i] >= 0 else alpha * src[i]``

        Args:
            src: Source register
            alpha: Negative slope scalar (e.g. 0.1)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def interleave(src0, src1, dtype: Optional[DType] = None):
        """Interleave two registers: combine even/odd elements.

        ``dst0 = src0[0], src1[0], src0[1], src1[1], ...``
        ``dst1 = src0[N/2], src1[N/2], ...``

        Tuple assignment form (the destination registers are declared implicitly):

            dst0, dst1 = vf.interleave(src0, src1)

        Args:
            src0: First source register
            src1: Second source register

        Kwargs:
            dtype: When src operands are MaskReg, specifies the interleave bit-width
                (selects ``pintlv_b8``/``b16``/``b32``). Inferred from src0 if omitted.
        """

    @staticmethod
    @_api_decl
    def pair_reduce_sum(src, mask, mode: Optional[MergeMode] = None):
        """Pairwise reduction sum: ``dst[i] = src[2i] + src[2i+1]``

        Adds adjacent pairs of elements. Maps to hardware ``vcpadd`` instruction.

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def abs_sub(src0, src1, mask, mode: Optional[MergeMode] = None):
        """Absolute difference: ``dst[i] = |src0[i] - src1[i]|``

        Maps to hardware ``vabsdif`` instruction.

        Args:
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def axpy(src, scalar, mask, mode: Optional[MergeMode] = None):
        """Fused AXPY: ``dst = src * scalar + dst``

        The destination register is both read (as addend) and written.
        Maps to hardware ``vaxpy`` instruction.

        Args:
            src: Source register
            scalar: Scalar multiplier
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def copy(src, mask, mode: Optional[MergeMode] = None):
        """Register copy: ``dst[i] = src[i]`` where mask is active.

        Maps to hardware ``vmov`` instruction with MODE_MERGING.

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def mul_dst_add(src0, src1, mask, mode: Optional[MergeMode] = None):
        """Multiply-dst-add: ``dst = dst * src0 + src1`` (vmadd)

        The destination register is both read (as multiplicand) and written.
        Maps to hardware ``vmadd`` instruction (AscendC MulDstAdd).

        Args:
            src0: First multiplicand register
            src1: Addend register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def pack(src, dtype: Optional[DType] = None,
             part: Optional[PackPart] = None):
        """Pack/narrow data type (e.g. u32->u16, u16->u8).

        Selects upper or lower half of each wider element.

        Args:
            src: Source register (wider type)

        Kwargs:
            dtype: Destination data type (e.g. ``pl.DT_UINT16``)
            part: ``pl.PackPart.LOWER`` (default) or ``pl.PackPart.UPPER``
        """

    @staticmethod
    @_api_decl
    def unpack(src, dtype: Optional[DType] = None,
               part: Optional[PackPart] = None):
        """Unpack/widen data type (e.g. u8->u16, u16->u32).

        Zero-extends or sign-extends narrower elements into wider type.

        Args:
            src: Source register (narrower type)

        Kwargs:
            dtype: Destination data type (e.g. ``pl.DT_UINT32``)
            part: ``pl.PackPart.LOWER`` (default) or ``pl.PackPart.UPPER`` --- which half of src to unpack
        """

    @staticmethod
    @_api_decl
    def prelu(src, slope, mask, mode: Optional[MergeMode] = None):
        """Parametric ReLU with per-element slope register.

        ``dst[i] = src[i] if src[i] >= 0 else src[i] * slope[i]``

        Unlike leaky_relu which uses a scalar alpha, prelu uses a per-element
        slope vector.

        Args:
            src: Source register
            slope: Slope register (per-element negative slope values)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def mull(src0, src1, mask, mode: Optional[MergeMode] = None):
        """Long multiply: 32x32->64, output split into lo/hi register pair.

        Multiplies two 32-bit registers and produces 64-bit result split
        across two 32-bit destination registers.

        Tuple assignment form (the destination registers are declared implicitly):

            dst_lo, dst_hi = vf.mull(src0, src1, pred)

        Args:
            src0: First source register (32-bit)
            src1: Second source register (32-bit)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def addc(src0, src1, carry_in, mask, mode: Optional[MergeMode] = None):
        """Add with carry (vaddcs): ``carry_out, dst = src0 + src1 + carry_in``

        Used for multi-word (e.g. 64-bit) arithmetic on 32-bit registers.
        Produces two outputs via tuple unpacking --- the carry-out flag register
        (declared as a MaskReg) and the sum register (RegTensor)::

            carry_out, dst = vf.addc(src0, src1, carry_in, mask)

        Args:
            src0: First source register
            src1: Second source register
            carry_in: Input carry flag register (MaskReg)
            mask: Predicate mask register

        Returns:
            (carry_out, dst): carry-out flag register and the sum register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def subc(src0, src1, borrow_in, mask, mode: Optional[MergeMode] = None):
        """Subtract with borrow (vsubcs): ``borrow_out, dst = src0 - src1 - borrow_in``

        Used for multi-word (e.g. 64-bit) arithmetic on 32-bit registers.
        Produces two outputs via tuple unpacking --- the borrow-out flag register
        (declared as a MaskReg) and the difference register (RegTensor)::

            borrow_out, dst = vf.subc(src0, src1, borrow_in, mask)

        Args:
            src0: First source register
            src1: Second source register
            borrow_in: Input borrow flag register (MaskReg)
            mask: Predicate mask register

        Returns:
            (borrow_out, dst): borrow-out flag register and the difference register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def load_unalign_init():
        """Declare an unaligned register for load operations.

        Must be called before load_unalign_pre/load_unalign to allocate
        the alignment state register.

        Returns:
            Unaligned register handle
        """

    @staticmethod
    @_api_decl
    def load_unalign_pre(ureg, src_ptr):
        """Setup unaligned load (vldas instruction).

        Initializes the alignment state for subsequent unaligned loads.

        Args:
            ureg: UnalignRegForLoad register (from load_unalign_init)
            src_ptr: Source UB pointer
        """

    @staticmethod
    @_api_decl
    def load_unalign(ureg, src_ptr, stride=None):
        """Unaligned load from UB to register (vldus instruction).

        Loads data from an unaligned UB address. Supports optional stride
        for POST_UPDATE mode.

        Args:
            ureg: UnalignRegForLoad register
            src_ptr: Source UB pointer
            stride: Optional post-update stride in bytes
        """

    @staticmethod
    @_api_decl
    def scatter(base_ptr, src, index, mask):
        """Scatter store by index (vscatter instruction).

        Writes register elements to non-contiguous UB locations specified
        by an index register.

        Args:
            base_ptr: Base UB pointer
            src: Source register (data to scatter)
            index: Index register (destination offsets)
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def unsqueeze(mask, dtype: Optional[DType] = None):
        """Unsqueeze mask bits into a register (vusqz instruction).

        Expands each mask bit into the corresponding register lane
        (1 for active, 0 for inactive).

        Args:
            mask: Mask register to unsqueeze

        Kwargs:
            dtype: Data type for the destination register (e.g. ``pl.DT_UINT32``)
        """

    @staticmethod
    @_api_decl
    def gatherb(*args, **kwargs):
        """Deprecated --- merged into :func:`vf.gather`.

        Use ``vf.gather(src_ub, indices, mask,
        data_copy_mode=pl.DataCopyMode.DATA_BLOCK_LOAD)`` for the former
        datablock-granularity gather. This name is no longer accepted by the
        parser and is retained only as a migration pointer.
        """

    @staticmethod
    @_api_decl
    def get_mask_spr(width: MaskWidth = MaskWidth.B32):
        """Get mask from special purpose register (movp_b32/movp_b16).

        Reads the {MASK1, MASK0} SPR set by SetVectorMask and converts it
        to a MaskReg. This is the pypto equivalent of AscendC ``MoveMask<T>``.

        - ``width=pl.MaskWidth.B32``: reads 64-bit MASK0, expands each bit to 4 bits (movp_b32)
        - ``width=pl.MaskWidth.B16``: reads full 128-bit {MASK1, MASK0}, expands each bit to 2 bits (movp_b16)

        Kwargs:
            width: ``pl.MaskWidth.B32`` (default) or ``pl.MaskWidth.B16`` --- selects mask width

        Returns:
            mask_reg with current SPR value
        """

    @staticmethod
    @_api_decl
    def mla(dst, src0, src1, src2, mask):
        """Multiply-add (3 source): ``dst[i] = src0[i] * src1[i] + src2[i]``

        Unlike mul_dst_add which accumulates into dst, mla takes a separate addend register.

        Args:
            dst: Destination register
            src0: First multiplicand register
            src1: Second multiplicand register
            src2: Addend register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def avg(dst, src0, src1, mask):
        """Element-wise average: ``dst[i] = (src0[i] + src1[i]) / 2``

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def add3(dst, src0, src1, src2, mask):
        """Three-operand add: ``dst[i] = src0[i] + src1[i] + src2[i]``

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            src2: Third source register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def gather2(dst, src_ub, index, mask):
        """Gather with two-element stride (vgather2 instruction).

        Gathers pairs of elements from UB memory.

        Args:
            dst: Destination register
            src_ub: Source UB pointer
            index: Index register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def select_r(src_true, src_false, mask, mode: Optional[MergeMode] = None):
        """Select by register (reversed select): ``dst[i] = src_false[i] if mask[i] else src_true[i]``

        Reversed polarity compared to ``select``.

        Args:
            src_true: Register selected when mask bit is 0
            src_false: Register selected when mask bit is 1
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def log2(src, mask, mode: Optional[MergeMode] = None):
        """Base-2 logarithm: ``dst[i] = log2(src[i])``

        Synthesized as ``vln(src) * (1/ln(2))``.

        Args:
            src: Source register (must be positive)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def log10(src, mask, mode: Optional[MergeMode] = None):
        """Base-10 logarithm: ``dst[i] = log10(src[i])``

        Synthesized as ``vln(src) * (1/ln(10))``.

        Args:
            src: Source register (must be positive)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def muls_cast(src, scalar, mask, dtype: Optional[DType] = None,
                  layout: Optional[CastLayout] = None):
        """Multiply by scalar then cast: ``dst(fp16) = cast(src(fp32) * scalar)``

        Fused operation combining vmuls and vcvt.

        Args:
            src: Source register (fp32)
            scalar: Scalar multiplier
            mask: Predicate mask register

        Kwargs:
            dtype: Destination data type (e.g. ``pl.DT_FP16``)
            layout: ``pl.CastLayout.ZERO`` (even half, default) or
                ``pl.CastLayout.ONE`` (odd half) for the half-width result
        """

    @staticmethod
    @_api_decl
    def load(src_ptr, stride=None, post_update: bool = False,
             repeat_stride=None, count=None):
        """Unified load (vldas+vldus, matches AscendC Load interface).

        Simple aligned load from UB to register. Supports optional post-update.

        Args:
            src_ptr: Source UB pointer
            stride: Post-update stride (optional, triggers POST_UPDATE mode)

        Kwargs:
            post_update: Enable post-update addressing
            repeat_stride: Stride for repeated loads
            count: Element count
        """

    @staticmethod
    @_api_decl
    def store(dst_ptr, src, count=None, post_update: bool = False,
              repeat_stride=None):
        """Unified store (vstus+vstas, matches AscendC Store interface).

        Simple store from register to UB. Supports optional post-update.

        Args:
            dst_ptr: Destination UB pointer
            src: Source register
            count: Element count (optional, defaults to 256/elem_bytes)

        Kwargs:
            post_update: Enable post-update addressing
            repeat_stride: Stride for repeated stores
        """

    @staticmethod
    @_api_decl
    def truncate(src, mask, mode: Optional[MergeMode] = None):
        """Truncate to integer (round toward zero): ``dst[i] = trunc(src[i])``

        Maps to hardware ``vtrc`` with ROUND_Z mode.

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def mask_gen_with_reg_tensor(src, offset: Optional[int] = None):
        """Generate mask from a register tensor bit at a given offset (movvp instruction).

        Converts a bit in a register element into a mask predicate.

        Args:
            src: Source register (uint16 or uint32)

        Kwargs:
            offset: Bit offset within the element

        Returns:
            mask_reg with generated mask
        """

    @staticmethod
    @_api_decl
    def create_addr_reg(index0, stride0, index1=None, stride1=None,
                        index2=None, stride2=None, index3=None, stride3=None,
                        dtype: Optional[DType] = None):
        """Create an address offset register for aligned load/store (CreateAddrReg).

        Computes ``offset = index0 * stride0 + index1 * stride1 + ...`` and
        returns an ``AddrReg`` that can be passed to ``vf.load_align`` /
        ``vf.store_align`` as the offset parameter. Supports 1-4 loop axes
        (index/stride pairs).

        Usage::

            aReg = vf.create_addr_reg(i, 64, dtype=pl.DT_FP32)
            reg = vf.load_align(src_tile, aReg)
            vf.store_align(dst_tile, reg, preg, aReg)

        Args:
            index0: Loop axis 0 index (loop variable)
            stride0: Loop axis 0 stride in elements
            index1: Optional loop axis 1 index
            stride1: Optional loop axis 1 stride in elements
            index2: Optional loop axis 2 index
            stride2: Optional loop axis 2 stride in elements
            index3: Optional loop axis 3 index
            stride3: Optional loop axis 3 stride in elements

        Kwargs:
            dtype: Data type for template instantiation (default ``pl.DT_FP32``).
                Determines the element width: b8/b16/b32/b64.

        Returns:
            AddrReg handle for use as offset in load_align/store_align
        """

    @staticmethod
    @_api_decl
    def move(src, mask=None, mode: Optional[MergeMode] = None):
        """Move/copy register elements (vmov for RegTensor, pmov for MaskReg).

        For RegTensor: copies valid elements from src to dst; masked-out
        positions retain dst's original value (MODE_MERGING).

        For MaskReg: copies src bits to dst; with mask, only masked bits
        are copied.

        Usage::

            # RegTensor with mask
            dst = vf.move(src_reg, mask)
            # RegTensor without mask
            dst = vf.move(src_reg)
            # MaskReg with mask
            dst = vf.move(src_mask, mask)
            # MaskReg without mask
            dst = vf.move(src_mask)

        Args:
            src: Source register (RegTensor or MaskReg)
            mask: Optional mask register

        Kwargs:
            mode: ``pl.MergeMode.MERGING`` (default, only supported mode)
        """
