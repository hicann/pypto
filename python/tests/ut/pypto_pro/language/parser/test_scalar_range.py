#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Range validation of numeric literals on their way into the IR.

Two bounds are enforced, and they are deliberately different:

  * The **storage band** ``[INT64_MIN, UINT64_MAX]`` applies wherever an integer materialises into
    ``ir.ConstInt``. Arbitrary precision stays legal before that point -- the parser evaluates
    intermediates in Python -- and floats have no such band at all, because the front end and
    ``ir.ConstFloat`` are both doubles.
  * The **API fit** check rejects a literal scalar operand the operator's element dtype could not
    represent. It reads the dtype from the operand, never from the literal, since a bare literal
    carries the uncommitted INDEX / FP32 placeholder.
"""
from pypto_pro import ir
from pypto_pro.ir._limits import INT64_MAX, INT64_MIN, UINT64_MAX
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.language.parser.diagnostics import FinalRejectionError, ParserSyntaxError, ParserTypeError
import pytest

_N, _M = 1, 64
_TILE_SIZE = _N * _M * 4

def _arange_body(dtype, scalar):
    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        reg_out = vf.arange(scalar, dtype=dtype)
        vf.store_align(t_out, reg_out, preg)

    return vf_body


def _adds_body(dtype, scalar):
    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        reg_a = vf.load_align(in_a, 0)
        reg_out = vf.adds(reg_a, scalar, preg)
        vf.store_align(t_out, reg_out, preg)

    return vf_body


def _subs_body(dtype, scalar):
    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        reg_a = vf.load_align(in_a, 0)
        reg_out = vf.subs(reg_a, scalar, preg)
        vf.store_align(t_out, reg_out, preg)

    return vf_body


def _muls_body(dtype, scalar):
    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        reg_a = vf.load_align(in_a, 0)
        reg_out = vf.muls(reg_a, scalar, preg)
        vf.store_align(t_out, reg_out, preg)

    return vf_body


def _mins_body(dtype, scalar):
    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        reg_a = vf.load_align(in_a, 0)
        reg_out = vf.mins(reg_a, scalar, preg)
        vf.store_align(t_out, reg_out, preg)

    return vf_body


def _maxs_body(dtype, scalar):
    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        reg_a = vf.load_align(in_a, 0)
        reg_out = vf.maxs(reg_a, scalar, preg)
        vf.store_align(t_out, reg_out, preg)

    return vf_body


def _shift_left_body(dtype, scalar):
    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        reg_a = vf.load_align(in_a, 0)
        reg_out = vf.shift_left(reg_a, scalar, preg)
        vf.store_align(t_out, reg_out, preg)

    return vf_body


def _shift_right_body(dtype, scalar):
    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        reg_a = vf.load_align(in_a, 0)
        reg_out = vf.shift_right(reg_a, scalar, preg)
        vf.store_align(t_out, reg_out, preg)

    return vf_body


# The parser resolves ``vf.<op>`` from the AST, so each op needs its own literal call site.
_VF_BODIES = {
    "adds": _adds_body,
    "subs": _subs_body,
    "muls": _muls_body,
    "mins": _mins_body,
    "maxs": _maxs_body,
    "shift_left": _shift_left_body,
    "shift_right": _shift_right_body,
    "arange": _arange_body,
}


def _parse_with_body(dtype, vf_body):
    """Parse a kernel whose vector section is just ``vf_body``."""

    @pl.jit()
    def kernel(a: pl.Tensor[[_N, _M], dtype], out: pl.Tensor[[_N, _M], dtype]):
        tf = pl.TileType(shape=[_N, _M], dtype=dtype, target_memory=pl.MemorySpace.Vec)
        in_a = pl.make_tile(tf, addr=0, size=_TILE_SIZE)
        t_out = pl.make_tile(tf, addr=_TILE_SIZE, size=_TILE_SIZE)
        with pl.section_vector():
            pl.load(in_a, a, [0, 0])
            vf_body(in_a, t_out)
            pl.store(out, t_out, [0, 0])

    return kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def _parse_vf_scalar_kernel(dtype, op_name, scalar):
    """Build and parse a kernel whose only VF op applies ``scalar`` to a register of ``dtype``."""
    return _parse_with_body(dtype, _VF_BODIES[op_name](dtype, scalar))


# ---------------------------------------------------------------------------
# API fit: a literal scalar operand must fit the operand's element dtype
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
@pytest.mark.parametrize("op_name", ["adds", "subs", "muls", "mins", "maxs"])
@pytest.mark.parametrize(
    ("dtype", "scalar", "expected"),
    [
        (pl.DT_INT8, 300, r"representable in int8, i\.e\. in \[-128, 127\], got 300"),
        (pl.DT_INT8, -129, r"representable in int8, i\.e\. in \[-128, 127\], got -129"),
        (pl.DT_UINT8, -1, r"representable in uint8, i\.e\. in \[0, 255\], got -1"),
        (pl.DT_UINT8, 256, r"representable in uint8, i\.e\. in \[0, 255\], got 256"),
        (pl.DT_FP16, 70000.0, r"representable in fp16, i\.e\. in \[-65504\.0, 65504\.0\], got 70000\.0"),
        (pl.DT_FP32, 1e300, r"representable in fp32"),
    ],
)
def test_out_of_range_scalar_operand_is_rejected(op_name, dtype, scalar, expected):
    with pytest.raises(FinalRejectionError, match=expected):
        _parse_vf_scalar_kernel(dtype, op_name, scalar)


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    ("dtype", "scalar"),
    [
        (pl.DT_INT8, 127),
        (pl.DT_INT8, -128),
        (pl.DT_UINT8, 0),
        (pl.DT_UINT8, 255),
        (pl.DT_INT32, 2**31 - 1),
        (pl.DT_INT32, -(2**31)),
        (pl.DT_FP16, 65504.0),
        (pl.DT_FP16, -65504.0),
        (pl.DT_FP32, 1e-30),
        (pl.DT_FP32, 1.5),
    ],
)
def test_boundary_scalar_operand_still_parses(dtype, scalar):
    _parse_vf_scalar_kernel(dtype, "adds", scalar)


@pytest.mark.soc("950")
def test_rejection_names_the_api():
    with pytest.raises(FinalRejectionError, match=r"vf\.muls: scalar operand"):
        _parse_vf_scalar_kernel(pl.DT_INT8, "muls", 300)


@pytest.mark.soc("950")
def test_negative_into_unsigned_explains_the_signedness():
    with pytest.raises(FinalRejectionError, match=r"hint: -1 is out of range of current dtype uint8"):
        _parse_vf_scalar_kernel(pl.DT_UINT8, "adds", -1)


@pytest.mark.soc("950")
def test_float_inf_to_int32():
    with pytest.raises(FinalRejectionError,
            match=r"ErrCode: F00001, vf.adds: scalar operand must be representable in int8"):
        _parse_vf_scalar_kernel(pl.DT_INT8, "adds", float("inf"))


@pytest.mark.soc("950")
def test_arange_check_ut_1():
    with pytest.raises(FinalRejectionError, match=r"scalar operand must be representable in int32"):
        _parse_vf_scalar_kernel(pl.DT_INT32, "arange", float("inf"))


@pytest.mark.soc("950")
def test_arange_check_ut_2():
    with pytest.raises(FinalRejectionError, match=r"hint: 100000000000000000 is out of range of current dtype int32"):
        _parse_vf_scalar_kernel(pl.DT_INT32, "arange", 100000000000000000)


@pytest.mark.soc("950")
def test_rejection_is_final_and_not_retried_as_python():
    """FinalRejectionError subclasses ParserTypeError, which parse_expression would otherwise retry."""
    with pytest.raises(ParserTypeError):
        _parse_vf_scalar_kernel(pl.DT_INT8, "adds", 300)


# ---------------------------------------------------------------------------
# Exemptions: a shift wider than the type is documented behaviour, not an error
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
@pytest.mark.parametrize("op_name", ["shift_left", "shift_right"])
@pytest.mark.parametrize("shift", [300, 64])
def test_shift_amount_outside_the_dtype_range_still_parses(op_name, shift):
    _parse_vf_scalar_kernel(pl.DT_INT8, op_name, shift)


# ---------------------------------------------------------------------------
# Storage band: the range ir.ConstInt can carry
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
@pytest.mark.parametrize("value", [UINT64_MAX + 1, INT64_MIN - 1, 2**200])
def test_integer_outside_the_storage_band_is_rejected(value):
    with pytest.raises(FinalRejectionError, match=r"must be in \[-9223372036854775808, 18446744073709551615\]"):
        _parse_vf_scalar_kernel(pl.DT_FP32, "adds", value)


@pytest.mark.parametrize(
    ("value", "dtype", "stored"),
    [
        (INT64_MAX, ir.DataType.INDEX, INT64_MAX),
        (INT64_MIN, ir.DataType.INDEX, INT64_MIN),
        (INT64_MAX + 1, ir.DataType.UINT64, INT64_MIN),
        (UINT64_MAX, ir.DataType.UINT64, -1),
    ],
)
def test_uint64_band_settles_on_uint64_and_folds(value, dtype, stored):
    """A value above INT64_MAX commits the uncommitted INDEX placeholder to UINT64 and folds."""
    from pypto_pro.language.parser.diagnostics import make_const_int

    const = make_const_int(value, ir.DataType.INDEX, span=ir.Span.unknown())

    assert const.type.dtype == dtype
    assert const.value == stored


def test_named_dtype_is_taken_at_its_word():
    """An explicitly named dtype is not promoted: the value has to fit it."""
    from pypto_pro.language.parser.diagnostics import make_const_int

    with pytest.raises(FinalRejectionError, match=r"representable in int64"):
        make_const_int(UINT64_MAX, ir.DataType.INT64, span=ir.Span.unknown())


# ---------------------------------------------------------------------------
# bool is an int in Python, and must not slip through as one
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [True, False])
def test_bool_fits_bool_dtype(value):
    from pypto_pro.ir._limits import fits

    assert fits(value, ir.DataType.BOOL)[0]


@pytest.mark.parametrize("value", [True, False])
def test_bool_is_in_range_for_a_narrow_int_dtype(value):
    from pypto_pro.ir._limits import fits

    assert fits(value, ir.DataType.INT8)[0]


# ---------------------------------------------------------------------------
# pl.const names its own dtype, so it is checked against it
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_pl_const_rejects_a_value_its_dtype_cannot_hold():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[_N, _M], pl.DT_FP16]):
        _unused = pl.const(300, pl.DT_INT8)

    with pytest.raises(FinalRejectionError, match=r"pl\.const: value must be representable in int8"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_pl_const_accepts_a_boundary_value():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[_N, _M], pl.DT_FP16]):
        _unused = pl.const(127, pl.DT_INT8)

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# Non-finite floats: a reduction seed for a float dtype, undefined for an integer one
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_infinity_is_accepted_for_a_float_dtype(value):
    """-inf is a legitimate max-reduction seed; rejecting it would break real kernels."""
    _parse_vf_scalar_kernel(pl.DT_FP32, "adds", value)


@pytest.mark.soc("950")
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_is_rejected_for_an_integer_dtype(value):
    """The backend lowers a float scalar via static_cast<int64_t>, undefined for inf / nan."""
    with pytest.raises(FinalRejectionError, match=r"representable in int32"):
        _parse_vf_scalar_kernel(pl.DT_INT32, "adds", value)


# ---------------------------------------------------------------------------
# The source-literal funnel (parse_constant), as opposed to a closure variable
# ---------------------------------------------------------------------------
def _source_literal_body(dtype):
    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        reg_a = vf.load_align(in_a, 0)
        reg_out = vf.adds(reg_a, 300, preg)  # written in the kernel source, not captured
        vf.store_align(t_out, reg_out, preg)

    return vf_body


@pytest.mark.soc("950")
def test_literal_written_in_the_kernel_source_is_checked():
    """Every other test here captures the scalar; this one exercises parse_constant itself."""
    with pytest.raises(FinalRejectionError, match=r"vf\.adds: scalar operand must be representable in int8"):
        _parse_with_body(pl.DT_INT8, _source_literal_body(pl.DT_INT8))


@pytest.mark.soc("950")
def test_literal_written_in_the_kernel_source_is_accepted_when_it_fits():
    _parse_with_body(pl.DT_INT32, _source_literal_body(pl.DT_INT32))


# ---------------------------------------------------------------------------
# The block tile-scalar chokepoint (pl.add / pl.sub / ... with a scalar rhs)
# ---------------------------------------------------------------------------
def _parse_tile_scalar_kernel(dtype, scalar):
    @pl.jit(auto_mutex=False)
    def kernel(a: pl.Tensor[[8, 64], dtype], out: pl.Tensor[[8, 64], dtype]):
        tf = pl.TileType(shape=[8, 64], dtype=dtype, target_memory=pl.MemorySpace.Vec)
        t_in = pl.make_tile(tf, addr=0, size=8 * 64 * 4)
        t_out = pl.make_tile(tf, addr=8 * 64 * 4, size=8 * 64 * 4)
        with pl.section_vector():
            pl.load(t_in, a, [0, 0])
            pl.add(t_out, t_in, scalar)
            pl.store(out, t_out, [0, 0])

    return kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_tile_scalar_op_checks_the_scalar_against_the_out_dtype():
    with pytest.raises(FinalRejectionError, match=r"pl\.add: scalar operand must be representable in int8"):
        _parse_tile_scalar_kernel(pl.DT_INT8, 300)


@pytest.mark.soc("950")
def test_tile_scalar_op_accepts_a_scalar_that_fits():
    _parse_tile_scalar_kernel(pl.DT_INT8, 127)


# ---------------------------------------------------------------------------
# The builder funnel (_normalize_expr) and the folded-constant funnel
# ---------------------------------------------------------------------------
def test_normalize_expr_rejects_an_integer_outside_the_storage_band():
    from pypto_pro.ir._utils import _normalize_expr

    with pytest.raises(FinalRejectionError, match=r"must be in \[-9223372036854775808, 18446744073709551615\]"):
        _normalize_expr(UINT64_MAX + 1)


def test_normalize_expr_settles_a_uint64_value_on_uint64():
    from pypto_pro.ir._utils import _normalize_expr

    const = _normalize_expr(UINT64_MAX)

    assert const.type.dtype == ir.DataType.UINT64
    assert const.value == -1


@pytest.mark.parametrize(
    ("value", "dtype", "expected"),
    [
        (70000.0, ir.DataType.FP16, r"float constant must be representable in fp16"),
        (300, ir.DataType.INT8, r"integer constant must be representable in int8"),
    ],
)
def test_make_scalar_constant_checks_both_branches(value, dtype, expected):
    """Folding must not silently turn an out-of-range result into inf or a wrapped integer."""
    from pypto_pro.language.parser._expression_parser import ExpressionParserMixin

    with pytest.raises(FinalRejectionError, match=expected):
        ExpressionParserMixin._make_scalar_constant(value, dtype, ir.Span.unknown())


@pytest.mark.parametrize(("value", "dtype"), [(65504.0, ir.DataType.FP16), (127, ir.DataType.INT8)])
def test_make_scalar_constant_accepts_a_boundary_value(value, dtype):
    from pypto_pro.language.parser._expression_parser import ExpressionParserMixin

    assert ExpressionParserMixin._make_scalar_constant(value, dtype, ir.Span.unknown()) is not None


# ---------------------------------------------------------------------------
# Which dtype the scalar is checked against: the src operand, not the dtype kwarg
# ---------------------------------------------------------------------------
def _muls_cast_body(scalar):
    """vf.muls_cast is type-changing: dtype= names the DST, so the scalar belongs to the src domain."""

    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        reg_src = vf.load_align(in_a, 0)
        reg_f16 = vf.muls_cast(reg_src, scalar, preg, dtype=pl.DT_FP16)
        reg_out = vf.astype(reg_f16, preg, dtype=pl.DT_FP32)
        vf.store_align(t_out, reg_out, preg)

    return vf_body


@pytest.mark.soc("950")
def test_type_changing_op_checks_the_scalar_against_the_source_dtype():
    """70000.0 does not fit the fp16 dst but does fit the fp32 src, which is the operand's domain."""
    _parse_with_body(pl.DT_FP32, _muls_cast_body(70000.0))


@pytest.mark.soc("950")
def test_type_changing_op_still_rejects_a_value_the_source_dtype_cannot_hold():
    with pytest.raises(FinalRejectionError, match=r"vf\.muls_cast: scalar operand must be representable in fp32"):
        _parse_with_body(pl.DT_FP32, _muls_cast_body(1e300))


def _full_body(dtype, scalar):
    """vf.full has no runtime operand, so the dtype kwarg is the only source of the operand dtype."""

    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        reg_out = vf.full(scalar, preg, dtype=dtype)
        vf.store_align(t_out, reg_out, preg)

    return vf_body


@pytest.mark.soc("950")
def test_op_without_a_runtime_operand_falls_back_to_the_dtype_kwarg():
    with pytest.raises(FinalRejectionError, match=r"vf\.full: scalar operand must be representable in fp16"):
        _parse_with_body(pl.DT_FP16, _full_body(pl.DT_FP16, 70000.0))


@pytest.mark.soc("950")
def test_op_without_a_runtime_operand_accepts_a_value_that_fits():
    _parse_with_body(pl.DT_FP16, _full_body(pl.DT_FP16, 65504.0))


# ---------------------------------------------------------------------------
# Consolidated hardware/limit ranges (the hand-rolled checks migrated onto check_in_range)
# ---------------------------------------------------------------------------
@pl.simt.function(max_threads=4096)
def _too_many_threads(dst: pl.Tensor[[1, 256], pl.DT_FP32], src: pl.Tensor[[1, 256], pl.DT_FP32]):
    tid = pl.simt.linear_thread_idx()
    dst[0, tid] = src[0, tid]


@pl.jit
def _launch_too_many_threads(x: pl.Tensor[[1, 256], pl.DT_FP32], out: pl.Tensor[[1, 256], pl.DT_FP32]):
    with pl.section_vector():
        pl.simt.launch(_too_many_threads, threads=256, args=(out, x))


@pytest.mark.soc("950")
def test_simt_max_threads_range_is_enforced():
    """max_threads moved onto check_in_range; nothing else pins the bound.

    The check raises a plain ValueError (it is a builder-level guard, not an expression position), so
    the parser wraps it: the message survives, the type does not.
    """
    with pytest.raises(ParserSyntaxError, match=r"max_threads must be in \[1, 2048\], got 4096"):
        _launch_too_many_threads.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# pl.expands: the splat value must fit the out tile's dtype
# ---------------------------------------------------------------------------
def _parse_expands_kernel(dtype, scalar, elem_bytes):
    size = 64 * 32 * elem_bytes

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[64, 32], dtype]):
        tf = pl.TileType(shape=[64, 32], dtype=dtype, target_memory=pl.MemorySpace.Vec)
        t_out = pl.make_tile(tf, addr=0, size=size)
        with pl.section_vector():
            pl.expands(t_out, scalar)
            pl.store(out, t_out, [0, 0])

    return kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    ("dtype", "elem_bytes", "scalar", "expected"),
    [
        (pl.DT_INT8, 1, 300, r"representable in int8, i\.e\. in \[-128, 127\], got 300"),
        (pl.DT_INT8, 1, -129, r"representable in int8"),
        (pl.DT_UINT8, 1, 256, r"representable in uint8"),
        # Inside the IR storage band -- carried as a uint64 constant -- but outside the tile dtype,
        # so only the per-dtype check can reject these two.
        (pl.DT_INT64, 8, 2**63, r"representable in int64, i\.e\. in \[-9223372036854775808, 9223372036854775807\]"),
        (pl.DT_UINT64, 8, -1, r"representable in uint64, i\.e\. in \[0, 18446744073709551615\], got -1"),
    ],
)
def test_expands_rejects_scalar_outside_the_out_dtype(dtype, elem_bytes, scalar, expected):
    with pytest.raises(FinalRejectionError, match=rf"pl\.expands: scalar operand must be {expected}"):
        _parse_expands_kernel(dtype, scalar, elem_bytes)


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    ("dtype", "elem_bytes", "scalar"),
    [
        (pl.DT_INT8, 1, 127),
        (pl.DT_INT8, 1, -128),
        (pl.DT_UINT8, 1, 255),
        (pl.DT_INT64, 8, 2**63 - 1),
        (pl.DT_UINT64, 8, 2**64 - 1),
    ],
)
def test_expands_accepts_a_boundary_scalar(dtype, elem_bytes, scalar):
    _parse_expands_kernel(dtype, scalar, elem_bytes)


@pytest.mark.soc("950")
def test_expands_still_rejects_a_scalar_outside_the_storage_band():
    """Above UINT64_MAX the band check fires first, before any dtype is consulted."""
    with pytest.raises(FinalRejectionError, match=r"must be in \[-9223372036854775808, 18446744073709551615\]"):
        _parse_expands_kernel(pl.DT_UINT64, UINT64_MAX + 1, 8)


# ---------------------------------------------------------------------------
# The check must agree with the dtype the dst register is actually declared with
# ---------------------------------------------------------------------------
def _full_no_dtype_kwarg_body(scalar):
    """vf.full with no dtype kwarg: the dst dtype is inferred from the first source.

    The first source is the literal, whose FP32 placeholder therefore wins over the fp16 mask -- so
    the register really is fp32 and an fp32-valued scalar has to be accepted. Checking against the
    mask instead would reject a value the emitted register can hold.
    """

    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
        reg_out = vf.full(scalar, preg)
        vf.store_align(t_out, reg_out, preg)

    return vf_body


@pytest.mark.soc("950")
def test_scalar_is_checked_against_the_inferred_dst_dtype_not_another_source():
    program, _ = _parse_with_body(pl.DT_FP16, _full_no_dtype_kwarg_body(70000.0))

    text = str(program)
    assert "vf.full" in text
    # The dst is fp32 ("float"), which is why 70000.0 is legal here.
    assert "vf.reg_tensor(dtype=float)" in text
