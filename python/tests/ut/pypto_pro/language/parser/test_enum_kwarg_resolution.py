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
"""Unit tests for enum-valued kwarg resolution in DSL function bodies.

Covers the "support const enum" / "expects an enum value" changes:

  * An enum kwarg (dtype / target_type / mode / ...) may be written as an enum
    literal (``pl.DT_FP32`` / ``pl.RoundMode.X``), or captured from a closure
    variable — including multi-level closure assignment (``a = enum; b = a``)
    and kernel-factory ``dtype`` parameters used for dtype generalization.
  * The same enum kwarg rejects a plain int, whether written directly
    (``target_type=1``) or captured from an int closure variable — raising
    ParserTypeError ("expects an enum value").
"""
from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError, UnsupportedFeatureError
import pytest

# Tile geometry for the VF-op tests below.
_VF_N, _VF_M = 1, 64
_VF_TILE_SIZE = _VF_N * _VF_M * 4  # 32-byte aligned


# ---------------------------------------------------------------------------
# Legal: enum kwarg written as an enum literal
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_dtype_kwarg_enum_literal():
    """A dtype kwarg written directly as pl.DT_* resolves to a DataType."""

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
        result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(x, target_type=pl.DT_FP32)
        return result

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_mode_kwarg_enum_literal():
    """A non-dtype enum kwarg (RoundMode) written directly is accepted."""

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
        result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(
            x, target_type=pl.DT_FP32, mode=pl.RoundMode.CAST_ROUND
        )
        return result

    assert isinstance(func, ir.Function)


# ---------------------------------------------------------------------------
# Legal: enum kwarg captured from a closure variable
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_dtype_kwarg_closure_enum_var():
    """A dtype kwarg captured from a one-level closure enum variable is accepted."""
    dt = pl.DT_FP32

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
        result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(x, target_type=dt)
        return result

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_mode_kwarg_closure_enum_var():
    """A RoundMode kwarg captured from a closure enum variable is accepted."""
    rounding = pl.RoundMode.CAST_FLOOR

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
        result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(
            x, target_type=pl.DT_FP32, mode=rounding
        )
        return result

    assert isinstance(func, ir.Function)


# ---------------------------------------------------------------------------
# Legal: multi-level closure assignment  a = enum; b = a; kwarg=b
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_dtype_kwarg_multilevel_closure_assignment():
    """An enum passed through multiple closure assignments still resolves."""
    dt_a = pl.DT_FP32
    dt_b = dt_a

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
        result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(x, target_type=dt_b)
        return result

    assert isinstance(func, ir.Function)


# ---------------------------------------------------------------------------
# Legal: kernel-factory dtype generalization (closure dtype param)
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_kernel_factory_closure_dtype():
    """A dtype closure parameter drives both the annotation and a dtype kwarg."""

    def make_kernel(dtype):
        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], dtype]:
            result: pl.Tensor[[64, 128], dtype] = pl.tensor.cast(x, target_type=dtype)
            return result

        return func

    assert isinstance(make_kernel(pl.DT_FP32), ir.Function)
    assert isinstance(make_kernel(pl.DT_INT32), ir.Function)


# ---------------------------------------------------------------------------
# Illegal: enum kwarg given a plain int (literal)
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_dtype_kwarg_int_literal_rejected():
    """A dtype kwarg given a raw int literal raises ParserTypeError."""
    with pytest.raises(ParserTypeError, match="expects an enum value"):

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
            result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(x, target_type=1)
            return result


@pytest.mark.soc("950")
def test_mode_kwarg_int_literal_rejected():
    """A RoundMode kwarg given a raw int literal raises ParserTypeError."""
    with pytest.raises(ParserTypeError, match="expects an enum value"):

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
            result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(
                x, target_type=pl.DT_FP32, mode=1
            )
            return result


# ---------------------------------------------------------------------------
# Illegal: enum kwarg given an int captured from a closure variable
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_dtype_kwarg_int_closure_var_rejected():
    """A dtype kwarg given an int closure variable raises ParserTypeError.

    ``target_type=iv`` and ``target_type=1`` reach the resolver as the same int,
    so the int closure variable is rejected exactly like the literal.
    """
    iv = 1
    with pytest.raises(ParserTypeError, match="expects an enum value"):

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
            result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(x, target_type=iv)
            return result


# ---------------------------------------------------------------------------
# MemorySpace enum kwarg (target_memory)
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_target_memory_enum_literal():
    """A MemorySpace kwarg written as an enum literal is accepted."""

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        a = pl.make_tile(tile_type, addr=0, size=16384)  # noqa: F841
        result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(x, target_type=pl.DT_FP32)
        return result

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_target_memory_int_rejected():
    """A MemorySpace kwarg given a raw int raises ParserTypeError."""
    with pytest.raises(ParserTypeError, match="expects an enum value"):

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
            tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=1)  # noqa: F841
            result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(x, target_type=pl.DT_FP32)
            return result


# ---------------------------------------------------------------------------
# dtype as a POSITIONAL arg (goes through builder args, not resolve_single_kwarg)
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_dtype_positional_enum_literal():
    """A dtype enum passed positionally (cast(x, pl.DT_FP32)) is accepted."""

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
        result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(x, pl.DT_FP32)
        return result

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_dtype_positional_int_not_guarded():
    """A positional int is NOT caught by the enum-kwarg guard.

    The guard only applies to keyword arguments (via ``resolve_single_kwarg``);
    positional args are parsed straight into the op's arg list. This documents
    that ``cast(x, 1)`` is not rejected by the enum guard (an int positional
    dtype is validated later at the C++ boundary, not here).
    """

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
        result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(x, 1)
        return result

    assert isinstance(func, ir.Function)


# ---------------------------------------------------------------------------
# Non-enum kwargs still accept ints (must NOT be caught by the enum guard)
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_cmp_mode_is_not_an_enum_kwarg():
    """``cmp_mode`` is a plain-int parameter and must be excluded from the guard.

    Guards against a regression where ``_VF_KWARG_ENUMS`` (which maps cmp_mode to
    CompareMode) is used verbatim as the guard list — that would wrongly reject
    the legitimate ``cmp_mode=<int>`` used by block cmp/gather.
    """
    from pypto_pro.language.parser._call_parser import CallParserMixin

    assert "cmp_mode" not in CallParserMixin._ENUM_KWARGS


@pytest.mark.soc("950")
def test_fractal_int_kwarg_still_allowed():
    """A numeric kwarg (fractal) still accepts a raw int (not an enum kwarg)."""

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Acc)
        a = pl.make_tile(tile_type, addr=0, size=16384, fractal=1024)  # noqa: F841
        result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.cast(x, target_type=pl.DT_FP32)
        return result

    assert isinstance(func, ir.Function)


# ---------------------------------------------------------------------------
# VF-op enum kwargs (parsed when the enclosing jit kernel is parsed)
# ---------------------------------------------------------------------------
def _parse_vf_mask_kernel(pattern):
    """Build a jit kernel whose section_vector calls a VF op with the given
    ``pattern`` kwarg, and force-parse it (VF ops are parsed as part of the
    enclosing kernel's Vector section)."""

    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pattern, dtype=pl.DT_FP32)
        reg_a = vf.load_align(in_a, 0)
        vf.store_align(t_out, reg_a, preg)

    @pl.jit()
    def kernel(a: pl.Tensor[[_VF_N, _VF_M], pl.DT_FP32], out: pl.Tensor[[_VF_N, _VF_M], pl.DT_FP32]):
        tf = pl.TileType(shape=[_VF_N, _VF_M], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        in_a = pl.make_tile(tf, addr=0, size=_VF_TILE_SIZE)
        t_out = pl.make_tile(tf, addr=_VF_TILE_SIZE, size=_VF_TILE_SIZE)
        with pl.section_vector():
            pl.load(in_a, a, [0, 0])
            vf_body(in_a, t_out)
            pl.store(out, t_out, [0, 0])

    return kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_vf_enum_kwarg_literal():
    """A VF-op enum kwarg (create_mask pattern) written as an enum literal parses."""
    _parse_vf_mask_kernel(pl.MaskPattern.ALL)


@pytest.mark.soc("950")
def test_vf_enum_kwarg_closure_var():
    """A VF-op enum kwarg captured from a closure enum variable parses."""
    pat = pl.MaskPattern.ALL
    _parse_vf_mask_kernel(pat)


@pytest.mark.soc("950")
def test_vf_enum_kwarg_int_rejected():
    """A VF-op enum kwarg given a raw int raises ParserTypeError."""
    with pytest.raises(ParserTypeError, match="expects an enum value"):
        _parse_vf_mask_kernel(1)


# ---------------------------------------------------------------------------
# pl.const() dtype validation (scalar_ops._parse_typed_constant)
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_const_valid_dtype_enum_literal():
    """pl.const() with a valid dtype enum literal (pl.DT_INT32) is accepted."""

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
        c = pl.const(42, pl.DT_INT32)  # noqa: F841
        return x

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_const_valid_dtype_closure_var():
    """pl.const() with a dtype from a closure variable is accepted."""
    dt = pl.DT_FP32

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
        c = pl.const(1.0, dt)  # noqa: F841
        return x

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_const_dtype_int_literal_rejected():
    """pl.const() with an int instead of a dtype raises ParserSyntaxError."""
    with pytest.raises(ParserSyntaxError, match="must be a dtype"):

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
            c = pl.const(42, 1)  # noqa: F841
            return x


@pytest.mark.soc("950")
def test_const_dtype_int_closure_var_rejected():
    """pl.const() with an int closure variable as dtype raises ParserSyntaxError."""
    bad_dtype = 1

    with pytest.raises(ParserSyntaxError, match="must be a dtype"):

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
            c = pl.const(42, bad_dtype)  # noqa: F841
            return x


@pytest.mark.soc("950")
def test_const_dtype_string_rejected():
    """pl.const() with a string instead of a dtype raises ParserSyntaxError."""
    with pytest.raises(ParserSyntaxError, match="must be a dtype"):

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
            c = pl.const(42, "fp32")  # noqa: F841
            return x


# ---------------------------------------------------------------------------
# Enum comparison: only == and != are supported
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_enum_compare_eq():
    """Enum == enum folds to a compile-time ConstBool (True case)."""

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
        if pl.DT_FP16 == pl.DT_FP16:
            pass
        return x

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_enum_compare_eq_false():
    """Enum == enum folds to a compile-time ConstBool (False case)."""

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
        if pl.DT_FP16 == pl.DT_FP32:
            pass
        return x

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_enum_compare_ne():
    """Enum != enum folds to a compile-time ConstBool."""

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
        if pl.DT_FP16 != pl.DT_FP32:
            pass
        return x

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_enum_compare_lt_rejected():
    """Enum < enum raises UnsupportedFeatureError (only == and != allowed)."""
    with pytest.raises(UnsupportedFeatureError, match="Only == and != are supported"):

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
            if pl.DT_FP16 < pl.DT_FP32:
                pass
            return x


@pytest.mark.soc("950")
def test_enum_compare_gt_rejected():
    """Enum > enum raises UnsupportedFeatureError."""
    with pytest.raises(UnsupportedFeatureError, match="Only == and != are supported"):

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
            if pl.DT_FP16 > pl.DT_FP32:
                pass
            return x


@pytest.mark.soc("950")
def test_enum_compare_le_rejected():
    """Enum <= enum raises UnsupportedFeatureError."""
    with pytest.raises(UnsupportedFeatureError, match="Only == and != are supported"):

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
            if pl.DT_FP16 <= pl.DT_FP32:
                pass
            return x


@pytest.mark.soc("950")
def test_enum_compare_ge_rejected():
    """Enum >= enum raises UnsupportedFeatureError."""
    with pytest.raises(UnsupportedFeatureError, match="Only == and != are supported"):

        @pl.function
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
            if pl.DT_FP16 >= pl.DT_FP32:
                pass
            return x


@pytest.mark.soc("950")
def test_enum_compare_closure_vars():
    """Enum comparison with closure variables folds correctly."""
    dt_a = pl.DT_FP32
    dt_b = pl.DT_FP32

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP16]:
        if dt_a == dt_b:
            pass
        return x

    assert isinstance(func, ir.Function)


if __name__ == "__main__":
    _tests = [
        test_dtype_kwarg_enum_literal,
        test_mode_kwarg_enum_literal,
        test_dtype_kwarg_closure_enum_var,
        test_mode_kwarg_closure_enum_var,
        test_dtype_kwarg_multilevel_closure_assignment,
        test_kernel_factory_closure_dtype,
        test_dtype_kwarg_int_literal_rejected,
        test_mode_kwarg_int_literal_rejected,
        test_dtype_kwarg_int_closure_var_rejected,
        test_target_memory_enum_literal,
        test_target_memory_int_rejected,
        test_dtype_positional_enum_literal,
        test_dtype_positional_int_not_guarded,
        test_cmp_mode_is_not_an_enum_kwarg,
        test_fractal_int_kwarg_still_allowed,
        test_vf_enum_kwarg_literal,
        test_vf_enum_kwarg_closure_var,
        test_vf_enum_kwarg_int_rejected,
        test_const_valid_dtype_enum_literal,
        test_const_valid_dtype_closure_var,
        test_const_dtype_int_literal_rejected,
        test_const_dtype_int_closure_var_rejected,
        test_const_dtype_string_rejected,
        test_enum_compare_eq,
        test_enum_compare_eq_false,
        test_enum_compare_ne,
        test_enum_compare_lt_rejected,
        test_enum_compare_gt_rejected,
        test_enum_compare_le_rejected,
        test_enum_compare_ge_rejected,
        test_enum_compare_closure_vars,
    ]
    for _t in _tests:
        _t()
        print(f"{_t.__name__} passed!")
    print(f"All {len(_tests)} enum-kwarg resolution tests passed!")
