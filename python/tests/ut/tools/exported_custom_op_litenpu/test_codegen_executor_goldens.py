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
"""Whole-text goldens for the GE registration and custom-executor emitters in
``exported_custom_op_litenpu.deploy.codegen``.

These pin the domi plugin TU, the ``REG_OP`` prototype, the ``PtoCustomOp`` subclass and the combined
``op_host`` TU that the built ``.so`` compiles, in the shipped forms: ``_generate_custom_executor_cpp``
and ``_generate_op_tu_cpp_with_snippet`` carry ``import_via="op_kernel"`` throughout, so a drift in the
deployed wrapper cannot hide behind a test-only path literal.
"""

from __future__ import annotations

# Running this file directly, with no conftest, needs the tools root on sys.path
# for the `exported_custom_op_litenpu.*` imports below.
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[5] / "tools"))

import importlib.util

from codegen_golden_data.harness import GOLDEN_SUFFIX, claimed_cases, golden_text
from exported_custom_op_litenpu.deploy import codegen as cpp_mod
import pytest
import torch

_UT_DIR = _pl.Path(__file__).resolve().parent
DATA_DIR = _UT_DIR / "codegen_golden_data"
_FRAMEWORK_TYPE__ONNX = "ONNX"

_BIAS = {"name": "bias", "type": "Int", "default": 0}
_ALPHA = {"name": "alpha", "type": "Float", "default": 0.01}
_MODE = {"name": "mode", "type": "String", "default": "sum"}
_CROP = {"name": "crop_size", "type": "ListInt", "default": [2, 2]}

# A required attr (no default) followed by a defaulted one: also the declaration-index contract case,
# since ExtractAttrs indexes by position in this list and not by position among the required ones.
_SPLITV = [{"name": "split_sizes", "type": "ListInt", "default": None},
           {"name": "dim", "type": "Int", "default": 0}]


def _load(unique_name: str, filename: str):
    """Load a sibling ``*.py`` sample module by path (works without a parent package)."""
    spec = importlib.util.spec_from_file_location(unique_name, _UT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


shape_samples = _load("infer_shape_samples_ut_exec_goldens", "infer_shape_samples.py")
kernel_samples = _load("kernel_compile_samples_ut_exec_goldens", "cpp_kernel_compile_samples.py")


def _executor_kwargs(**overrides):
    """The factory-mode keyword set every whole-TU case shares."""
    kwargs = dict(
        create_kernel_func=kernel_samples.create_add_kernel,
        infer_shape_func=kernel_samples.add_infer_shape,
        infer_dtype_func=kernel_samples.add_infer_dtype,
        kernel_body_func=kernel_samples.add_kernel_body,
        dtypes=[torch.float16, torch.float16],
    )
    kwargs.update(overrides)
    return kwargs


def _shape_meta(func, attr_specs=None):
    return cpp_mod._parse_infer_shape_for_codegen(func, attr_specs)


def _dtype_meta(func):
    return cpp_mod._parse_infer_dtype_for_codegen(func)


def _attr_parse(spec_dict):
    return cpp_mod._attr_parse_param_block(cpp_mod._normalize_attr_specs([spec_dict])[0])


# A case name carries its own language extension; the golden is that name plus GOLDEN_SUFFIX.
GOLDEN_CASES = {
    "plugin_stub.cpp": lambda: cpp_mod._generate_op_custom_plugin_cpp(
        "AddBias", framework_type=_FRAMEWORK_TYPE__ONNX),
    "plugin_int_attr.cpp": lambda: cpp_mod._generate_op_custom_plugin_cpp(
        "AddBias", framework_type=_FRAMEWORK_TYPE__ONNX, attr_specs=[_BIAS]),
    "plugin_float_string_listint.cpp": lambda: cpp_mod._generate_op_custom_plugin_cpp(
        "MixOp", framework_type=_FRAMEWORK_TYPE__ONNX, attr_specs=[_ALPHA, _MODE, _CROP]),
    "plugin_caffe_domain_opset.cpp": lambda: cpp_mod._generate_op_custom_plugin_cpp(
        "MyOp", framework_type="CAFFE", domain="pypto_x", domain_opset_version=11),
    "reg_op_no_attrs.inc": lambda: cpp_mod._reg_op_block(
        kernel_samples.add_infer_shape, kernel_samples.add_infer_dtype,
        op_type="AddBias", dtypes=[torch.float16, torch.float16]),
    "reg_op_required_and_default_attr.inc": lambda: cpp_mod._reg_op_block(
        kernel_samples.add_infer_shape, kernel_samples.add_infer_dtype,
        op_type="SplitV", dtypes=[torch.float16, torch.float16], attr_specs=_SPLITV),
    "reg_op_two_outputs.inc": lambda: cpp_mod._reg_op_block(
        shape_samples.infer_shape_two_outputs, shape_samples.infer_dtype_two_outputs,
        op_type="DualOut", dtypes=[torch.float16, torch.float32]),
    "executor_class_no_attrs.inc": lambda: cpp_mod._custom_executor_class_cpp(
        "Add",
        infer_shape_meta=_shape_meta(kernel_samples.add_infer_shape),
        infer_dtype_meta=_dtype_meta(kernel_samples.add_infer_dtype)),
    "executor_class_two_attrs.inc": lambda: cpp_mod._custom_executor_class_cpp(
        "SplitV",
        infer_shape_meta=_shape_meta(kernel_samples.add_infer_shape, _SPLITV),
        infer_dtype_meta=_dtype_meta(kernel_samples.add_infer_dtype),
        attr_specs=_SPLITV),
    "executor_no_attrs.cpp": lambda: cpp_mod._generate_custom_executor_cpp(
        "Add", **_executor_kwargs()),
    "executor_int_attr.cpp": lambda: cpp_mod._generate_custom_executor_cpp(
        "AddBias", attr_specs=[_BIAS], **_executor_kwargs()),
    "executor_float_string_listint.cpp": lambda: cpp_mod._generate_custom_executor_cpp(
        "MixOp", attr_specs=[_ALPHA, _MODE, _CROP], **_executor_kwargs()),
    "executor_two_outputs.cpp": lambda: cpp_mod._generate_custom_executor_cpp(
        "DualOut", **_executor_kwargs(
            infer_shape_func=shape_samples.infer_shape_two_outputs,
            infer_dtype_func=shape_samples.infer_dtype_two_outputs,
            dtypes=[torch.float16, torch.float32])),
    # Element 1 of the pair is #5601's kernel snippet and is goldened with that emitter.
    "op_host_tu_no_attrs.cpp": lambda: cpp_mod._generate_op_tu_cpp_with_snippet(
        "Add", **_executor_kwargs())[0],
    "op_host_tu_with_attrs.cpp": lambda: cpp_mod._generate_op_tu_cpp_with_snippet(
        "SplitV", attr_specs=_SPLITV, **_executor_kwargs())[0],
    "attr_parse_param_float.inc": lambda: _attr_parse(_ALPHA),
    "attr_parse_param_string.inc": lambda: _attr_parse(_MODE),
    "attr_parse_param_listint.inc": lambda: _attr_parse(_CROP),
    "attr_reg_op_line_required.inc": lambda: cpp_mod._attr_reg_op_line(
        cpp_mod._normalize_attr_specs([_SPLITV[0]])[0]),
    "attr_extract_attrs_listint_int.inc": lambda: cpp_mod._attr_extract_attrs_method(
        cpp_mod._normalize_attr_specs(_SPLITV)),
}


# Parametrized from the case table rather than the directory, so a missing golden fails instead of
# quietly shrinking the parameter set.
@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_generated_text_matches_golden(case):
    path = DATA_DIR / (case + GOLDEN_SUFFIX)
    assert path.exists(), f"{case}{GOLDEN_SUFFIX} is missing; run codegen_golden_data/render.py"
    golden = path.read_text(encoding="utf-8")
    assert golden.strip(), f"{case}{GOLDEN_SUFFIX} is empty"
    assert golden_text(GOLDEN_CASES[case]()) == golden, (
        f"{case}{GOLDEN_SUFFIX} is out of sync with the emitter; run codegen_golden_data/render.py"
    )


def test_every_golden_on_disk_is_claimed_by_a_case():
    """A golden no case builds is dead weight that render.py would delete."""
    on_disk = {p.name[:-len(GOLDEN_SUFFIX)] for p in DATA_DIR.glob("*" + GOLDEN_SUFFIX)}
    claimed, n_suites = claimed_cases()
    assert on_disk <= claimed, f"unclaimed goldens: {sorted(on_disk - claimed)} (suites: {n_suites})"


def test_no_golden_would_be_rewritten_by_pre_commit():
    """trailing-whitespace and end-of-file-fixer have no file filter, so they apply to the goldens."""
    checked = 0
    for p in sorted(DATA_DIR.glob("*" + GOLDEN_SUFFIX)):
        text = p.read_text(encoding="utf-8")
        assert text.endswith("\n"), f"{p.name} does not end with a newline"
        assert "\r" not in text, f"{p.name} carries CR"
        for i, line in enumerate(text.split("\n"), 1):
            assert line == line.rstrip(), f"{p.name}:{i} has trailing whitespace"
        checked += 1
    claimed, n_suites = claimed_cases()
    assert checked == len(claimed), f"checked {checked} goldens, {len(claimed)} cases in {n_suites} suites"
