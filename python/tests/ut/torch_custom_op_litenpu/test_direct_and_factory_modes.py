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
"""Unit tests for the polymorphic ``kernel=`` (DIRECT vs FROM_FACTORY) and the emitted snippet glue.

No backend: these assert the discriminator routing, the export-time factory sanity check, and the
compile-snippet glue string for each mode. The real-backend DIRECT build lives in
``test_from_jit_kernel.py``.
"""
import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

import pypto
from pypto.extensions.torch_custom_op_litenpu.common import kernel_snippet
from pypto.extensions.torch_custom_op_litenpu.common.exported_custom_op import ExportedCustomOp

_EXPORT_TEST_DIR = Path(__file__).resolve().parent


def _load_samples():
    path = _EXPORT_TEST_DIR / "kernel_snippet_samples.py"
    spec = importlib.util.spec_from_file_location("kernel_snippet_samples_modes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


samples = _load_samples()


# ---- a real, source-capturable DIRECT jit kernel + infer hooks (module-level so getsource works) ----
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def direct_add_kernel(
    input0: pypto.Tensor([...]),
    input1: pypto.Tensor([...]),
    output: pypto.Tensor([...]),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output.move(input0 + input1)


def direct_infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def direct_infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def test_op_kernel_direct_form_sets_bare_kernel_fn():
    op = ExportedCustomOp(kernel=direct_add_kernel, infer_shape=direct_infer_shape, infer_dtype=direct_infer_dtype)
    assert op._bare_kernel_fn is direct_add_kernel
    assert op._create_kernel_fn is None


def test_direct_kernel_arity_mismatch_is_rejected_at_construction():
    # The kernel takes 2 tensors but the hooks declare 2 inputs + 1 output. Without the gate this
    # first fails at runtime compile (possibly on the device) with an FeError about tensor counts.
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def two_param_kernel(input0: pypto.Tensor([...]), output: pypto.Tensor([...])):
        output.move(input0 + input0)

    with pytest.raises(ValueError, match="tensor parameter"):
        ExportedCustomOp(kernel=two_param_kernel, infer_shape=direct_infer_shape,
                         infer_dtype=direct_infer_dtype)


def test_op_kernel_factory_form_sets_create_kernel_fn():
    op = ExportedCustomOp(kernel=samples.create_add_kernel, infer_shape=samples.add_infer_shape,
                          infer_dtype=samples.add_infer_dtype)
    assert op._create_kernel_fn is samples.create_add_kernel
    assert op._bare_kernel_fn is None


def test_factory_inner_kernel_arity_mismatch_is_rejected_at_construction():
    # The DIRECT gate's factory counterpart: the inner jit kernel takes 2 tensors but the hooks
    # declare 2 inputs + 1 output. Without it the mismatch first fails at runtime compile.
    def create_two_param_kernel(shape, dtype, soc_version):
        @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
        def two_param_inner(input0: pypto.Tensor([...]), output: pypto.Tensor([...])):
            output.move(input0 + input0)

        return two_param_inner

    with pytest.raises(ValueError, match="inner jit kernel declares"):
        ExportedCustomOp(kernel=create_two_param_kernel, infer_shape=direct_infer_shape,
                         infer_dtype=direct_infer_dtype)


def test_factory_inner_kernel_matching_arity_is_accepted():
    # The same shape of factory with the arity the hooks declare constructs without complaint.
    def create_three_param_kernel(shape, dtype, soc_version):
        @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
        def three_param_inner(input0: pypto.Tensor([...]), input1: pypto.Tensor([...]),
                              output: pypto.Tensor([...])):
            output.move(input0 + input1)

        return three_param_inner

    op = ExportedCustomOp(kernel=create_three_param_kernel, infer_shape=direct_infer_shape,
                          infer_dtype=direct_infer_dtype)
    assert op._create_kernel_fn is create_three_param_kernel


def test_op_kernel_rejects_plain_non_factory():
    def add_body(x, y, out):  # a plain body: no inner @jit def, no return
        out.move(x + y)

    def create_add_kernel_no_return(shapes, dtypes, soc_version):  # inner @jit def, never returned
        @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
        def inner_add(a: pypto.Tensor([...]), b: pypto.Tensor([...]), out: pypto.Tensor([...])):
            out.move(a + b)

    for bad in (add_body, create_add_kernel_no_return):
        with pytest.raises(ValueError, match="nor a factory that defines and returns one"):
            ExportedCustomOp(kernel=bad, infer_shape=direct_infer_shape, infer_dtype=direct_infer_dtype)


def test_direct_decorator_with_soc_is_rejected_at_export():
    # A DIRECT kernel whose jit decorator pins soc_version would clobber the global at build time.
    from pypto.extensions.torch_custom_op_litenpu.common.authoring import _validate_direct_decorator_no_soc

    @pypto.frontend.jit(
        codegen_options={"soc_version": "Kirin9030"},
        runtime_options={"run_mode": pypto.RunMode.SIM},
    )
    def bad_kernel(a: pypto.Tensor([...]), b: pypto.Tensor([...]), out: pypto.Tensor([...])):
        out.move(a + b)

    with pytest.raises(ValueError, match="soc_version"):
        _validate_direct_decorator_no_soc(bad_kernel)


def test_direct_soc_pinned_kernel_is_rejected_at_construction():
    # ExportedCustomOp.__init__ validates the kernel: a DIRECT jit kernel that pins soc_version
    # must RAISE at construction, not later at export.
    @pypto.frontend.jit(
        codegen_options={"soc_version": "Kirin9030"},
        runtime_options={"run_mode": pypto.RunMode.SIM},
    )
    def bad_kernel(a: pypto.Tensor([...]), b: pypto.Tensor([...]), out: pypto.Tensor([...])):
        out.move(a + b)

    with pytest.raises(ValueError, match="soc_version"):
        ExportedCustomOp(
            torch_op_qualname="pypto::direct_soc_pinned",
            kernel=bad_kernel, infer_shape=direct_infer_shape, infer_dtype=direct_infer_dtype,
        )


def test_snippet_jit_kernel_mode_glue():
    snippet = kernel_snippet.build_kernel_compile_snippet(
        mode="jit_kernel",
        jit_kernel_func=direct_add_kernel,
        infer_shape_func=direct_infer_shape,
        infer_dtype_func=direct_infer_dtype,
        n_inputs=2,
    )
    compile(snippet, "<snippet>", "exec")
    # DIRECT: the jit decorator is KEPT (nothing else to strip); glue passes jit_kernel=.
    assert "@pypto.frontend.jit" in snippet
    assert "def direct_add_kernel(" in snippet
    assert "pypto.extensions.torch_custom_op_litenpu.CompileEntry(\n    jit_kernel=" in snippet
    assert "num_inputs=2" in snippet
    assert "num_outputs=1" in snippet
    assert "_custom_op_compile.infer_shape = direct_infer_shape" in snippet
    assert "def __pypto_compile(in_shapes, in_dtypes, attrs, soc_version):" in snippet


def test_snippet_factory_mode_glue():
    snippet = kernel_snippet.build_kernel_compile_snippet(
        mode="factory",
        create_kernel_func=samples.create_add_kernel,
        infer_shape_func=samples.add_infer_shape,
        infer_dtype_func=samples.add_infer_dtype,
        n_inputs=2,
        kernel_body_func=samples.add_kernel_body,
    )
    compile(snippet, "<snippet>", "exec")
    assert "pypto.extensions.torch_custom_op_litenpu.CompileEntry(\n    factory=" in snippet
    assert "jit_kernel=" not in snippet
    assert "def create_add_kernel(" in snippet


def test_snippet_jit_kernel_mode_requires_jit_kernel_func():
    with pytest.raises(ValueError, match="jit_kernel"):
        kernel_snippet.build_kernel_compile_snippet(
            mode="jit_kernel",
            infer_shape_func=direct_infer_shape,
            infer_dtype_func=direct_infer_dtype,
            n_inputs=2,
        )


def test_detect_factory_signature_from_signature():
    """The kernel= factory calling convention is inferred from the factory's leading params."""
    from pypto.extensions.torch_custom_op_litenpu.common.authoring import _detect_factory_signature

    def create_single(shape, dtype, soc_version, run_mode=None): ...
    def create_lists(shapes, dtypes, soc_version, run_mode=None): ...
    def create_full(shapes, dtypes, attrs, soc_version, run_mode=None): ...

    assert _detect_factory_signature(create_single) == "single"
    assert _detect_factory_signature(create_lists) == "lists"
    assert _detect_factory_signature(create_full) == "full"

    def create_bad(s, d, soc): ...
    with pytest.raises(ValueError, match="unrecognized signature"):
        _detect_factory_signature(create_bad)
