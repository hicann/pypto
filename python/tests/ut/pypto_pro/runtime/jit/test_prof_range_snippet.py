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

"""The aclprof tensor-info range embedded in the generated launcher.

``_generate_prof_range_snippet`` turns IR param specs into an ``aclprofTensor[]``
initializer pushed around the ``<<<>>>`` launch, so msprof can attach shape/dtype/
format per tensor to the kernel row (Tx channel -- independent of the mspf node
chain). These tests check the generated snippet text directly: tensor direction
comes from the store-target AST inference, static dims become literals, named
dynamic dims reference the launcher's dim parameters, and unknown dynamics
degrade to 0.

One pipeline test drives ``_parse_and_codegen_targets`` into
``_generate_caller_cpp`` end-to-end, checking the include, the registered op
name, the push/pop placement around the launch line and both tensor directions
on the generated caller source.
"""

import pypto_pro.language as pl
from pypto_pro.runtime.jit import (
    ParamKind,
    ParamSpec,
    _generate_caller_cpp,
    _generate_prof_range_snippet,
    _parse_and_codegen_targets,
)


def _snippet(kernel_name, specs, dims):
    return _generate_prof_range_snippet(
        kernel_name, specs, dims, launch_stmt="    k<<<blockDim, nullptr, stream>>>(a, out);\n"
    )


def _spec(name, shape, dtype_str, direction):
    return ParamSpec(name, ParamKind.TENSOR, dtype_str, shape, direction)


def test_annotation_marker_sets_direction():
    """pl.Output in the annotation flows to ParamSpec.direction; omitted defaults to in."""
    @pl.jit()
    def marked_kernel(
        a: pl.Tensor[[64, 64], pl.DT_FP32],
        out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.store(out, ta, [0, 0])

    cube, vector = _parse_and_codegen_targets(marked_kernel.to_kernel_def(), "a5", "")
    cg = cube or vector
    assert cg.param_specs[0].direction == "in"
    assert cg.param_specs[1].direction == "out"


def test_no_marker_defaults_to_in():
    """Without pl.Output, a tensor param is input even if the kernel stores into it."""

    @pl.jit()
    def unmarked_kernel(
        out: pl.Tensor[[64, 64], pl.DT_FP32],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        with pl.section_vector():
            pl.store(out, ta, [0, 0])

    cube, vector = _parse_and_codegen_targets(unmarked_kernel.to_kernel_def(), "a5", "")
    cg = cube or vector
    assert cg.param_specs[0].direction == "in"


def test_static_shapes_are_literals():
    specs = [
        _spec("a", [64, 64], "fp32", "in"),
        _spec("b", [64, 64], "fp32", "in"),
        _spec("out", [64, 64], "fp32", "out"),
    ]
    _inc, push, pop = _snippet("k", specs, set())
    assert '#include "acl/acl_prof.h"' in _inc
    assert "{0, 2, 0, 2, {64, 64, 0, 0, 0, 0, 0, 0}}," in push  # a: in
    assert "{1, 2, 0, 2, {64, 64, 0, 0, 0, 0, 0, 0}}," in push  # out: out
    assert 'aclprofRangePushEx(&pyptoProfAttrs);' in push
    assert '(void)aclprofRangePop();' in pop
    assert "aclprofStr2Id(\"PYPTO_k\")" in push
    assert "pyptoProfInfo.tensorNum = 3;" in push


def test_dynamic_dims_reference_launcher_params():
    dims = {"a", "b", "out", "__pypto_dyn_a_0", "__pypto_dyn_a_1"}
    specs = [_spec("a", ["__pypto_dyn_a_0", "__pypto_dyn_a_1"], "fp32", "in")]
    _inc, push, _pop = _snippet("k", specs, dims)
    assert "{static_cast<uint32_t>(__pypto_dyn_a_0), static_cast<uint32_t>(__pypto_dyn_a_1)," in push


def test_unknown_dynamic_degrades_to_zero():
    specs = [_spec("a", [-1, 8], "fp16", "in")]
    _inc, push, _pop = _snippet("k", specs, set())
    assert "{0, 2, 1, 2, {0, 8, 0, 0, 0, 0, 0, 0}}," in push


def test_dtype_mapping():
    specs = [
        _spec("a", [4], "fp16", "in"),
        _spec("b", [4], "bf16", "in"),
        _spec("c", [4], "int32", "in"),
        _spec("d", [4], "nonexistent", "in"),
    ]
    _inc, push, _pop = _snippet("k", specs, set())
    assert "{0, 2, 1, 1," in push  # fp16 -> 1
    assert "{0, 2, 27, 1," in push  # bf16 -> 27
    assert "{0, 2, 3, 1," in push  # int32 -> 3
    assert "{0, 2, 0, 1," in push  # unknown dtype -> 0 fallback


def test_no_tensor_params_disables_snippet():
    assert _snippet("k", [], set()) == ("", "", "")


def test_caller_embeds_range_around_launch_static():
    """End-to-end caller generation: pl.Output marker flows into the snippet."""
    @pl.jit()
    def static_kernel(
        a: pl.Tensor[[64, 64], pl.DT_FP32],
        out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        tc = pl.make_tile(tt, addr=0x8000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.add(tc, ta, ta)
            pl.store(out, tc, [0, 0])

    cube, vector = _parse_and_codegen_targets(static_kernel.to_kernel_def(), "a5", "")
    from pypto_pro.runtime.compile_config import get_jit_compile_config

    target = get_jit_compile_config().resolve_kernel_target("a5", has_cube=cube is not None,
                                                            has_vector=vector is not None)
    cg = cube or vector
    content = _generate_caller_cpp(
        kernel_params=cg.kernel_params,
        kernel_cpp_name="kernel.cpp",
        kernel_name=cg.kernel_name,
        target=target,
        prof_param_specs=cg.param_specs,
    )
    assert '#include "acl/acl_prof.h"' in content
    assert 'aclprofStr2Id("PYPTO_static_kernel")' in content
    push_idx = content.index("aclprofRangePushEx")
    launch_idx = content.index("<<<blockDim")
    pop_idx = content.index("aclprofRangePop")
    assert push_idx < launch_idx < pop_idx
    assert "{0, 2, 0, 2, {64, 64" in content   # a: in
    assert "{1, 2, 0, 2, {64, 64" in content   # out: out


def test_guard_wraps_struct_definitions():
    """The MsprofGetPath switch wraps the struct definitions; else branch launches bare."""
    @pl.jit()
    def bare_kernel(
        a: pl.Tensor[[64, 64], pl.DT_FP32],
        out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        tc = pl.make_tile(tt, addr=0x8000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.add(tc, ta, ta)
            pl.store(out, tc, [0, 0])

    cube, vector = _parse_and_codegen_targets(bare_kernel.to_kernel_def(), "a5", "")
    from pypto_pro.runtime.compile_config import get_jit_compile_config

    target = get_jit_compile_config().resolve_kernel_target("a5", has_cube=cube is not None,
                                                            has_vector=vector is not None)
    cg = cube or vector
    content = _generate_caller_cpp(
        kernel_params=cg.kernel_params,
        kernel_cpp_name="kernel.cpp",
        kernel_name=cg.kernel_name,
        target=target,
        prof_param_specs=cg.param_specs,
    )
    on_idx = content.index("if (pyptoProfOn) {")
    tensors_idx = content.index("aclprofTensor pyptoProfTensors[]")
    push_idx = content.index("aclprofRangePushEx")
    report_idx = content.index("pypto::ReportCaptureTensorInfo")
    pop_idx = content.index("aclprofRangePop")
    else_idx = content.index("} else {")
    assert on_idx < tensors_idx < push_idx  # definitions inside the guard
    assert push_idx < report_idx < pop_idx  # report lands between launch and pop
    assert pop_idx < else_idx               # pop stays in the same guarded block
    assert content.count("<<<blockDim") == 2  # instrumented branch + bare else branch
    assert "ReportCaptureTensorInfo" not in content[else_idx:]  # else launches uninstrumented
