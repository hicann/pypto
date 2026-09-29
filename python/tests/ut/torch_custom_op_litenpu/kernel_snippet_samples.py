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
"""Real-file kernel/infer samples for the snippet generator, following the ``create_*_kernel``
factory contract and the DIRECT ``@pypto.frontend.jit`` form.

These live in a real module so ``inspect.getsource`` captures them (the snippet generator and the jit
parser both need real source on disk); tests pass them to ``kernel_snippet.build_kernel_compile_snippet``
directly, not through an ``ExportedCustomOp``. The two annotation-only DIRECT kernels come from the
leaf ``kernel_compile_samples`` module and are re-exported here so a snippet test needs one import.
"""

import os
import sys

import torch

import pypto

# Make the sibling helper modules (``sample_helper_pkg``, ``kernel_compile_samples``, the collision and
# import-hoist fixtures) importable when this module is loaded standalone.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Imported as a module so kernels below can reference its helpers via dotted chains.
from kernel_compile_samples import (  # noqa: E402, F401 - re-exported for the snippet tests
    direct_add_kernel_erased,
    direct_add_kernel_pinned,
)
import sample_helper_pkg  # noqa: E402
from sample_helper_pkg.compute import add_into, tile_helper  # noqa: E402 (bare-name refs)


def add_kernel_body(input0, input1):
    pypto.set_vec_tile_shapes(*add_infer_tileshape(input0, input1))
    return input0 + input1


def add_infer_tileshape(input0, input1):
    return (1, 4, 1, 64)


# Annotated like a real ``infer_shape`` / ``infer_dtype`` pair (the return annotation drives
# the output arity).
def add_infer_shape(input0_shape: torch.Size, input1_shape: torch.Size) -> torch.Size:
    return input0_shape


def add_infer_dtype(input0_dtype: torch.dtype, input1_dtype: torch.dtype) -> torch.dtype:
    return input0_dtype


def create_add_kernel(shape, dtype, soc_version, run_mode=pypto.RunMode.SIM):
    @pypto.frontend.jit(
        codegen_options={"soc_version": soc_version},
        runtime_options={"run_mode": run_mode},
    )
    def add_kernel(
        input0: pypto.Tensor([...], dtype),
        input1: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        output.move(add_kernel_body(input0, input1))

    return add_kernel


# Module-level constant the inlined kernel below closes over (exercises constant auto-capture).
_SAMPLE_TILE = (1, 4, 1, 64)


def create_add_kernel_inlined(shape, dtype, soc_version, run_mode=pypto.RunMode.SIM):
    """Inlines all compute (no separate ``*_body``), so it is valid with ``kernel_body_func`` omitted."""
    @pypto.frontend.jit(
        codegen_options={"soc_version": soc_version},
        runtime_options={"run_mode": run_mode},
    )
    def add_kernel(
        input0: pypto.Tensor([...], dtype),
        input1: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*_SAMPLE_TILE)
        output.move(input0 + input1)

    return add_kernel


def create_add_kernel_cross_module(shape, dtype, soc_version, run_mode=pypto.RunMode.SIM):
    """Factory whose kernel calls helpers from another module, so the factory path exercises the
    cross-module canonical helper naming (``sample_helper_pkg__compute__*``)."""
    @pypto.frontend.jit(
        codegen_options={"soc_version": soc_version},
        runtime_options={"run_mode": run_mode},
    )
    def add_kernel(
        input0: pypto.Tensor([...], dtype),
        input1: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_helper())
        add_into(input0, input1, output)

    return add_kernel


# ---- DIRECT (jit_kernel) samples for the tracer ----

# DIRECT kernel whose source decorator is run_mode=NPU; the deploy snippet must rewrite the decorator
# to SIM while the body reference to RunMode (below) stays untouched.
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def direct_add_kernel_npu_source(
    input0: pypto.Tensor([...]),
    input1: pypto.Tensor([...]),
    output: pypto.Tensor([...]),
):
    _unused = pypto.RunMode.NPU  # body reference to RunMode, not rewritten
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output.move(input0 + input1)


# DIRECT kernel pinning shape + dtype via named module constants referenced only from the annotations
# (never from the body), so the annotation channel alone must capture _PIN_SHAPE / _PIN_DTYPE.
_PIN_B, _PIN_N, _PIN_R, _PIN_C = 1, 8, 1, 64
_PIN_SHAPE = (_PIN_B, _PIN_N, _PIN_R, _PIN_C)
_PIN_DTYPE = pypto.DT_FP16


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def direct_add_kernel_pinned_const(
    input0: pypto.Tensor(_PIN_SHAPE, _PIN_DTYPE),
    input1: pypto.Tensor(_PIN_SHAPE, _PIN_DTYPE),
    output: pypto.Tensor(_PIN_SHAPE, _PIN_DTYPE),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output.move(input0 + input1)


# DIRECT kernel calling helpers defined in a different file; the reference chains:
#   direct_multifile_kernel -> tile_helper (plain, other file) -> TILE (const, third file)
#   direct_multifile_kernel -> add_into (@function, other file) -> _bias (plain, other file)
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def direct_multifile_kernel(
    input0: pypto.Tensor([...]),
    input1: pypto.Tensor([...]),
    output: pypto.Tensor([...]),
):
    pypto.set_vec_tile_shapes(*tile_helper())
    add_into(input0, input1, output)


# Reaches the same helpers via the module-rooted dotted chain ``sample_helper_pkg.compute.<fn>(...)``
# rather than the bare names.
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def direct_dotted_helper_kernel(
    input0: pypto.Tensor([...]),
    input1: pypto.Tensor([...]),
    output: pypto.Tensor([...]),
):
    pypto.set_vec_tile_shapes(*sample_helper_pkg.compute.tile_helper())
    sample_helper_pkg.compute.add_into(input0, input1, output)


# ---- Function-local imports inside an entry body (kernel body + infer hook) ----

# The body-local ``from import_hoist_bias import BIAS`` must hoist: import gone from the emitted kernel block,
# BIAS packed as a const.
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def direct_kernel_local_import(
    input0: pypto.Tensor([...]),
    input1: pypto.Tensor([...]),
    output: pypto.Tensor([...]),
):
    from import_hoist_bias import BIAS
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output.move(input0 + input1 + BIAS)


def infer_shape_local_import(input0_shape: torch.Size, input1_shape: torch.Size) -> torch.Size:
    """Infer hook with a body-local ``import math``, hoisted to a header import."""
    import math
    _ = math.prod(input0_shape)
    return input0_shape


# ---- Annotation-vs-body constant name collision ----

# The kernel body reaches annot_collision_lib.PIN_SHAPE (value A) while the kernel annotation pins this
# module's same-named PIN_SHAPE (value B); the two values must pack under distinct names.
from annot_collision_lib import body_shape  # noqa: E402

# Same bare name as annot_collision_lib.PIN_SHAPE, different object + value.
PIN_SHAPE = (1, 1, 8, 64)  # value B, pinned in the kernel annotation below


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def direct_annot_collision_kernel(
    input0: pypto.Tensor(PIN_SHAPE, pypto.DT_FP16),
    input1: pypto.Tensor(PIN_SHAPE, pypto.DT_FP16),
    output: pypto.Tensor(PIN_SHAPE, pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(*body_shape())
    output.move(input0 + input1)
