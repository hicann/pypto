# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Root-function outcast tests for Assemble LogicalTensor versions."""

import pypto
from pypto import ir, pil

from ..test_common import _ssa_verify, check_snapshot

IR = """
# ir.program: main
@ir.function(type=ir.FunctionType.Orchestration)
def foo(a@0: ir.Tensor, aux@1: ir.Tensor, out@2: ir.Tensor):
    [aux_3@1, out_0@2] = CALL(a@0, attrs=["callee": "foo_PATH0_7"])
    return a@0, aux_3@1, out_0@2
@ir.function(type=ir.FunctionType.InCore)
def foo_PATH0_7(a_3@11: ir.Tensor, aux_7@13: ir.Tensor, out_4@15: ir.Tensor):
    INCAST_LOCAL_BUF0_0@12 = VIEW(a_3@11, attrs=["fromOffset": [0, 0], "toValidShape": [32, 16]])
    [OUTCAST_LOCAL_BUF0@14, OUTCAST_LOCAL_BUF1_0@16] = CALL(INCAST_LOCAL_BUF0_0@12, attrs=["callee": "foo_PATH0_hiddenfunc_5"])
    aux_7@13 = ASSEMBLE(OUTCAST_LOCAL_BUF0@14, attrs=["toOffset": [0, 0]])
    out_4@15 = ASSEMBLE(OUTCAST_LOCAL_BUF1_0@16, attrs=["toOffset": [0, 0]])
@ir.function(type=ir.FunctionType.InCore)
def foo_PATH0_hiddenfunc_5(a_1@6: ir.Tensor, aux_5@8: ir.Tensor, out_2@9: ir.Tensor):
    INCAST_LOCAL_BUF0@7 = VIEW(a_1@6, attrs=["fromOffset": [0, 0], "toValidShape": [32, 16]])
    $0@3 = VEC_DUP()
    aux_1@8 = ASSEMBLE($0@3, attrs=["toOffset": [0, 0]])
    $1@4 = VEC_DUP()
    aux_5@8 = ASSEMBLE($1@4, attrs=["toOffset": [16, 0]])
    OUTCAST_LOCAL_BUF1@10 = ADD(INCAST_LOCAL_BUF0@7, aux_5@8)
    out_2@9 = ASSEMBLE(OUTCAST_LOCAL_BUF1@10, attrs=["toOffset": [0, 0]])
""" # noqa: E501


def _run_root_builder(func, *args):
    """Compile a kernel and run canonicalize + dce + create_root_functions, returning the program
    so callers can inspect hiddenfuncs and outcast versions."""
    compiled = pil.compile(func, *args, create_new_logical_tensor=True)
    builder = ir.IRBuilder()
    program = builder.create_program([compiled], "main", ir.Span.unknown())
    verifier = ir.IRVerifier.create_default()
    _ssa_verify(verifier, program, "original")
    program = ir.Pass.aggressive_dce()(ir.Pass.canonicalize()(program))
    _ssa_verify(verifier, program, "dce")
    program = ir.Pass.create_root_functions()(program)
    _ssa_verify(verifier, program, "root_functions")
    return program


def test_assemble_outcast_versions_share_raw_tensor():
    """MakeOutcasts preserves distinct Assemble versions on one RawTensor."""

    def foo(a, aux, out):
        pypto.set_vec_tile_shapes(16, 16)
        pypto.assemble(pypto.full([16, 16], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([16, 16], 2.0, pypto.DT_FP32), [16, 0], aux)
        out[:] = a + aux

    a = pypto.Tensor([32, 16], pypto.DT_FP32, name="a")
    aux = pypto.Tensor([32, 16], pypto.DT_FP32, name="aux")
    out = pypto.Tensor([32, 16], pypto.DT_FP32, name="out")
    program = _run_root_builder(foo, a, aux, out)

    check_snapshot(program, IR)
