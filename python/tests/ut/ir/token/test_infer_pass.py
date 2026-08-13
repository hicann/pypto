# -*- coding: utf-8 -*-
# ruff: noqa: E501
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for ir.Pass.infer_token_pass."""

import re

import pypto
from pypto import ir
from pypto.pil.compile_pipeline import compile_new_ir

from ..test_common import check_snapshot


def _run_infer_token_pass(func, *args):
    pipeline = [("infer_token_pass", ir.Pass.infer_token_pass())]
    return compile_new_ir(func, *args, pipeline=pipeline, create_new_logical_tensor=True)


def _tensor(name="x"):
    return pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name=name)


def _condition():
    return pypto.Tensor(shape=(-1,), dtype=pypto.DT_INT32, name="condition")


class _NormalizedSnapshot:
    def __init__(self, func):
        self._func = func
        self.name = func.name

    def __str__(self):
        return _normalize_token_names(str(self._func))


def _normalize_token_names(text):
    token_names = {}
    pattern = re.compile(r"(_[A-Za-z0-9]+_token_|__token_|_if_token_|_for_token_)\d+")

    def replace(match):
        name = match.group(0)
        if name not in token_names:
            token_names[name] = match.group(1) + str(len(token_names) + 1)
        return token_names[name]

    return re.sub(r"loop_idx_\d+", "loop_idx_N", pattern.sub(replace, text))


def _check_snapshot(func, golden):
    check_snapshot(_NormalizedSnapshot(func), _normalize_token_names(golden))


IR_0 = """
@ir.function
def foo(x@0: ir.Tensor):
    $0@1, $0_w, x_r = ADD(x@0, x@0)
    return x@0
"""


def test_repeated_input():
    def foo(x):
        pypto.add(x, x)

    func = _run_infer_token_pass(foo, _tensor())
    check_snapshot(func, IR_0)


IR_1 = """
@ir.function
def foo(x@0: ir.Tensor):
    [$1@2, $0@1], $1_w, $0_w, x_r = TOPK(x@0)
    return x@0
"""


def test_multiple_outputs():
    def foo(x):
        pypto.topk(x, 2)

    func = _run_infer_token_pass(foo, _tensor())
    check_snapshot(func, IR_1)


IR_2 = """
@ir.function
def foo(x@0: ir.Tensor):
    $0@1, $0_w, x_r = ADDS(x@0)
    $1@2, $1_w, x_r = SUBS(x@0)
    $2@3, $2_w, $0_r, $1_r = ADD($0@1, $1@2, tokens=[$0_w, $1_w])
    return x@0
"""


def test_multiple_reads():
    def foo(x):
        lhs = x + 1.0
        rhs = x - 1.0
        pypto.add(lhs, rhs)

    func = _run_infer_token_pass(foo, _tensor())
    check_snapshot(func, IR_2)


IR_3 = """
@ir.function
def foo(a@0: ir.Tensor, b@1: ir.Tensor):
    $0@2, $0_w, b_r = EXP(b@1)
    a_1@0, a_1_w, $0_r = ASSEMBLE($0@2, tokens=[$0_w], attrs=["toOffset": [0, 0]])
    $1@3, $1_w, a_1_r = EXP(a_1@0, tokens=[a_1_w])
    $2@4, $2_w, a_1_r = EXP(a_1@0, tokens=[a_1_w])
    return a_1@0, b@1
"""


def test_read_after_write():
    def foo(a, b):
        value = pypto.exp(b)
        pypto.assemble(value, [0, 0], a)
        pypto.exp(a)
        pypto.exp(a)

    func = _run_infer_token_pass(foo, _tensor("a"), _tensor("b"))
    check_snapshot(func, IR_3)


IR_4 = """
@ir.function
def foo(x@0: ir.Tensor):
    aux@1, aux_w = TENSOR_ALLOC()
    $0@2, $0_w, aux_r = ADDS(aux@1, tokens=[aux_w])
    $1@3, $1_w = VEC_DUP()
    aux_1@1, aux_1_w, $1_r = ASSEMBLE($1@3, tokens=[$1_w, aux_r], attrs=["toOffset": [0, 0]])
    $2@4, $2_w, $0_r, x_r = ADD($0@2, x@0, tokens=[$0_w])
    return x@0
"""


def test_write_after_read():
    def foo(x):
        pypto.set_vec_tile_shapes(16, 16)
        aux = pypto.tensor([32, 16], pypto.DT_FP32, name="aux")
        read = aux + 1.0
        src = pypto.full([16, 16], 2.0, pypto.DT_FP32)
        pypto.assemble(src, [0, 0], aux)
        pypto.add(read, x)

    func = _run_infer_token_pass(foo, _tensor())
    check_snapshot(func, IR_4)


IR_5 = """
@ir.function
def foo(a@0: ir.Tensor):
    $0@1, $0_w, a_r = EXP(a@0)
    $1@2, $1_w, a_r = EXP(a@0)
    a_1@0, a_1_w, $0_r = ASSEMBLE($0@1, tokens=[$0_w, a_r], attrs=["toOffset": [0, 0]])
    return a_1@0
"""


def test_read_read_write():
    def foo(a):
        value = pypto.exp(a)
        pypto.exp(a)
        pypto.assemble(value, [0, 0], a)

    func = _run_infer_token_pass(foo, _tensor("a"))
    check_snapshot(func, IR_5)


IR_6 = """
@ir.function
def foo(a@0: ir.Tensor, b@1: ir.Tensor, c@2: ir.Tensor):
    $0@3, $0_w, b_r = EXP(b@1)
    a_1@0, a_1_w, $0_r = ASSEMBLE($0@3, tokens=[$0_w], attrs=["toOffset": [0, 0]])
    $1@4, $1_w, c_r = EXP(c@2)
    a_3@0, a_3_w, $1_r = ASSEMBLE($1@4, tokens=[$1_w, a_1_w], attrs=["toOffset": [0, 0]])
    $2@5, $2_w, a_3_r = EXP(a_3@0, tokens=[a_3_w])
    return a_3@0, b@1, c@2
"""


def test_write_write_read():
    def foo(a, b, c):
        first = pypto.exp(b)
        pypto.assemble(first, [0, 0], a)
        second = pypto.exp(c)
        pypto.assemble(second, [0, 0], a)
        pypto.exp(a)

    func = _run_infer_token_pass(foo, _tensor("a"), _tensor("b"), _tensor("c"))
    check_snapshot(func, IR_6)


IR_7 = """
@ir.function
def foo(x@0: ir.Tensor, aux@1: ir.Tensor, out@2: ir.Tensor, condition@3: ir.Tensor):
    $0@4, $0_w = VEC_DUP()
    aux_1@1, aux_1_w, $0_r = ASSEMBLE($0@4, tokens=[$0_w], attrs=["toOffset": [0, 0]])
    if (0<RUNTIME_GetInputShapeDim(ARG_condition,0)):
        $1@5, $1_w, aux_1_r, x_r = ADD(aux_1@1, x@0, tokens=[aux_1_w])
        result, result_w_if = ir.yield_($1@5, $1_w)
    else:
        $2@6, $2_w, aux_1_r_0, x_r_0 = SUB(aux_1@1, x@0, tokens=[aux_1_w])
        result, result_w_if = ir.yield_($2@6, $2_w)
    out_1@2, out_1_w, result_r = ASSEMBLE(result@5, tokens=[result_w_if], attrs=["toOffset": [0, 0]])
    return x@0, aux_1@1, out_1@2, condition@3
"""


def test_if_input():
    def foo(x, aux, out, condition):
        pypto.set_vec_tile_shapes(16, 16)
        src = pypto.full([16, 16], 1.0, pypto.DT_FP32)
        pypto.assemble(src, [0, 0], aux)
        if condition.shape[0] > 0:
            result = pypto.add(aux, x)
        else:
            result = pypto.sub(aux, x)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"), _condition())
    _check_snapshot(func, IR_7)


IR_8 = """
@ir.function
def foo(x@0: ir.Tensor, aux@1: ir.Tensor, out@2: ir.Tensor, condition@3: ir.Tensor):
    if (0<RUNTIME_GetInputShapeDim(ARG_condition,0)):
        $0@4, $0_w = VEC_DUP()
        aux_1@1, aux_1_w, $0_r = ASSEMBLE($0@4, tokens=[$0_w], attrs=["toOffset": [0, 0]])
        src_else, src_then, src_else_w_if, src_then_w_if, aux_1_w_if = ir.yield_(None, $0@4, None, $0_w, aux_1_w)
    else:
        $1@5, $1_w = VEC_DUP()
        aux_3@1, aux_3_w, $1_r = ASSEMBLE($1@5, tokens=[$1_w], attrs=["toOffset": [0, 0]])
        src_else, src_then, src_else_w_if, src_then_w_if, aux_1_w_if = ir.yield_($1@5, None, $1_w, None, aux_3_w)
    $4@6, $4_w, aux_r, x_r = ADD(aux@1, x@0, tokens=[aux_1_w_if])
    out_1@2, out_1_w, $4_r = ASSEMBLE($4@6, tokens=[$4_w], attrs=["toOffset": [0, 0]])
    return x@0, aux@1, out_1@2, condition@3
"""


def test_if_output():
    def foo(x, aux, out, condition):
        pypto.set_vec_tile_shapes(16, 16)
        if condition.shape[0] > 0:
            src_then = pypto.full([16, 16], 1.0, pypto.DT_FP32)
            pypto.assemble(src_then, [0, 0], aux)
        else:
            src_else = pypto.full([16, 16], 2.0, pypto.DT_FP32)
            pypto.assemble(src_else, [0, 0], aux)
        result = pypto.add(aux, x)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"), _condition())
    _check_snapshot(func, IR_8)


IR_9 = """
@ir.function
def foo(x@0: ir.Tensor, aux@1: ir.Tensor, out@2: ir.Tensor, condition@3: ir.Tensor):
    $0@4, $0_w = VEC_DUP()
    aux_1@1, aux_1_w, $0_r = ASSEMBLE($0@4, tokens=[$0_w], attrs=["toOffset": [0, 0]])
    if (0<RUNTIME_GetInputShapeDim(ARG_condition,0)):
        $1@5, $1_w = VEC_DUP()
        aux_3@1, aux_3_w, $1_r = ASSEMBLE($1@5, tokens=[$1_w, aux_1_w], attrs=["toOffset": [0, 0]])
        inside, inside_w_if, aux_1_w_if = ir.yield_($1@5, $1_w, aux_3_w)
    else:
        inside, inside_w_if, aux_1_w_if = ir.yield_(None, None, aux_1_w)
    $3@6, $3_w, aux_1_r, x_r = ADD(aux_1@1, x@0, tokens=[aux_1_w_if])
    out_1@2, out_1_w, $3_r = ASSEMBLE($3@6, tokens=[$3_w], attrs=["toOffset": [0, 0]])
    return x@0, aux_1@1, out_1@2, condition@3
"""


def test_if_passthrough():
    def foo(x, aux, out, condition):
        pypto.set_vec_tile_shapes(16, 16)
        before = pypto.full([16, 16], 1.0, pypto.DT_FP32)
        pypto.assemble(before, [0, 0], aux)
        if condition.shape[0] > 0:
            inside = pypto.full([16, 16], 2.0, pypto.DT_FP32)
            pypto.assemble(inside, [0, 0], aux)
        result = pypto.add(aux, x)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"), _condition())
    _check_snapshot(func, IR_9)


IR_10 = """
@ir.function
def foo(x@0: ir.Tensor, aux@1: ir.Tensor, out@2: ir.Tensor, condition@3: ir.Tensor):
    if (0<RUNTIME_GetInputShapeDim(ARG_condition,0)):
        $0@4, $0_w, aux_r = ADDS(aux@1)
        $1@5, $1_w, aux_r = SUBS(aux@1)
        $2@6, $2_w, $0_r, $1_r = ADD($0@4, $1@5, tokens=[$0_w, $1_w])
        read0, read1, result, read0_w_if, read1_w_if, result_w_if, aux_r_if = ir.yield_($0@4, $1@5, $2@6, $0_w, $1_w, $2_w, aux_r)
    else:
        $3@7, $3_w, x_r = ADDS(x@0)
        read0, read1, result, read0_w_if, read1_w_if, result_w_if, aux_r_if = ir.yield_(None, None, $3@7, None, None, $3_w, None)
    $7@8, $7_w = VEC_DUP()
    aux_1@1, aux_1_w, $7_r = ASSEMBLE($7@8, tokens=[$7_w, aux_r_if], attrs=["toOffset": [0, 0]])
    out_1@2, out_1_w, result_r = ASSEMBLE(result@6, tokens=[result_w_if], attrs=["toOffset": [0, 0]])
    return x@0, aux_1@1, out_1@2, condition@3
"""


def test_if_multiple_reads():
    def foo(x, aux, out, condition):
        pypto.set_vec_tile_shapes(16, 16)
        if condition.shape[0] > 0:
            read0 = aux + 1.0
            read1 = aux - 1.0
            result = pypto.add(read0, read1)
        else:
            result = x + 1.0
        src = pypto.full([16, 16], 2.0, pypto.DT_FP32)
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"), _condition())
    _check_snapshot(func, IR_10)


IR_11 = """
@ir.function
def foo(x@0: ir.Tensor, aux@1: ir.Tensor, out@2: ir.Tensor):
    $0@3, $0_w = VEC_DUP()
    aux_1@1, aux_1_w, $0_r = ASSEMBLE($0@3, tokens=[$0_w], attrs=["toOffset": [0, 0]])
    for loop_idx_N, (_, result, result_0_w_for_iter) in ir.range(0, 2, 1, init_values=(None, None, None), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        $1@4, $1_w, aux_1_r, x_r = ADD(aux_1@1, x@0, tokens=[aux_1_w])
        __0, result_0, result_0_w_for = continue loop_idx_N, $1@4, $1_w
    out_1@2, out_1_w, result_0_r = ASSEMBLE(result_0@4, tokens=[result_0_w_for], attrs=["toOffset": [0, 0]])
    return x@0, aux_1@1, out_1@2
"""


def test_for_input():
    def foo(x, aux, out):
        pypto.set_vec_tile_shapes(16, 16)
        src = pypto.full([16, 16], 1.0, pypto.DT_FP32)
        pypto.assemble(src, [0, 0], aux)
        for _ in pypto.loop(2):
            result = pypto.add(aux, x)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"))
    _check_snapshot(func, IR_11)


IR_12 = """
@ir.function
def foo(x@0: ir.Tensor, aux@1: ir.Tensor, out@2: ir.Tensor):
    for loop_idx_N, (i, src, src_0_w_for_iter, aux_1_w_for_iter) in ir.range(0, 2, 1, init_values=(None, None, None, None), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        $0@3, $0_w = VEC_DUP()
        aux_1@1, aux_1_w, $0_r = ASSEMBLE($0@3, tokens=[$0_w], attrs=["toOffset": [(loop_idx_N*8), 0]])
        i_0, src_0, src_0_w_for, aux_1_w_for = continue loop_idx_N, $0@3, $0_w, aux_1_w
    $2@4, $2_w, x_r, aux_1_r = ADD(x@0, aux_1@1, tokens=[aux_1_w_for])
    out_1@2, out_1_w, $2_r = ASSEMBLE($2@4, tokens=[$2_w], attrs=["toOffset": [0, 0]])
    return x@0, aux_1@1, out_1@2
"""


def test_for_output():
    def foo(x, aux, out):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            src = pypto.full([16, 16], 1.0, pypto.DT_FP32)
            pypto.assemble(src, [i * 8, 0], aux)
        result = pypto.add(x, aux)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"))
    _check_snapshot(func, IR_12)


IR_13 = """
@ir.function
def foo(x@0: ir.Tensor, aux@1: ir.Tensor, out@2: ir.Tensor, condition@3: ir.Tensor):
    if (0<RUNTIME_GetInputShapeDim(ARG_condition,0)):
        $0@4, $0_w = VEC_DUP()
        aux_1@1, aux_1_w, $0_r = ASSEMBLE($0@4, tokens=[$0_w], attrs=["toOffset": [0, 0]])
        inside, inside_w_if, aux_1_w_if = ir.yield_($0@4, $0_w, aux_1_w)
    else:
        inside, inside_w_if, aux_1_w_if = ir.yield_(None, None, None)
    $2@5, $2_w, aux_r, x_r = ADD(aux@1, x@0, tokens=[aux_1_w_if])
    out_1@2, out_1_w, $2_r = ASSEMBLE($2@5, tokens=[$2_w], attrs=["toOffset": [0, 0]])
    return x@0, aux@1, out_1@2, condition@3
"""


def test_if_none():
    def foo(x, aux, out, condition):
        pypto.set_vec_tile_shapes(16, 16)
        if condition.shape[0] > 0:
            inside = pypto.full([16, 16], 1.0, pypto.DT_FP32)
            pypto.assemble(inside, [0, 0], aux)
        result = pypto.add(aux, x)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"), _condition())
    _check_snapshot(func, IR_13)


IR_14 = """
@ir.function
def foo(x@0: ir.Tensor, aux@1: ir.Tensor, out@2: ir.Tensor):
    for loop_idx_N, (_, read, read_0_w_for_iter, aux_r_for_iter) in ir.range(0, 2, 1, init_values=(None, None, None, None), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        $0@3, $0_w, aux_r, x_r = ADD(aux@1, x@0)
        __0, read_0, read_0_w_for, aux_r_for = continue loop_idx_N, $0@3, $0_w, aux_r
    $2@4, $2_w = VEC_DUP()
    aux_1@1, aux_1_w, $2_r = ASSEMBLE($2@4, tokens=[$2_w, aux_r_for], attrs=["toOffset": [0, 0]])
    out_1@2, out_1_w, read_0_r = ASSEMBLE(read_0@3, tokens=[read_0_w_for], attrs=["toOffset": [0, 0]])
    return x@0, aux_1@1, out_1@2
"""


def test_for_read_output():
    def foo(x, aux, out):
        pypto.set_vec_tile_shapes(16, 16)
        for _ in pypto.loop(2):
            read = pypto.add(aux, x)
        src = pypto.full([16, 16], 1.0, pypto.DT_FP32)
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(read, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"))
    _check_snapshot(func, IR_14)
