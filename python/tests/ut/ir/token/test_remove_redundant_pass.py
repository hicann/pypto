# -*- coding: utf-8 -*-
# ruff: noqa: E501
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for removing redundant token dependencies after MergeStmtsIntoIf."""

import pypto
from pypto import ir
from pypto.pil.compile_pipeline import compile_new_ir

from ..test_common import check_snapshot


class _Snapshot:
    def __init__(self, name, text):
        self.name = name
        self.text = text

    def __str__(self):
        return self.text


def _run_passes(func, *args):
    before_remove = []

    def capture_before_remove(program):
        compiled_func = program.functions[func.__name__]
        before_remove.append(_Snapshot(compiled_func.name, str(compiled_func)))
        return program

    dce = ir.Pass.aggressive_dce()
    canonicalize = ir.Pass.canonicalize()
    merge_stmts = ir.Pass.merge_stmts_into_if()
    pipeline = [
        ("infer_token_pass", ir.Pass.infer_token_pass()),
        ("first_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("second_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("canonicalize(merge_stmts)", lambda p:canonicalize(merge_stmts(p))),
        ("capture_before_remove_redundant_token_pass", capture_before_remove),
        ("remove_redundant_token_pass", ir.Pass.remove_redundant_token_pass()),
    ]
    after_remove = compile_new_ir(func, *args, pipeline=pipeline, create_new_logical_tensor=True)
    return before_remove[0], after_remove


def _tensor(shape, name):
    return pypto.Tensor(shape=shape, dtype=pypto.DT_FP32, name=name)




BEFORE_IR_0 = """
function foo incast(logical_tensor %x, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w, token %x_r = ADDS(%x) token();
    logical_tensor %out_1, token %out_1_w, token %$0_r = ASSEMBLE(%$0) token(%$0_w);
    return %x, %out_1;
}
"""


IR_0 = """
function foo incast(logical_tensor %x, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0 = ADDS(%x) token();
    logical_tensor %out_1 = ASSEMBLE(%$0) token();
    return %x, %out_1;
}
"""


def test_remove_ssa_dependency():
    def foo(x, out):
        value = x + 1.0
        pypto.assemble(value, [0, 0], out)

    before_remove, after_remove = _run_passes(foo, _tensor((16, 16), "x"), _tensor((16, 16), "out"))
    check_snapshot(before_remove, BEFORE_IR_0)
    check_snapshot(after_remove, IR_0)


BEFORE_IR_1 = """
function foo incast(logical_tensor %src, logical_tensor %aux, logical_tensor %result) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_1, token %aux_1_w, token %src_r = ASSEMBLE(%src) token();
    logical_tensor %aux_3, token %aux_3_w, token %src_r = ASSEMBLE(%src) token(%aux_1_w);
    logical_tensor %$0, token %$0_w, token %aux_3_r = ADDS(%aux_3) token(%aux_3_w);
    return %src, %aux_3, %$0;
}
"""


IR_1 = """
function foo incast(logical_tensor %src, logical_tensor %aux, logical_tensor %result) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_1, token %aux_1_w = ASSEMBLE(%src) token();
    logical_tensor %aux_3 = ASSEMBLE(%src) token();
    logical_tensor %$0 = ADDS(%aux_3) token(%aux_1_w);
    return %src, %aux_3, %$0;
}
"""


def test_forward_disjoint_write_dependency():
    def foo(src, aux, result):
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(src, [16, 0], aux)
        result = aux + 1.0  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_1)
    check_snapshot(after_remove, IR_1)


BEFORE_IR_2 = """
function foo incast(logical_tensor %src, logical_tensor %aux, logical_tensor %result) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_1, token %aux_1_w, token %src_r = ASSEMBLE(%src) token();
    logical_tensor %aux_3, token %aux_3_w, token %src_r = ASSEMBLE(%src) token(%aux_1_w);
    logical_tensor %$0, token %$0_w, token %aux_3_r = ADDS(%aux_3) token(%aux_3_w);
    return %src, %aux_3, %$0;
}
"""


IR_2 = """
function foo incast(logical_tensor %src, logical_tensor %aux, logical_tensor %result) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_1, token %aux_1_w = ASSEMBLE(%src) token();
    logical_tensor %aux_3 = ASSEMBLE(%src) token(%aux_1_w);
    logical_tensor %$0 = ADDS(%aux_3) token();
    return %src, %aux_3, %$0;
}
"""


def test_keep_overlapping_write_dependency():
    def foo(src, aux, result):
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(src, [4, 0], aux)
        result = aux + 1.0  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_2)
    check_snapshot(after_remove, IR_2)


BEFORE_IR_3 = """
function foo incast(logical_tensor %src, logical_tensor %aux, logical_tensor %shape) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_1, token %aux_1_w, token %src_r = ASSEMBLE(%src) token();
    logical_tensor %aux_3, token %aux_3_w, token %src_r = ASSEMBLE(%src) token(%aux_1_w);
    return %src, %aux_3, %shape;
}
"""


IR_3 = """
function foo incast(logical_tensor %src, logical_tensor %aux, logical_tensor %shape) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_1, token %aux_1_w = ASSEMBLE(%src) token();
    logical_tensor %aux_3 = ASSEMBLE(%src) token(%aux_1_w);
    return %src, %aux_3, %shape;
}
"""


def test_keep_dynamic_write_dependency():
    def foo(src, aux, shape):
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(src, [shape.shape[0], 0], aux)

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((-1,), "shape"),
    )
    check_snapshot(before_remove, BEFORE_IR_3)
    check_snapshot(after_remove, IR_3)


BEFORE_IR_4 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w = VEC_DUP() token();
    logical_tensor %aux_1, token %aux_1_w, token %$0_r = ASSEMBLE(%$0) token(%$0_w);
    logical_tensor %$1, token %$1_w = VEC_DUP() token();
    logical_tensor %aux_3, token %aux_3_w, token %$1_r = ASSEMBLE(%$1) token(%$1_w, %aux_1_w);
    logical_tensor %$2, token %$2_w, token %a_r, token %aux_3_r = ADD(%a, %aux_3) token(%aux_3_w);
    logical_tensor %out_1, token %out_1_w, token %$2_r = ASSEMBLE(%$2) token(%$2_w);
    return %a, %aux_3, %out_1;
}
"""


IR_4 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0 = VEC_DUP() token();
    logical_tensor %aux_1, token %aux_1_w = ASSEMBLE(%$0) token();
    logical_tensor %$1 = VEC_DUP() token();
    logical_tensor %aux_3 = ASSEMBLE(%$1) token();
    logical_tensor %$2 = ADD(%a, %aux_3) token(%aux_1_w);
    logical_tensor %out_1 = ASSEMBLE(%$2) token();
    return %a, %aux_3, %out_1;
}
"""


def test_assemble_scenario_1_parallel_non_overlapping_writes():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([64, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([64, 128], 2.0, pypto.DT_FP32), [64, 0], aux)
        pypto.assemble(a + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_4)
    check_snapshot(after_remove, IR_4)


BEFORE_IR_5 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w = VEC_DUP() token();
    logical_tensor %aux_1, token %aux_1_w, token %$0_r = ASSEMBLE(%$0) token(%$0_w);
    logical_tensor %$1, token %$1_w, token %aux_1_r = ADDS(%aux_1) token(%aux_1_w);
    logical_tensor %$2, token %$2_w = VEC_DUP() token();
    logical_tensor %aux_3, token %aux_3_w, token %$2_r = ASSEMBLE(%$2) token(%$2_w, %aux_1_r);
    logical_tensor %$3, token %$3_w, token %$1_r, token %aux_3_r = ADD(%$1, %aux_3) token(%$1_w, %aux_3_w);
    logical_tensor %out_1, token %out_1_w, token %$3_r = ASSEMBLE(%$3) token(%$3_w);
    return %a, %aux_3, %out_1;
}
"""


IR_5 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0 = VEC_DUP() token();
    logical_tensor %aux_1 = ASSEMBLE(%$0) token();
    logical_tensor %$1, token %aux_1_r = ADDS(%aux_1) token();
    logical_tensor %$2 = VEC_DUP() token();
    logical_tensor %aux_3 = ASSEMBLE(%$2) token(%aux_1_r);
    logical_tensor %$3 = ADD(%$1, %aux_3) token();
    logical_tensor %out_1 = ASSEMBLE(%$3) token();
    return %a, %aux_3, %out_1;
}
"""


def test_assemble_scenario_2_war_between_non_overlapping_writes():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([64, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        value = aux + 1.0
        pypto.assemble(pypto.full([64, 128], 2.0, pypto.DT_FP32), [64, 0], aux)
        pypto.assemble(value + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_5)
    check_snapshot(after_remove, IR_5)


BEFORE_IR_6 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w = VEC_DUP() token();
    logical_tensor %aux_1, token %aux_1_w, token %$0_r = ASSEMBLE(%$0) token(%$0_w);
    logical_tensor %$1, token %$1_w = VEC_DUP() token();
    logical_tensor %aux_3, token %aux_3_w, token %$1_r = ASSEMBLE(%$1) token(%$1_w, %aux_1_w);
    logical_tensor %$2, token %$2_w, token %a_r, token %aux_3_r = ADD(%a, %aux_3) token(%aux_3_w);
    logical_tensor %out_1, token %out_1_w, token %$2_r = ASSEMBLE(%$2) token(%$2_w);
    return %a, %aux_3, %out_1;
}
"""


IR_6 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0 = VEC_DUP() token();
    logical_tensor %aux_1, token %aux_1_w = ASSEMBLE(%$0) token();
    logical_tensor %$1 = VEC_DUP() token();
    logical_tensor %aux_3 = ASSEMBLE(%$1) token(%aux_1_w);
    logical_tensor %$2 = ADD(%a, %aux_3) token();
    logical_tensor %out_1 = ASSEMBLE(%$2) token();
    return %a, %aux_3, %out_1;
}
"""


def test_assemble_scenario_3_overlapping_waw():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([129, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([64, 128], 2.0, pypto.DT_FP32), [64, 0], aux)
        pypto.assemble(a + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_6)
    check_snapshot(after_remove, IR_6)


BEFORE_IR_7 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w = VEC_DUP() token();
    logical_tensor %aux_1, token %aux_1_w, token %$0_r = ASSEMBLE(%$0) token(%$0_w);
    logical_tensor %$1, token %$1_w, token %aux_1_r = ADDS(%aux_1) token(%aux_1_w);
    logical_tensor %$2, token %$2_w = VEC_DUP() token();
    logical_tensor %aux_3, token %aux_3_w, token %$2_r = ASSEMBLE(%$2) token(%$2_w, %aux_1_r);
    logical_tensor %$3, token %$3_w, token %aux_3_r = ADDS(%aux_3) token(%aux_3_w);
    logical_tensor %$4, token %$4_w = VEC_DUP() token();
    logical_tensor %aux_5, token %aux_5_w, token %$4_r = ASSEMBLE(%$4) token(%$4_w, %aux_3_r);
    logical_tensor %$5, token %$5_w, token %$1_r, token %$3_r = ADD(%$1, %$3) token(%$1_w, %$3_w);
    logical_tensor %$6, token %$6_w, token %$5_r, token %aux_5_r = ADD(%$5, %aux_5) token(%$5_w, %aux_5_w);
    logical_tensor %out_1, token %out_1_w, token %$6_r = ASSEMBLE(%$6) token(%$6_w);
    return %a, %aux_5, %out_1;
}
"""


IR_7 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0 = VEC_DUP() token();
    logical_tensor %aux_1 = ASSEMBLE(%$0) token();
    logical_tensor %$1, token %aux_1_r = ADDS(%aux_1) token();
    logical_tensor %$2 = VEC_DUP() token();
    logical_tensor %aux_3 = ASSEMBLE(%$2) token(%aux_1_r);
    logical_tensor %$3, token %aux_3_r = ADDS(%aux_3) token();
    logical_tensor %$4 = VEC_DUP() token();
    logical_tensor %aux_5 = ASSEMBLE(%$4) token(%aux_3_r);
    logical_tensor %$5 = ADD(%$1, %$3) token();
    logical_tensor %$6 = ADD(%$5, %aux_5) token();
    logical_tensor %out_1 = ASSEMBLE(%$6) token();
    return %a, %aux_5, %out_1;
}
"""


def test_assemble_scenario_4_all_versions_consumed():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([32, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        value1 = aux + 1.0
        pypto.assemble(pypto.full([32, 128], 2.0, pypto.DT_FP32), [96, 0], aux)
        value2 = aux + 2.0
        pypto.assemble(pypto.full([64, 128], 3.0, pypto.DT_FP32), [16, 0], aux)
        pypto.assemble(value1 + value2 + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_7)
    check_snapshot(after_remove, IR_7)


BEFORE_IR_8 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w = VEC_DUP() token();
    logical_tensor %aux_1, token %aux_1_w, token %$0_r = ASSEMBLE(%$0) token(%$0_w);
    logical_tensor %$1, token %$1_w = VEC_DUP() token();
    logical_tensor %aux_3, token %aux_3_w, token %$1_r = ASSEMBLE(%$1) token(%$1_w, %aux_1_w);
    logical_tensor %$2, token %$2_w = VEC_DUP() token();
    logical_tensor %aux_5, token %aux_5_w, token %$2_r = ASSEMBLE(%$2) token(%$2_w, %aux_3_w);
    logical_tensor %$3, token %$3_w, token %a_r, token %aux_5_r = ADD(%a, %aux_5) token(%aux_5_w);
    logical_tensor %out_1, token %out_1_w, token %$3_r = ASSEMBLE(%$3) token(%$3_w);
    return %a, %aux_5, %out_1;
}
"""


IR_8 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0 = VEC_DUP() token();
    logical_tensor %aux_1, token %aux_1_w = ASSEMBLE(%$0) token();
    logical_tensor %$1 = VEC_DUP() token();
    logical_tensor %aux_3, token %aux_3_w = ASSEMBLE(%$1) token();
    logical_tensor %$2 = VEC_DUP() token();
    logical_tensor %aux_5 = ASSEMBLE(%$2) token(%aux_1_w);
    logical_tensor %$3 = ADD(%a, %aux_5) token(%aux_3_w);
    logical_tensor %out_1 = ASSEMBLE(%$3) token();
    return %a, %aux_5, %out_1;
}
"""


def test_assemble_scenario_5_no_intermediate_version_consumed():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([32, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([32, 128], 2.0, pypto.DT_FP32), [96, 0], aux)
        pypto.assemble(pypto.full([64, 128], 3.0, pypto.DT_FP32), [16, 0], aux)
        pypto.assemble(a + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_8)
    check_snapshot(after_remove, IR_8)


BEFORE_IR_9 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w = VEC_DUP() token();
    logical_tensor %aux_1, token %aux_1_w, token %$0_r = ASSEMBLE(%$0) token(%$0_w);
    logical_tensor %$1, token %$1_w, token %aux_1_r = ADDS(%aux_1) token(%aux_1_w);
    logical_tensor %$2, token %$2_w = VEC_DUP() token();
    logical_tensor %aux_3, token %aux_3_w, token %$2_r = ASSEMBLE(%$2) token(%$2_w, %aux_1_r);
    logical_tensor %$3, token %$3_w = VEC_DUP() token();
    logical_tensor %aux_5, token %aux_5_w, token %$3_r = ASSEMBLE(%$3) token(%$3_w, %aux_3_w);
    logical_tensor %$4, token %$4_w, token %$1_r, token %aux_5_r = ADD(%$1, %aux_5) token(%$1_w, %aux_5_w);
    logical_tensor %out_1, token %out_1_w, token %$4_r = ASSEMBLE(%$4) token(%$4_w);
    return %a, %aux_5, %out_1;
}
"""


IR_9 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0 = VEC_DUP() token();
    logical_tensor %aux_1 = ASSEMBLE(%$0) token();
    logical_tensor %$1, token %aux_1_r = ADDS(%aux_1) token();
    logical_tensor %$2 = VEC_DUP() token();
    logical_tensor %aux_3, token %aux_3_w = ASSEMBLE(%$2) token(%aux_1_r);
    logical_tensor %$3 = VEC_DUP() token();
    logical_tensor %aux_5 = ASSEMBLE(%$3) token(%aux_1_r);
    logical_tensor %$4 = ADD(%$1, %aux_5) token(%aux_3_w);
    logical_tensor %out_1 = ASSEMBLE(%$4) token();
    return %a, %aux_5, %out_1;
}
"""


def test_assemble_scenario_6_middle_version_not_consumed():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([32, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        value = aux + 1.0
        pypto.assemble(pypto.full([32, 128], 2.0, pypto.DT_FP32), [96, 0], aux)
        pypto.assemble(pypto.full([64, 128], 3.0, pypto.DT_FP32), [16, 0], aux)
        pypto.assemble(value + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_9)
    check_snapshot(after_remove, IR_9)


BEFORE_IR_10 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w = VEC_DUP() token();
    logical_tensor %aux_1, token %aux_1_w, token %$0_r = ASSEMBLE(%$0) token(%$0_w);
    logical_tensor %$1, token %$1_w = VEC_DUP() token();
    logical_tensor %aux_3, token %aux_3_w, token %$1_r = ASSEMBLE(%$1) token(%$1_w, %aux_1_w);
    logical_tensor %$2, token %$2_w, token %aux_3_r = ADDS(%aux_3) token(%aux_3_w);
    logical_tensor %$3, token %$3_w = VEC_DUP() token();
    logical_tensor %aux_5, token %aux_5_w, token %$3_r = ASSEMBLE(%$3) token(%$3_w, %aux_3_r);
    logical_tensor %$4, token %$4_w, token %$2_r, token %aux_5_r = ADD(%$2, %aux_5) token(%$2_w, %aux_5_w);
    logical_tensor %out_1, token %out_1_w, token %$4_r = ASSEMBLE(%$4) token(%$4_w);
    return %a, %aux_5, %out_1;
}
"""


IR_10 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0 = VEC_DUP() token();
    logical_tensor %aux_1, token %aux_1_w = ASSEMBLE(%$0) token();
    logical_tensor %$1 = VEC_DUP() token();
    logical_tensor %aux_3 = ASSEMBLE(%$1) token();
    logical_tensor %$2, token %aux_3_r = ADDS(%aux_3) token(%aux_1_w);
    logical_tensor %$3 = VEC_DUP() token();
    logical_tensor %aux_5 = ASSEMBLE(%$3) token(%aux_3_r);
    logical_tensor %$4 = ADD(%$2, %aux_5) token();
    logical_tensor %out_1 = ASSEMBLE(%$4) token();
    return %a, %aux_5, %out_1;
}
"""


def test_assemble_scenario_7_first_version_not_consumed():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([32, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([32, 128], 2.0, pypto.DT_FP32), [96, 0], aux)
        value = aux + 1.0
        pypto.assemble(pypto.full([64, 128], 3.0, pypto.DT_FP32), [16, 0], aux)
        pypto.assemble(value + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_10)
    check_snapshot(after_remove, IR_10)


BEFORE_IR_11 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w, token %a_r, token %aux_r = ADD(%a, %aux) token();
    logical_tensor %$1, token %$1_w = VEC_DUP() token();
    logical_tensor %a_1, token %a_1_w, token %$1_r = ASSEMBLE(%$1) token(%$1_w, %a_r);
    logical_tensor %$2, token %$2_w = VEC_DUP() token();
    logical_tensor %aux_1, token %aux_1_w, token %$2_r = ASSEMBLE(%$2) token(%$2_w, %aux_r);
    logical_tensor %out_1, token %out_1_w, token %$0_r = ASSEMBLE(%$0) token(%$0_w);
    return %a_1, %aux_1, %out_1;
}
"""


IR_11 = """
function foo incast(logical_tensor %a, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %a_r, token %aux_r = ADD(%a, %aux) token();
    logical_tensor %$1 = VEC_DUP() token();
    logical_tensor %a_1 = ASSEMBLE(%$1) token(%a_r);
    logical_tensor %$2 = VEC_DUP() token();
    logical_tensor %aux_1 = ASSEMBLE(%$2) token(%aux_r);
    logical_tensor %out_1 = ASSEMBLE(%$0) token();
    return %a_1, %aux_1, %out_1;
}
"""


def test_assemble_scenario_8_separate_writes_after_shared_read():
    def foo(a, aux, out):
        value = a + aux
        pypto.assemble(pypto.full([64, 128], 1.0, pypto.DT_FP32), [0, 0], a)
        pypto.assemble(pypto.full([64, 128], 2.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(value, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_11)
    check_snapshot(after_remove, IR_11)


BEFORE_IR_12 = """
function foo incast(logical_tensor %src, logical_tensor %aux, logical_tensor %result) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_1, token %aux_1_w, token %src_r = ASSEMBLE(%src) token();
    logical_tensor %aux_3, token %aux_3_w, token %src_r = ASSEMBLE(%src) token(%aux_1_w);
    logical_tensor %aux_5, token %aux_5_w, token %src_r = ASSEMBLE(%src) token(%aux_3_w);
    logical_tensor %$0, token %$0_w, token %aux_5_r = ADDS(%aux_5) token(%aux_5_w);
    return %src, %aux_5, %$0;
}
"""


IR_12 = """
function foo incast(logical_tensor %src, logical_tensor %aux, logical_tensor %result) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_1, token %aux_1_w = ASSEMBLE(%src) token();
    logical_tensor %aux_3, token %aux_3_w = ASSEMBLE(%src) token();
    logical_tensor %aux_5 = ASSEMBLE(%src) token(%aux_1_w);
    logical_tensor %$0 = ADDS(%aux_5) token(%aux_3_w);
    return %src, %aux_5, %$0;
}
"""


def test_rewire_four_op_token_chain():
    def foo(src, aux, result):
        # A: write aux[0:8].
        pypto.assemble(src, [0, 0], aux)
        # B: write aux[16:24].
        pypto.assemble(src, [16, 0], aux)
        # C: write aux[0:8].
        pypto.assemble(src, [0, 0], aux)
        # D: read aux.
        result = aux + 1.0  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_12)
    check_snapshot(after_remove, IR_12)


BEFORE_IR_13 = """
function foo incast(logical_tensor %src_a, logical_tensor %src_b, logical_tensor %src_c, logical_tensor %aux, logical_tensor %result) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_1, token %aux_1_w, token %src_a_r = ASSEMBLE(%src_a) token();
    logical_tensor %aux_3, token %aux_3_w, token %src_b_r = ASSEMBLE(%src_b) token(%aux_1_w);
    logical_tensor %aux_5, token %aux_5_w, token %src_c_r = ASSEMBLE(%src_c) token(%aux_3_w);
    logical_tensor %$0, token %$0_w, token %aux_5_r = ADDS(%aux_5) token(%aux_5_w);
    return %src_a, %src_b, %src_c, %aux_5, %$0;
}
"""


IR_13 = """
function foo incast(logical_tensor %src_a, logical_tensor %src_b, logical_tensor %src_c, logical_tensor %aux, logical_tensor %result) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_1, token %aux_1_w = ASSEMBLE(%src_a) token();
    logical_tensor %aux_3, token %aux_3_w = ASSEMBLE(%src_b) token(%aux_1_w);
    logical_tensor %aux_5 = ASSEMBLE(%src_c) token(%aux_1_w);
    logical_tensor %$0 = ADDS(%aux_5) token(%aux_3_w);
    return %src_a, %src_b, %src_c, %aux_5, %$0;
}
"""


def test_rewire_middle_edge_to_diamond():
    def foo(src_a, src_b, src_c, aux, result):
        # A overlaps both B and C; B and C are disjoint.
        pypto.assemble(src_a, [0, 0], aux)   # A: write aux[0:32].
        pypto.assemble(src_b, [0, 0], aux)   # B: write aux[0:8].
        pypto.assemble(src_c, [24, 0], aux)  # C: write aux[24:32].
        result = aux + 1.0  # D: read the whole tensor.  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((32, 16), "src_a"),
        _tensor((8, 16), "src_b"),
        _tensor((8, 16), "src_c"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_13)
    check_snapshot(after_remove, IR_13)


BEFORE_IR_14 = """
function foo incast(logical_tensor %src, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_1, token %aux_1_w, token %src_r = ASSEMBLE(%src) token();
    logical_tensor %aux_3, token %aux_3_w, token %src_r = ASSEMBLE(%src) token(%aux_1_w);
    logical_tensor %value_0, token %value_0_w_for = for %loop_idx_21 inrange 0, 2, 1 iter {none %value = %None;token %value_0_w_for_iter = %None;} #parallel(false) #submit_before_loop(false) #_loop_conds(Unsupported) #_config_scope(Unsupported) #unroll_times(1) {
        logical_tensor %$0, token %$0_w, token %aux_3_r = ADDS(%aux_3) token(%aux_3_w);
        continue %$0, %$0_w;
    }
    logical_tensor %out_1, token %out_1_w, token %value_0_r = ASSEMBLE(%value_0) token(%value_0_w_for);
    return %src, %aux_3, %out_1;
}
"""


IR_14 = """
function foo incast(logical_tensor %src, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_1, token %aux_1_w = ASSEMBLE(%src) token();
    logical_tensor %aux_3, token %aux_3_w = ASSEMBLE(%src) token(%aux_1_w);
    logical_tensor %value_0, token %value_0_w_for = for %loop_idx_21 inrange 0, 2, 1 iter {none %value = %None;token %value_0_w_for_iter = %None;} #parallel(false) #submit_before_loop(false) #_loop_conds(Unsupported) #_config_scope(Unsupported) #unroll_times(1) {
        logical_tensor %$0, token %$0_w = ADDS(%aux_3) token(%aux_3_w);
        continue %$0, %$0_w;
    }
    logical_tensor %out_1 = ASSEMBLE(%value_0) token(%value_0_w_for);
    return %src, %aux_3, %out_1;
}
"""


def test_keep_dependency_when_result_is_read_in_loop():
    def foo(src, aux, out):
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(src, [16, 0], aux)
        for _ in pypto.loop(2):
            value = aux + 1.0
        pypto.assemble(value, [0, 0], out)

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "out"),
    )
    check_snapshot(before_remove, BEFORE_IR_14)
    check_snapshot(after_remove, IR_14)


BEFORE_IR_15 = """
function foo incast(logical_tensor %src, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_7, token %aux_7_w_for = for %loop_idx_5 inrange 0, 2, 1 iter {logical_tensor %aux_1 = %aux;token %aux_7_w_for_iter = %None;} #parallel(false) #submit_before_loop(false) #_loop_conds(Unsupported) #_config_scope(Unsupported) #unroll_times(1) {
        logical_tensor %aux_3, token %aux_3_w, token %src_r = ASSEMBLE(%src) token();
        logical_tensor %aux_5, token %aux_5_w, token %src_r = ASSEMBLE(%src) token(%aux_3_w);
        continue %aux_5, %aux_5_w;
    }
    logical_tensor %$0, token %$0_w, token %aux_7_r = ADDS(%aux_7) token(%aux_7_w_for);
    logical_tensor %out_1, token %out_1_w, token %$0_r = ASSEMBLE(%$0) token(%$0_w);
    return %src, %aux_7, %out_1;
}
"""


IR_15 = """
function foo incast(logical_tensor %src, logical_tensor %aux, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %aux_7, token %aux_7_w_for = for %loop_idx_5 inrange 0, 2, 1 iter {logical_tensor %aux_1 = %aux;token %aux_7_w_for_iter = %None;} #parallel(false) #submit_before_loop(false) #_loop_conds(Unsupported) #_config_scope(Unsupported) #unroll_times(1) {
        logical_tensor %aux_3, token %aux_3_w = ASSEMBLE(%src) token();
        logical_tensor %aux_5, token %aux_5_w = ASSEMBLE(%src) token(%aux_3_w);
        continue %aux_5, %aux_5_w;
    }
    logical_tensor %$0 = ADDS(%aux_7) token(%aux_7_w_for);
    logical_tensor %out_1 = ASSEMBLE(%$0) token();
    return %src, %aux_7, %out_1;
}
"""


def test_keep_dependency_when_loop_writes_are_read_after_loop():
    def foo(src, aux, out):
        for _ in pypto.loop(2):
            pypto.assemble(src, [0, 0], aux)
            pypto.assemble(src, [16, 0], aux)
        pypto.assemble(aux + 1.0, [0, 0], out)

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "out"),
    )
    check_snapshot(before_remove, BEFORE_IR_15)
    check_snapshot(after_remove, IR_15)
