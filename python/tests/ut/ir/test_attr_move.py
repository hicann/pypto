# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pypto
from pypto import pil

from .test_common import check_snapshot

_GOLDEN_DIR = Path(__file__).parent

IR1 = _GOLDEN_DIR / "test_attr_move1.pypto"

IR2 = _GOLDEN_DIR / "test_attr_move2.pypto"

IR3 = _GOLDEN_DIR / "test_attr_move3.pypto"

IR4 = _GOLDEN_DIR / "test_attr_move4.pypto"

IR5 = _GOLDEN_DIR / "test_attr_move5.pypto"

IR6 = _GOLDEN_DIR / "test_attr_move6.pypto"

IR7 = _GOLDEN_DIR / "test_attr_move7.pypto"


@dataclass
class Data:
    a: Optional[pypto.Tensor] = None


def test_attr_move():

    def bar(x):
        return pypto.add(x, 1)

    def foo(a, b):
        d = Data()
        d.a = pypto.full([32, 32], 0.0, pypto.DT_FP32)
        for i in pypto.loop(32):
            d.a[:] = bar(d.a)
        a[:] = d.a + b

    x = pypto.Tensor((32, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((32, 32), pypto.DT_FP32, 'y')
    func = pil.compile(foo, x, y)
    check_snapshot(func, IR1)


@dataclass
class Pair:
    a: Optional[pypto.Tensor] = None
    b: Optional[pypto.Tensor] = None


def test_attr_move_pair():

    def foo(a, b):
        d = Pair()
        d.a = pypto.full([32, 32], 0.0, pypto.DT_FP32)
        d.b = pypto.full([32, 32], 0.0, pypto.DT_FP32)
        for i in pypto.loop(32):
            d.a[:] = pypto.add(d.a, 1)
            d.b[:] = pypto.add(d.b, 1)
        a[:] = d.a + d.b

    x = pypto.Tensor((32, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((32, 32), pypto.DT_FP32, 'y')
    func = pil.compile(foo, x, y)
    check_snapshot(func, IR2)


def test_attr_assign():

    def foo(a, b):
        d = Data()
        d.a = pypto.full([32, 32], 0.0, pypto.DT_FP32)
        for i in pypto.loop(32):
            d.a = pypto.add(d.a, 1)
        a[:] = d.a + b

    x = pypto.Tensor((32, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((32, 32), pypto.DT_FP32, 'y')
    func = pil.compile(foo, x, y)
    check_snapshot(func, IR3)


@dataclass
class Outer:
    inner: Data = field(default_factory=Data)


def test_attr_move_multi_level():

    def bar(e):
        for i in pypto.loop(32):
            e.a = pypto.add(e.a, 1)

    def foo(a, b):
        d = Outer()
        d.inner.a = pypto.full([32, 32], 0.0, pypto.DT_FP32)
        bar(d.inner)
        a[:] = d.inner.a + b

    x = pypto.Tensor((32, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((32, 32), pypto.DT_FP32, 'y')
    func = pil.compile(foo, x, y)
    check_snapshot(func, IR4)


@dataclass
class Inner:
    c: Optional[pypto.Tensor] = None


@dataclass
class Holder:
    inner: Inner = field(default_factory=Inner)


def test_cross_func_attr_carry():
    """Caller-side loop must carry a slot the callee writes through a param
    passed as a plain local (carry name: inner.c)."""

    def bar(bb):
        bb.c = pypto.add(bb.c, 1)

    def foo(a, b):
        inner = Inner()
        inner.c = pypto.full([32, 32], 0.0, pypto.DT_FP32)
        for i in pypto.loop(32):
            bar(inner)
        a[:] = inner.c + b

    x = pypto.Tensor((32, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((32, 32), pypto.DT_FP32, 'y')
    func = pil.compile(foo, x, y)
    check_snapshot(func, IR5)


def test_cross_func_dotted_arg_carry():
    """Argument itself an attribute: bar(h.inner) writing bb.c carries
    h.inner.c (the chained name follows the object across the boundary)."""

    def bar(bb):
        bb.c = pypto.add(bb.c, 1)

    def foo(a, b):
        h = Holder()
        h.inner.c = pypto.full([32, 32], 0.0, pypto.DT_FP32)
        for i in pypto.loop(32):
            bar(h.inner)
        a[:] = h.inner.c + b

    x = pypto.Tensor((32, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((32, 32), pypto.DT_FP32, 'y')
    func = pil.compile(foo, x, y)
    check_snapshot(func, IR6)


def test_cross_func_move_carry():
    """MOVE through a helper param marks the caller's dotted slot."""

    def update(dst, val):
        dst.move(val)

    def foo(a, b):
        d = Data()
        d.a = pypto.full([32, 32], 0.0, pypto.DT_FP32)
        for i in pypto.loop(10):
            tmp = b + 1
            update(d.a, tmp)
        a[:] = d.a

    x = pypto.Tensor((32, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((32, 32), pypto.DT_FP32, 'y')
    func = pil.compile(foo, x, y)
    check_snapshot(func, IR7)
