# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import operator
import pytest

import pypto
from pypto import pil, logging
from pypto.pil.op_registry import OpRegistry
from pypto.pil import pir


def _parse_func(func, dump=False):
    body = pil.ast2pil(func).body
    if dump:
        logging.log_info(str(body))
    return body


def _has(blk, callee):
    return any(s.callee == callee for s in blk.calls if isinstance(s, pir.Call))


def test_visit_binop():
    def f(x, y):
        z = x + y
        z = x - y
        z = x * y
        z = x // y
        z = x / y
        z = x % y
        z = x ** y
        z = x | y
        z = x ^ y
        z = x & y
        z = x << y
        z = x >> y

    blk = _parse_func(f)
    for op in [operator.add, operator.sub, operator.mul, operator.floordiv,
               operator.truediv, operator.mod, operator.pow, operator.or_,
               operator.xor, operator.and_, operator.lshift, operator.rshift]:
        assert _has(blk, op), f"Missing binop {op}"


def test_visit_unaryop():
    def f(x):
        y = -x
        y = +x
        y = ~x
        y = not x

    blk = _parse_func(f)
    for op in [operator.neg, operator.pos, operator.invert, operator.not_]:
        assert _has(blk, op), f"Missing unaryop {op}"


def test_visit_compare():
    def f(x, y):
        z = x == y
        z = x != y
        z = x < y
        z = x <= y
        z = x > y
        z = x >= y
        z = x is y
        z = x is not y

    blk = _parse_func(f)
    for op in [operator.eq, operator.ne, operator.lt, operator.le,
               operator.gt, operator.ge, operator.is_, operator.is_not]:
        assert _has(blk, op), f"Missing compare {op}"


def test_visit_compare_chained():
    def f(a, b, c):
        z = a < b < c

    blk = _parse_func(f)
    assert _has(blk, "pil.if_else"), "Missing if_else call"


def test_visit_boolop():
    def f(a, b, c):
        z = a and b
        z = a or b
        z = a and b and c
        z = a or b or c

    blk = _parse_func(f)
    assert _has(blk, "pil.if_else"), "Missing if_else call"


def test_visit_ifexp():
    def f(x, y, z):
        w = x if y else z

    blk = _parse_func(f)
    assert _has(blk, "pil.if_else"), "Missing if_else call"


def test_visit_attribute():
    def f(x):
        y = x.attr

    blk = _parse_func(f)
    assert _has(blk, getattr)


def test_visit_subscript():
    def f(x):
        y = x[0]
        y = x[1:2:3]
        y = x[1:]

    blk = _parse_func(f)
    assert _has(blk, operator.getitem)
    assert _has(blk, slice)


def test_visit_tuple_and_list():
    def f(x, y):
        z = (x, y)
        z = [x, y]

    blk = _parse_func(f)
    assert _has(blk, tuple)
    assert _has(blk, list)


def test_visit_joined_str():
    def f(x):
        y = f"hello {x}"
        y = f"{x:.2f}"
        y = f"a{x}b"

    blk = _parse_func(f)
    assert len([s for s in blk.calls if isinstance(s, pir.Call) and s.callee == "pil.fstring"]) >= 2


# ---------- Statements ----------

def test_visit_assign():
    def f(x, v):
        y = x
        a, b = x
        [a, b] = x
        x[0] = v
        x.attr = v

    blk = _parse_func(f)
    store_calls = [s for s in blk.calls if isinstance(s, pir.Call) and s.callee == "pil.store"]
    assert any("y" in c.args for c in store_calls)
    assert len([s for s in blk.calls if isinstance(s, pir.Call) and s.callee == "pil.unpack"]) == 2
    assert _has(blk, operator.setitem)
    assert _has(blk, setattr)


def test_visit_augassign():
    def f(x, y):
        x += y
        x[0] += y
        x.attr += y

    blk = _parse_func(f)
    assert _has(blk, operator.add)
    assert _has(blk, operator.getitem)
    assert _has(blk, operator.setitem)
    assert _has(blk, getattr)
    assert _has(blk, setattr)


def test_visit_annassign():
    def f(x):
        y: int = x

    blk = _parse_func(f)
    assert any(s.callee == "pil.store" and "y" in s.args for s in blk.calls if isinstance(s, pir.Call))

    def g():
        x: int

    assert len([s for s in _parse_func(g).calls if isinstance(s, pir.Call)]) == 0


def test_visit_if():
    def f(x, y, z):
        if x:
            w = y
        else:
            w = z

    blk = _parse_func(f)
    assert _has(blk, "pil.if_else"), "Missing if_else call"


def test_visit_return():
    def f(x):
        return x

    blk = _parse_func(f)
    assert blk.jump == pir.Jump.RETURN and blk.result is not None

    def g():
        return

    blk2 = _parse_func(g)
    assert blk2.jump == pir.Jump.RETURN and blk2.result is None


def test_visit_pass_functiondef_expr():
    def f(x):
        pass

        def g():
            pass
        _ = x + 1

    blk = _parse_func(f)
    assert len(blk.calls) >= 1


def test_visit_while():
    def f(x):
        while x:
            x = x - 1

    blk = _parse_func(f)
    assert _has(blk, "pil.loop")

    def g(x):
        while x:
            break

    assert _has(_parse_func(g), "pil.loop")


def test_visit_for():
    def f(xs):
        for _ in xs:
            pass

    blk = _parse_func(f)
    assert _has(blk, "pil.loop")


def test_visit_for_with_body():
    def f(xs):
        for x in xs:
            _ = x

    blk = _parse_func(f)
    assert _has(blk, "pil.loop")


def test_visit_continue_break():
    def f(xs):
        for _ in xs:
            continue

    blk = _parse_func(f)
    assert _has(blk, "pil.loop")

    def g(xs):
        for _ in xs:
            break

    assert _has(_parse_func(g), "pil.loop")


def test_visit_while_else_error():
    def f(x):
        while x:
            pass
        else:
            pass

    with pytest.raises(Exception):
        _parse_func(f)


def test_visit_for_else_error():
    def f(xs):
        for _ in xs:
            pass
        else:
            pass

    with pytest.raises(Exception):
        _parse_func(f)


def test_pypto_loop():
    def f(x: pypto.Tensor[-1, 64], y: pypto.Tensor[-1, 64]):
        for i in pypto.loop(x.shape[0] // 32):
            tx = pypto.view(x, [16, 16], [i * 32, 0])
            tx2 = pypto.add(tx, tx)
            pypto.assemble(tx2, [i * 32, 0], y)

    _parse_func(f)


def test_pypto_loop2():
    def f(x: pypto.Tensor[-1, 64], y: pypto.Tensor[-1, 64]):
        z = pypto.Tensor((x.shape[0], 64), dtype=pypto.DT_FP32, name="z")
        for i in pypto.loop(x.shape[0] // 32):
            tx = pypto.view(x, [16, 16], [i * 32, 0])
            tx2 = pypto.add(tx, tx)
            pypto.assemble(tx2, [i * 32, 0], z)

        for i in pypto.loop(z.shape[0] // 32):
            tz = pypto.view(z, [16, 16], [i * 32, 0])
            tz2 = pypto.add(tz, tz)
            pypto.assemble(tz2, [i * 32, 0], y)

    _parse_func(f)


def test_op_registry():
    registry = OpRegistry()

    @registry.impl('pil.foo', partial=True)
    def foo(ctx, s, x):
        """
        Foo helper
        """
        return f'{s} {x}'

    assert "Foo helper" in str(foo.__doc__)
    assert registry.dispatch('pil.foo', None, 2) == 'pil.foo 2'

    @registry.impl('pil.bar')
    def bar(ctx, x):
        """
        Bar helper
        """
        return x + 1

    assert "Bar helper" in str(bar.__doc__)
    assert registry.dispatch('pil.bar', None, 2) == 3
