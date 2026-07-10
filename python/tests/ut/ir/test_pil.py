# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import pypto
import pytest
from pypto import pil, ir, logging
from pypto.pil.ops import has_scalar

# ---------- Ops compile tests ----------


def _compile_ops(func, *args):
    return pil.compile(func, *args)


def _tensor_ops_of(ops):
    return [op for op in ops if isinstance(op, ir.TensorOpStmt)]


def test_tensor_binary_ops():
    def f(x, y):
        pypto.set_vec_tile_shapes(16, 16)
        z = x / y
        z = x ** y

    x = pypto.Tensor(shape=(4, 4), dtype=pypto.DT_FP32)
    y = pypto.Tensor(shape=(4, 4), dtype=pypto.DT_FP32)
    func = _compile_ops(f, x, y)
    ts = _tensor_ops_of(func.body)
    opcodes = {s.opcode for s in ts}
    assert 'DIV' in opcodes
    assert 'POW' in opcodes


def test_scalar_binary_bitwise_ops():
    def f(x, y):
        z = x | y
        z = x ^ y
        z = x | y
        z = x ^ y
        z = x & y
        z = x << y
        z = x >> y

    func = _compile_ops(f, 1, 2)
    ts = _tensor_ops_of(func.body)
    assert not ts


def test_tensor_matmul_op():
    def f(x, y):
        pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
        z = x @ y

    x = pypto.Tensor(shape=(32, 32), dtype=pypto.DT_FP32)
    y = pypto.Tensor(shape=(32, 32), dtype=pypto.DT_FP32)
    func = _compile_ops(f, x, y)
    ts = _tensor_ops_of(func.body)
    opcodes = {s.opcode for s in ts}
    assert 'A_MUL_B' in opcodes


def test_create_func():
    def f(a, b, out):
        pypto.set_vec_tile_shapes(16, 16)
        c = a + b
        c1 = a - b
        for i in pypto.loop(2):
            d = c * c1
            if i > 1:
                e = d * 2
            else:
                e = d + b
            out[:] = e

    shape = [32, 32]
    a = pypto.Tensor(shape=shape, dtype=pypto.DT_FP32, name="a")
    b = pypto.Tensor(shape=shape, dtype=pypto.DT_FP32, name="b")
    out = pypto.Tensor(shape=shape, dtype=pypto.DT_FP32, name="out")

    func = pil.compile(f, a, b, out)
    ts = _tensor_ops_of(func.body)
    opcodes = {s.opcode for s in ts}
    assert 'ADD' in opcodes


def test_tensor_unary_ops():
    def f(x):
        pypto.set_vec_tile_shapes(16, 16)
        z = -x
        z = +x

    x = pypto.Tensor(shape=(4, 4), dtype=pypto.DT_INT32)
    func = _compile_ops(f, x)
    ts = _tensor_ops_of(func.body)
    opcodes = {s.opcode for s in ts}
    # pypto use MULS to impl neg and pos just return self
    assert 'MULS' in opcodes


def test_pypto2ir():
    def f(x, y):
        pypto.set_vec_tile_shapes(16, 16)
        z = pypto.Tensor((x.shape[0], 64), dtype=pypto.DT_FP32, name="z")
        i = 0
        tx = pypto.view(x, [16, 16], [i * 32, 0])
        tx2 = pypto.add(tx, tx)
        pypto.assemble(tx2, [i * 32, 0], z)

        tz = pypto.view(z, [16, 16], [i * 32, 0])
        tz2 = pypto.add(tz, tz)
        pypto.assemble(tz2, [i * 32, 0], y)
        min(1, 2, 3, 4, 5)

    x = pypto.Tensor(shape=(-1, 64), dtype=pypto.DT_FP32, name="x")
    y = pypto.Tensor(shape=(-1, 64), dtype=pypto.DT_FP32, name="y")
    func = pil.compile(f, x, y)


def test_ir_range():
    results = []

    def foo(n):
        total = 0
        for i in range(n):
            # nested if
            if i < 3:
                if i == 0:
                    results.append(("nested_if", i))
                else:
                    results.append(("nested_else", i))
                    total += i
            # nested loop with continue/break
            if i == 5:
                for j in range(4):
                    if j == 1:
                        continue
                    if j == 2:
                        break
                    results.append(("inner", j))
                    total += j
            # break in outer loop
            if i == 7:
                break
            total += i
        results.append(("final", i, total))

    foo(10)
    expected, results = results, []
    pil.compile(foo, 10)
    assert results == expected


def test_ir_loop():
    def foo(n):
        ans = 0
        for i in pypto.loop(n):
            ans += i
        ans = ans + 10
    pil.compile(foo, 10)

# ---------- Loop IR tests ----------


def _for_ops_of(ops):
    return [op for op in ops if isinstance(op, ir.ForStmt)]


def info(x):
    logging.log_info(f"{x}")


def test_ir_loop_basic():
    """Carry variables: ans and loop var i."""
    def foo(n):
        ans = 0
        for i in pypto.loop(n):
            ans += i
    func = pil.compile(foo, 10)
    for_stmts = _for_ops_of(func.body)
    assert len(for_stmts) == 1
    f = for_stmts[0]
    assert len(f.iter_args) == 2
    assert len(f.return_vars) == 2
    assert isinstance(f.start, ir.ConstInt) and f.start.value == 0
    assert isinstance(f.stop, ir.ConstInt) and f.stop.value == 10
    assert isinstance(f.step, ir.ConstInt) and f.step.value == 1


def test_ir_loop_no_carry():
    """Loop var i is always carried."""
    def foo(n):
        for _ in pypto.loop(n):
            pass
    func = pil.compile(foo, 5)
    for_stmts = _for_ops_of(func.body)
    assert len(for_stmts) == 1
    f = for_stmts[0]
    assert len(f.iter_args) == 1
    assert len(f.return_vars) == 1


def test_ir_loop_two_carries():
    """Loop carrying ans, count, and loop var i."""
    def foo(n):
        ans = 0
        count = 1
        for i in pypto.loop(n):
            ans += i
            count += 1
    func = pil.compile(foo, 10)
    for_stmts = _for_ops_of(func.body)
    assert len(for_stmts) == 1
    f = for_stmts[0]
    assert len(f.iter_args) == 3
    assert len(f.return_vars) == 3


def test_ir_loop_unroll_supports_local_float_carry():
    def foo(n):
        for i, k in pypto.loop_unroll(0, n, 1, unroll_list=[8, 1]):
            scale = 1e15
            _ = i + k
            _ = scale

    func = pil.compile(foo, 16)
    for_stmts = _for_ops_of(func.body)
    assert len(for_stmts) == 2


def test_ir_loop_range_two_args():
    """pypto.loop(start, stop) form."""
    def foo():
        ans = 0
        for i in pypto.loop(2, 10):
            ans += i
    func = pil.compile(foo)
    for_stmts = _for_ops_of(func.body)
    assert len(for_stmts) == 1
    f = for_stmts[0]
    assert isinstance(f.start, ir.ConstInt) and f.start.value == 2
    assert isinstance(f.stop, ir.ConstInt) and f.stop.value == 10
    assert isinstance(f.step, ir.ConstInt) and f.step.value == 1
    assert len(f.iter_args) == 2


def test_ir_loop_range_three_args():
    """pypto.loop(start, stop, step) form."""
    def foo():
        ans = 0
        for i in pypto.loop(1, 20, 3):
            ans += i
    func = pil.compile(foo)
    for_stmts = _for_ops_of(func.body)
    assert len(for_stmts) == 1
    f = for_stmts[0]
    assert isinstance(f.start, ir.ConstInt) and f.start.value == 1
    assert isinstance(f.stop, ir.ConstInt) and f.stop.value == 20
    assert isinstance(f.step, ir.ConstInt) and f.step.value == 3
    assert len(f.iter_args) == 2


def test_ir_loop_sequential():
    """Two sequential loops, both carry ans and their loop var."""
    def foo(n):
        ans = 0
        for i in pypto.loop(n):
            ans += i
        for j in pypto.loop(n):
            ans += j
    func = pil.compile(foo, 5)
    for_stmts = _for_ops_of(func.body)
    assert len(for_stmts) == 2
    # Both loops carry ans and their loop var
    for f in for_stmts:
        assert len(f.iter_args) == 2
        assert len(f.return_vars) == 2


def test_ir_loop_nested():
    """Nested loops: inner ForStmt is inside outer's body."""
    def foo(n, m):
        ans = 0
        for i in pypto.loop(n):
            for j in pypto.loop(m):
                if j % 2 == 0:
                    ans += i + j
                else:
                    ans += i - j
            ans += n
    pil.compile(foo, 4, 5)


def test_ir_loop_nested1():
    """Nested loops: inner ForStmt is inside outer's body."""
    def foo(n, m):
        ans = 0
        for i in pypto.loop(n):
            for j in pypto.loop(m):
                if j % 2 == 0:
                    ans += i + j
                    break
                else:
                    ans += i - j
            ans += n
    with pytest.raises(SyntaxError):
        pil.compile(foo, 4, 5)


def test_ir_loop_carry_used_after():
    """Carry variable used after the loop produces a scalar op."""
    def foo(n):
        ans = 0
        for i in pypto.loop(n):
            ans += i
        ans = ans + 10
    func = pil.compile(foo, 10)
    for_stmts = _for_ops_of(func.body)
    assert len(for_stmts) == 1
    assert len(for_stmts[0].iter_args) == 2  # i and ans


def test_ir_loop_body_multiple_ops():
    """i, a and ans are all carried."""
    def foo(n):
        ans = 0
        for i in pypto.loop(n):
            a = i * 2
            ans += a
        ans = ans + a
    func = pil.compile(foo, 10)
    for_stmts = _for_ops_of(func.body)
    assert len(for_stmts) == 1
    f = for_stmts[0]
    assert len(f.iter_args) == 3
    assert len(f.return_vars) == 3
    # Body should contain mul and add scalar ops


def test_ir_deadcode():
    def foo(n):
        for i in pypto.loop(n):
            x = i + 1
            break
            return x

    with pytest.raises(SyntaxError):
        pil.compile(foo, 10)

# ---------- Loop control flow IR tests ----------


def _collect_stmts(stmt, cls):
    """Recursively collect all stmts of a given ir."""
    result = []
    if isinstance(stmt, cls):
        result.append(stmt)
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            result.extend(_collect_stmts(s, cls))
    if isinstance(stmt, ir.IfStmt):
        result.extend(_collect_stmts(stmt.then_body, cls))
        result.extend(_collect_stmts(stmt.else_body, cls))
    if isinstance(stmt, ir.ForStmt):
        result.extend(_collect_stmts(stmt.body, cls))
    return result


def test_ir_loop_if_else_both_branches():
    """if/else inside loop compiles both branches."""
    def foo(n):
        ans = 0
        for i in pypto.loop(n):
            if i:
                ans += i
            else:
                ans += 1
    func = pil.compile(foo, 10)
    for_stmts = _for_ops_of(func.body)
    assert len(for_stmts) == 1
    if_stmts = _collect_stmts(for_stmts[0].body, ir.IfStmt)
    assert len(if_stmts) >= 1


def test_tensor_add_dyn():
    """Add dynamic tensor should be supported."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            y[i:, :] = ta + 1

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = pil.compile(foo, x, y)
    b = ir.IRBuilder()
    prog = b.create_program([func], "main", ir.Span.unknown())
    logging.log_info(f"\norigin: {prog}")
    prog = ir.Pass.canonicalize()(prog)
    logging.log_info(f"\ncanonical: {prog}")


def test_fstring():
    def foo(x, y):
        # basic expressions
        assert f"{x + y=} , {x - y=}" == "x + y=30 , x - y=-10"
        # simple variable reference
        assert f"x={x}, y={y}" == "x=10, y=20"
        # arithmetic in f-string
        assert f"sum={x + y}, diff={x - y}, prod={x * y}" == "sum=30, diff=-10, prod=200"
        # format specifiers
        assert f"x={x:05d}, y={y:05d}" == "x=00010, y=00020"
        # nested expressions with modulo
        assert f"mod={x % 3}, pow={x ** 2}" == "mod=1, pow=100"
        # mixed literal and expression parts
        assert f"result: ({x} + {y}) = {x + y}" == "result: (10 + 20) = 30"
        # conversion specifiers
        assert f"{x!r}, {y!s}" == "10, 20"
    pil.compile(foo, 10, 20)


def test_tensor_loop_unroll():
    """Add dynamic tensor should be supported."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32, unroll_list=[4]):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            y[i:, :] = ta + 1

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = pil.compile(foo, x, y)
    b = ir.IRBuilder()
    prog = b.create_program([func], "main", ir.Span.unknown())
    prog = ir.Pass.canonicalize()(prog)
    info(f"\ncanonical: {prog}")


def test_tensor_loop_unroll_batch():
    """Add dynamic tensor should be supported."""
    def foo(x, y):
        for i, k in pypto.loop_unroll(x.shape[0], unroll_list=[32, 16, 1]):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + k, :]
            y[i:, :] = ta + 1

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = pil.compile(foo, x, y)
    b = ir.IRBuilder()
    prog = b.create_program([func], "main", ir.Span.unknown())
    prog = ir.Pass.canonicalize()(prog)
    info(f"\ncanonical: {prog}")


def test_loop_unroll_idx_name():
    def foo(n):
        for i, k in pypto.loop_unroll(n, name="LoopA", idx_name="i"):
            _ = i + k

    func = pil.compile(foo, 10)
    for_stmts = _for_ops_of(func.body)
    assert len(for_stmts) == 1
    assert str(for_stmts[0].loop_var) == "loop_idx_i"


def test_loop_unroll_default_idx_name():
    def foo(n):
        for i, k in pypto.loop_unroll(n):
            _ = i + k

    func = pil.compile(foo, 10)
    for_stmts = _for_ops_of(func.body)
    assert len(for_stmts) == 1
    assert str(for_stmts[0].loop_var).startswith("loop_idx_")


def _opcode_set(stmt, acc=None):
    """Recursively collect TensorOpStmt opcodes."""
    if acc is None:
        acc = set()
    if isinstance(stmt, ir.TensorOpStmt):
        acc.add(stmt.opcode)
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            _opcode_set(s, acc)
    if isinstance(stmt, ir.IfStmt):
        _opcode_set(stmt.then_body, acc)
        _opcode_set(stmt.else_body, acc)
    if isinstance(stmt, ir.ForStmt):
        _opcode_set(stmt.body, acc)
    return acc


def test_nested_function_no_args():
    """Nested function with no arguments."""
    def foo():
        def bar():
            return 41
        info(bar() + 1)
    func = pil.compile(foo)
    assert func is not None


def test_nested_function_args():
    """Nested function taking positional arguments."""
    def foo(x):
        def bar(a, b):
            return a + b
        info(bar(x, x))
    pil.compile(foo, 5)


def test_nested_function_default_args():
    """Nested function with default argument values."""
    def foo():
        def bar(a, b=10):
            return a + b
        info(bar(5))
    pil.compile(foo)


def test_nested_function_capture_scalar():
    """Nested function captures a scalar from the enclosing scope."""
    def foo(x):
        def bar():
            return x + 1
        info(bar() + 1)
    pil.compile(foo, 5)


def test_nested_function_capture_tensor():
    """Nested function captures a tensor; the captured tensor op inlines."""
    def foo(a):
        pypto.set_vec_tile_shapes(16, 16)

        def bar():
            return a + a
        info(bar())

    a = pypto.Tensor((4, 4), dtype=pypto.DT_FP32)
    func = pil.compile(foo, a)
    assert _opcode_set(func.body) == {'ADD'}


def test_nested_function_recursion():
    """Nested function calls itself (base case via early return)."""
    def foo(n):
        def fac(k):
            if k <= 1:
                return 1
            return k * fac(k - 1)
        info(fac(n))
    pil.compile(foo, 5)


def test_nested_function_in_loop():
    """Nested function called inside a loop body."""
    def foo(n):
        def sq(x):
            return x * x
        total = 0
        for i in range(n):
            total += sq(i)
        info(total)
    pil.compile(foo, 4)


def test_nested_function_posonly_args():
    """Nested function with positional-only parameters (before ``/``)."""
    def foo(x):
        def bar(a, b, /):
            return a + b
        info(bar(x, x))
    pil.compile(foo, 5)


def test_nested_function_kwonly_args():
    """Nested function with keyword-only parameters (after ``*``)."""
    def foo(x):
        def bar(a, *, k):
            return a + k
        info(bar(x, k=3))
    pil.compile(foo, 5)


def test_nested_function_kwonly_default_args():
    """Keyword-only parameters with defaults, mixed with positional defaults."""
    def foo(x):
        def bar(a, b=2, /, c=3, *, k=10):
            return a + b + c + k
        info(bar(x))  # b, c and k all fall back to their defaults
    pil.compile(foo, 1)


def test_nested_function_return_tensor():
    """Add dynamic tensor should be supported."""
    def foo(x, y):
        def add1(a):
            return a + 1

        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            y[i:, :] = add1(ta)

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = pil.compile(foo, x, y)
    b = ir.IRBuilder()
    prog = b.create_program([func], "main", ir.Span.unknown())
    prog = ir.Pass.canonicalize()(prog)
    logging.log_info(f"\ncanonical: {prog}")


def test_pil_function():
    """Add dynamic tensor should be supported."""
    @pil.function
    def add1(n, a):
        for _ in pypto.loop(n):
            a = a + 1
        return a

    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            y[i:, :] = add1(10, ta)

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = pil.compile(foo, x, y)
    b = ir.IRBuilder()
    prog = b.create_program([func], "main", ir.Span.unknown())
    prog = ir.Pass.canonicalize()(prog)
    info(f"\ncanonical: {prog}")


def test_pil_starred():
    res = []

    def f(x):
        res.append((*x, *x))
        res.append([*x, *x])
        res.append({*x, *x})

    pil.compile(f, [1, 2])
    assert res[0] == (1, 2, 1, 2)
    assert res[1] == [1, 2, 1, 2]
    assert res[2] == {1, 2}


def test_pil_dict():
    res = []

    def f(x):
        x = {1: 2, 3: 4}
        res.append(x)

    pil.compile(f, [1, 2])
    assert res[0] == {1: 2, 3: 4}


def test_pil_dict_unpack():
    res = []

    def f(x):
        res.append({1: 2, **x, 3: 4})

    pil.compile(f, {5: 6})
    assert res[0] == {1: 2, 5: 6, 3: 4}


def test_pil_call_kwargs():
    res = []

    def g(a, b, c):
        res.append((a, b, c))

    def f(x):
        g(**x)

    pil.compile(f, {'a': 1, 'b': 2, 'c': 3})
    assert res[0] == (1, 2, 3)


def test_pil_ifexpr():
    res = []

    def foo(a, b):
        c = a if a > b else b
        res.append(c)
    pil.compile(foo, 1, 2)
    assert res == [2]


def test_pil_multi_and():
    res = []

    def foo(a, b, c):
        c = a if a and b and c else b
        res.append(c)
    pil.compile(foo, 1, 2, 0)
    assert res == [2]


def test_has_scalar_symbolic_scalar():
    assert has_scalar([pypto.symbolic_scalar("n"), 1]) is True
    assert has_scalar([1, 2, 3]) is False
