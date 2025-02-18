#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import re
from typing import Union

import sympy

from . import pypto_impl

_replacements = {
    "RUNTIME_Min": "min",
    "RUNTIME_Max": "max",
    "RUNTIME_Eq": "Eq",
    "RUNTIME_Ne": "Ne",
    "/": "//",  # 'a' / 8 * 8 may be optomized by sympy, that's bad
}

_REPLACE_PATTERN = re.compile("|".join(_replacements.keys()))


def _expr_preprocess(s: str) -> str:
    return _REPLACE_PATTERN.sub(lambda m: _replacements[m.group(0)], s)


class SymbolicScalar:

    def __init__(self, arg0: Union[int, str, 'SymbolicScalar'],
                 arg1: Union[int, None] = None):
        """
        Construct a SymbolicScalar.

        Args:
            arg0 Union[int, str]: The value or name of the symbolic scalar
            arg1 Union[int, None]: The value of the symbolic scalar. Defaults to None.

        Examples:
            >>> b = SymbolicScalar(10)
            >>> c = SymbolicScalar("x")
            >>> d = SymbolicScalar("x", 10)
        """
        if isinstance(arg0, int):
            self._base = pypto_impl.SymbolicScalar(arg0)
        elif isinstance(arg0, str):
            if isinstance(arg1, int):
                self._base = pypto_impl.SymbolicScalar(arg0, arg1)
            else:
                self._base = pypto_impl.SymbolicScalar(arg0)
        elif isinstance(arg0, SymbolicScalar):
            self._base = arg0._base
        else:
            raise ValueError(f"Invalid arguments")

    def __str__(self) -> str:
        return self._base.Dump()

    def __repr__(self) -> str:
        return f"SymbolicScalar({self._base.Dump()})"

    def __eq__(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__eq__', lambda a, b: a == b, lambda a, b: a.Eq(b))

    def __ne__(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__ne__', lambda a, b: a != b, lambda a, b: a.Ne(b))

    def __lt__(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__lt__', lambda a, b: a < b, lambda a, b: a.Lt(b))

    def __le__(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__le__', lambda a, b: a <= b, lambda a, b: a.Le(b))

    def __gt__(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__gt__', lambda a, b: a > b, lambda a, b: a.Gt(b))

    def __ge__(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__ge__', lambda a, b: a >= b, lambda a, b: a.Ge(b))

    def __add__(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__add__', lambda a, b: a + b, lambda a, b: a.Add(b))

    def __radd__(self, other: int) -> 'SymbolicScalar':
        return self._binary_ops(other, '__radd__', lambda a, b: b + a, lambda a, b: a.RAdd(b))

    def __sub__(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__sub__', lambda a, b: a - b, lambda a, b: a.Sub(b))

    def __rsub__(self, other: int) -> 'SymbolicScalar':
        return self._binary_ops(other, '__rsub__', lambda a, b: b - a, lambda a, b: a.RSub(b))

    def __mul__(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__mul__', lambda a, b: a * b, lambda a, b: a.Mul(b))

    def __rmul__(self, other: int) -> 'SymbolicScalar':
        return self._binary_ops(other, '__rmul__', lambda a, b: b * a, lambda a, b: a.RMul(b))

    def __truediv__(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__truediv__', lambda a, b: a // b, lambda a, b: a.Div(b))

    def __rtruediv__(self, other: int) -> 'SymbolicScalar':
        return self._binary_ops(other, '__rtruediv__', lambda a, b: b // a, lambda a, b: a.RDiv(b))

    def __mod__(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__mod__', lambda a, b: a % b, lambda a, b: a.Mod(b))

    def __rmod__(self, other: int) -> 'SymbolicScalar':
        return self._binary_ops(other, '__rmod__', lambda a, b: b % a, lambda a, b: a.RMod(b))

    def __floordiv__(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__floordiv__', lambda a, b: a // b, lambda a, b: a.Div(b))

    def __rfloordiv__(self, other: int) -> 'SymbolicScalar':
        return self._binary_ops(other, '__rfloordiv__', lambda a, b: b // a, lambda a, b: a.RDiv(b))

    def __neg__(self) -> 'SymbolicScalar':
        return self._unary_ops(lambda a: -a, lambda a: a.Neg())

    def __pos__(self) -> 'SymbolicScalar':
        return self._unary_ops(lambda a: +a, lambda a: a.Pos())

    def __invert__(self) -> 'SymbolicScalar':
        return self._unary_ops(lambda a: not a, lambda a: a.Not())

    def __int__(self) -> int:
        return self.concrete()

    def __bool__(self) -> bool:
        return bool(self.concrete())

    @classmethod
    def from_base(cls, base: pypto_impl.SymbolicScalar) -> 'SymbolicScalar':
        obj = cls.__new__(cls)
        obj._base = base
        return obj

    def is_immediate(self) -> bool:
        return self._base.IsImmediate()

    def is_symbol(self) -> bool:
        return self._base.IsSymbol()

    def is_expression(self) -> bool:
        return self._base.IsExpression()

    def is_concrete(self) -> bool:
        return self._base.ConcreteValid()

    def concrete(self) -> int:
        if self.is_concrete():
            return self._base.Concrete()
        else:
            raise ValueError("Not concrete value")

    def as_variable(self) -> None:
        self._base.AsIntermediateVariable()

    def base(self) -> pypto_impl.SymbolicScalar:
        return self._base

    def min(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__min__', lambda a, b: min(a, b), lambda a, b: a.Min(b))

    def max(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar':
        return self._binary_ops(other, '__max__', lambda a, b: max(a, b), lambda a, b: a.Max(b))

    def _binary_ops(self, other, name: str, bop, sym_bop):
        if isinstance(other, int):
            if self.is_concrete():
                out = SymbolicScalar(bop(self.concrete(), other))
            else:
                out = self.from_base(sym_bop(self._base, other))
        else:
            if self.is_concrete() and other.is_concrete():
                out = SymbolicScalar(bop(self.concrete(), other.concrete()))
            else:
                out = self.from_base(sym_bop(self._base, other._base))

        if not out.is_concrete():
            expr = _expr_preprocess(str(out))
            try:
                expr = sympy.simplify(expr)
                if isinstance(expr, sympy.Integer):
                    out = SymbolicScalar(int(expr))
                elif expr == sympy.true:
                    out = SymbolicScalar(1)
                elif expr == sympy.false:
                    out = SymbolicScalar(0)
            except Exception:
                pass
        return out

    def _unary_ops(self, uop, sym_uop):
        if self.is_concrete():
            return SymbolicScalar(uop(self.concrete()))
        else:
            return self.from_base(sym_uop(self._base))


SymInt = Union[int, SymbolicScalar]
