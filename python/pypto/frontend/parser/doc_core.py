#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# pylint: disable=missing-class-docstring,useless-parent-delegation,too-few-public-methods,invalid-name
# pylint: disable=redefined-builtin, redefined-outer-name

"""PTO Script Parser Core AST Nodes - Stable AST Abstraction Layer.

This module defines the core AST (Abstract Syntax Tree) nodes for the PTO Script Parser.
The design is inspired by TVM's Script Parser doc AST and provides a standalone AST
structure that remains stable across Python versions, protecting the parser from
breaking changes in Python's standard ast module.

Purpose
-------
The doc AST serves as an abstraction layer providing version independence, simplified
structure for PTO scripts, and extensibility for PTO-specific features. It mirrors
Python's AST closely enough for straightforward bidirectional conversion.

Node Categories
---------------
- mod: Module-level nodes (Module, Interactive, Expression)
- stmt: Statement nodes (FunctionDef, Assign, For, If, Return, etc.)
- expr: Expression nodes (BinOp, Call, Name, Constant, etc.)
- operator: Arithmetic operators (Add, Sub, Mult, Div, etc.)
- boolop: Boolean operators (And, Or)
- unaryop: Unary operators (Not, UAdd, USub, Invert)
- cmpop: Comparison operators (Eq, Lt, Gt, etc.)
- expr_context: Expression context (Load, Store, Del)

Helper Classes
--------------
- arguments: Function argument specification
- arg: Single function argument
- keyword: Keyword argument in function calls
- alias: Import alias
- comprehension: List/set/dict comprehension specification
- excepthandler: Exception handler specification
- withitem: Context manager item

Base Classes
------------
- AST: Base class for all nodes with field initialization and comparison
- NodeVisitor: Base class for AST traversal using visitor pattern
- NodeTransformer: Base class for AST transformation

Note
----
- Minimal python version is 3.9
- Many classes are simple data containers
- Type annotations use forward references (strings) for self-referential types
- Position information (lineno, col_offset) is propagated from Python AST
"""

from typing import Any, Optional


class AST:
    def __init__(self) -> None:
        pass


class mod(AST):
    def __init__(self) -> None:
        super().__init__()


class Module(mod):
    body: list["stmt"]

    def __init__(self, body: list["stmt"]) -> None:
        super().__init__()
        self.body = body


class Interactive(mod):
    body: list["stmt"]

    def __init__(self, body: list["stmt"]) -> None:
        super().__init__()
        self.body = body


class Expression(mod):
    body: "expr"

    def __init__(self, body: "expr") -> None:
        super().__init__()
        self.body = body


class stmt(AST):
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int

    def __init__(
        self, lineno: int, col_offset: int, end_lineno: int, end_col_offset: int
    ) -> None:
        super().__init__()
        self.lineno = lineno
        self.col_offset = col_offset
        self.end_lineno = end_lineno
        self.end_col_offset = end_col_offset


class FunctionDef(stmt):
    name: str
    args: "arguments"
    body: list["stmt"]
    decorator_list: list["expr"]
    returns: Optional["expr"]

    def __init__(
        self,
        name: str,
        args: "arguments",
        body: list["stmt"],
        decorator_list: list["expr"],
        returns: Optional["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.name = name
        self.args = args
        self.body = body
        self.decorator_list = decorator_list
        self.returns = returns


class ClassDef(stmt):
    name: str
    bases: list["expr"]
    keywords: list["keyword"]
    body: list["stmt"]
    decorator_list: list["expr"]

    def __init__(
        self,
        name: str,
        bases: list["expr"],
        keywords: list["keyword"],
        body: list["stmt"],
        decorator_list: list["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.name = name
        self.bases = bases
        self.keywords = keywords
        self.body = body
        self.decorator_list = decorator_list


class Return(stmt):
    value: Optional["expr"]

    def __init__(
        self,
        value: Optional["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value


class Delete(stmt):
    targets: list["expr"]

    def __init__(
        self,
        targets: list["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.targets = targets


class Assign(stmt):
    targets: list["expr"]
    value: "expr"

    def __init__(
        self,
        targets: list["expr"],
        value: "expr",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.targets = targets
        self.value = value


class AugAssign(stmt):
    target: "expr"
    op: "operator"
    value: "expr"

    def __init__(
        self,
        target: "expr",
        op: "operator",
        value: "expr",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.target = target
        self.op = op
        self.value = value


class AnnAssign(stmt):
    target: "expr"
    annotation: "expr"
    value: Optional["expr"]
    simple: int

    def __init__(
        self,
        target: "expr",
        annotation: "expr",
        value: Optional["expr"],
        simple: int,
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.target = target
        self.annotation = annotation
        self.value = value
        self.simple = simple


class For(stmt):
    target: "expr"
    iter: "expr"
    body: list["stmt"]
    orelse: list["stmt"]

    def __init__(
        self,
        target: "expr",
        iter: "expr",
        body: list["stmt"],
        orelse: list["stmt"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.target = target
        self.iter = iter
        self.body = body
        self.orelse = orelse


class While(stmt):
    test: "expr"
    body: list["stmt"]
    orelse: list["stmt"]

    def __init__(
        self,
        test: "expr",
        body: list["stmt"],
        orelse: list["stmt"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.test = test
        self.body = body
        self.orelse = orelse


class If(stmt):
    test: "expr"
    body: list["stmt"]
    orelse: list["stmt"]

    def __init__(
        self,
        test: "expr",
        body: list["stmt"],
        orelse: list["stmt"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.test = test
        self.body = body
        self.orelse = orelse


class With(stmt):
    items: list["withitem"]
    body: list["stmt"]

    def __init__(
        self,
        items: list["withitem"],
        body: list["stmt"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.items = items
        self.body = body


class Raise(stmt):
    exc: Optional["expr"]
    cause: Optional["expr"]

    def __init__(
        self,
        exc: Optional["expr"],
        cause: Optional["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.exc = exc
        self.cause = cause


class Try(stmt):
    body: list["stmt"]
    handlers: list["excepthandler"]
    orelse: list["stmt"]
    finalbody: list["stmt"]

    def __init__(
        self,
        body: list["stmt"],
        handlers: list["excepthandler"],
        orelse: list["stmt"],
        finalbody: list["stmt"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.body = body
        self.handlers = handlers
        self.orelse = orelse
        self.finalbody = finalbody


class Assert(stmt):
    test: "expr"
    msg: Optional["expr"]

    def __init__(
        self,
        test: "expr",
        msg: Optional["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.test = test
        self.msg = msg


class Import(stmt):
    names: list["alias"]

    def __init__(
        self,
        names: list["alias"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.names = names


class ImportFrom(stmt):
    module: Optional[str]
    names: list["alias"]
    level: int

    def __init__(
        self,
        module: Optional[str],
        names: list["alias"],
        level: int,
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.module = module
        self.names = names
        self.level = level


class Global(stmt):
    names: list[str]

    def __init__(
        self,
        names: list[str],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.names = names


class Nonlocal(stmt):
    names: list[str]

    def __init__(
        self,
        names: list[str],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.names = names


class Expr(stmt):
    value: "expr"

    def __init__(
        self,
        value: "expr",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value


class Pass(stmt):

    def __init__(
        self, lineno: int, col_offset: int, end_lineno: int, end_col_offset: int
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Break(stmt):

    def __init__(
        self, lineno: int, col_offset: int, end_lineno: int, end_col_offset: int
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Continue(stmt):

    def __init__(
        self, lineno: int, col_offset: int, end_lineno: int, end_col_offset: int
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class expr(AST):
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int

    def __init__(
        self, lineno: int, col_offset: int, end_lineno: int, end_col_offset: int
    ) -> None:
        super().__init__()
        self.lineno = lineno
        self.col_offset = col_offset
        self.end_lineno = end_lineno
        self.end_col_offset = end_col_offset


class BoolOp(expr):
    op: "boolop"
    values: list["expr"]

    def __init__(
        self,
        op: "boolop",
        values: list["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.op = op
        self.values = values


class BinOp(expr):
    left: "expr"
    op: "operator"
    right: "expr"

    def __init__(
        self,
        left: "expr",
        op: "operator",
        right: "expr",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.left = left
        self.op = op
        self.right = right


class UnaryOp(expr):
    op: "unaryop"
    operand: "expr"

    def __init__(
        self,
        op: "unaryop",
        operand: "expr",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.op = op
        self.operand = operand


class Lambda(expr):
    args: "arguments"
    body: "expr"

    def __init__(
        self,
        args: "arguments",
        body: "expr",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.args = args
        self.body = body


class IfExp(expr):
    test: "expr"
    body: "expr"
    orelse: "expr"

    def __init__(
        self,
        test: "expr",
        body: "expr",
        orelse: "expr",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.test = test
        self.body = body
        self.orelse = orelse


class Dict(expr):
    keys: list[Optional["expr"]]
    values: list["expr"]

    def __init__(
        self,
        keys: list[Optional["expr"]],
        values: list["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.keys = keys
        self.values = values


class Set(expr):
    elts: list["expr"]

    def __init__(
        self,
        elts: list["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.elts = elts


class ListComp(expr):
    elt: "expr"
    generators: list["comprehension"]

    def __init__(
        self,
        elt: "expr",
        generators: list["comprehension"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.elt = elt
        self.generators = generators


class SetComp(expr):
    elt: "expr"
    generators: list["comprehension"]

    def __init__(
        self,
        elt: "expr",
        generators: list["comprehension"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.elt = elt
        self.generators = generators


class DictComp(expr):
    key: "expr"
    value: "expr"
    generators: list["comprehension"]

    def __init__(
        self,
        key: "expr",
        value: "expr",
        generators: list["comprehension"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.key = key
        self.value = value
        self.generators = generators


class GeneratorExp(expr):
    elt: "expr"
    generators: list["comprehension"]

    def __init__(
        self,
        elt: "expr",
        generators: list["comprehension"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.elt = elt
        self.generators = generators


class Yield(expr):
    value: Optional["expr"]

    def __init__(
        self,
        value: Optional["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value


class YieldFrom(expr):
    value: "expr"

    def __init__(
        self,
        value: "expr",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value


class Compare(expr):
    left: "expr"
    ops: list["cmpop"]
    comparators: list["expr"]

    def __init__(
        self,
        left: "expr",
        ops: list["cmpop"],
        comparators: list["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.left = left
        self.ops = ops
        self.comparators = comparators


class Call(expr):
    func: "expr"
    args: list["expr"]
    keywords: list["keyword"]

    def __init__(
        self,
        func: "expr",
        args: list["expr"],
        keywords: list["keyword"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.func = func
        self.args = args
        self.keywords = keywords


class FormattedValue(expr):
    value: "expr"
    conversion: int
    format_spec: Optional["expr"]

    def __init__(
        self,
        value: "expr",
        conversion: int,
        format_spec: Optional["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value
        self.conversion = conversion
        self.format_spec = format_spec


class JoinedStr(expr):
    values: list["expr"]

    def __init__(
        self,
        values: list["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.values = values


class Constant(expr):
    value: Any
    kind: Optional[str]

    def __init__(
        self,
        value: Any,
        kind: Optional[str],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value
        self.kind = kind


class NamedExpr(expr):
    target: "expr"
    value: "expr"

    def __init__(
        self,
        target: "expr",
        value: "expr",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.target = target
        self.value = value


class Attribute(expr):
    value: "expr"
    attr: str
    ctx: "expr_context"

    def __init__(
        self,
        value: "expr",
        attr: str,
        ctx: "expr_context",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value
        self.attr = attr
        self.ctx = ctx


class Subscript(expr):
    value: "expr"
    slice: "expr"
    ctx: "expr_context"

    def __init__(
        self,
        value: "expr",
        slice: "expr",
        ctx: "expr_context",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value
        self.slice = slice
        self.ctx = ctx


class Starred(expr):
    value: "expr"
    ctx: "expr_context"

    def __init__(
        self,
        value: "expr",
        ctx: "expr_context",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value
        self.ctx = ctx


class Name(expr):
    id: str
    ctx: "expr_context"

    def __init__(
        self,
        id: str,
        ctx: "expr_context",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.id = id
        self.ctx = ctx


class List(expr):
    elts: list["expr"]
    ctx: "expr_context"

    def __init__(
        self,
        elts: list["expr"],
        ctx: "expr_context",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.elts = elts
        self.ctx = ctx


class Tuple(expr):
    elts: list["expr"]
    ctx: "expr_context"

    def __init__(
        self,
        elts: list["expr"],
        ctx: "expr_context",
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.elts = elts
        self.ctx = ctx


class Slice(AST):
    lower: Optional["expr"]
    upper: Optional["expr"]
    step: Optional["expr"]

    def __init__(
        self, lower: Optional["expr"], upper: Optional["expr"], step: Optional["expr"]
    ) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.step = step


class expr_context(AST):

    def __init__(self) -> None:
        super().__init__()


class Del(expr_context):

    def __init__(self) -> None:
        super().__init__()


class Load(expr_context):

    def __init__(self) -> None:
        super().__init__()


class Store(expr_context):

    def __init__(self) -> None:
        super().__init__()


class boolop(AST):

    def __init__(self) -> None:
        super().__init__()


class And(boolop):

    def __init__(self) -> None:
        super().__init__()


class Or(boolop):

    def __init__(self) -> None:
        super().__init__()


class operator(AST):

    def __init__(self) -> None:
        super().__init__()


class Add(operator):

    def __init__(self) -> None:
        super().__init__()


class BitAnd(operator):

    def __init__(self) -> None:
        super().__init__()


class BitOr(operator):

    def __init__(self) -> None:
        super().__init__()


class BitXor(operator):

    def __init__(self) -> None:
        super().__init__()


class Div(operator):

    def __init__(self) -> None:
        super().__init__()


class FloorDiv(operator):

    def __init__(self) -> None:
        super().__init__()


class LShift(operator):

    def __init__(self) -> None:
        super().__init__()


class Mod(operator):

    def __init__(self) -> None:
        super().__init__()


class Mult(operator):

    def __init__(self) -> None:
        super().__init__()


class MatMult(operator):

    def __init__(self) -> None:
        super().__init__()


class Pow(operator):

    def __init__(self) -> None:
        super().__init__()


class RShift(operator):

    def __init__(self) -> None:
        super().__init__()


class Sub(operator):

    def __init__(self) -> None:
        super().__init__()


class unaryop(AST):

    def __init__(self) -> None:
        super().__init__()


class Invert(unaryop):

    def __init__(self) -> None:
        super().__init__()


class Not(unaryop):

    def __init__(self) -> None:
        super().__init__()


class UAdd(unaryop):

    def __init__(self) -> None:
        super().__init__()


class USub(unaryop):

    def __init__(self) -> None:
        super().__init__()


class cmpop(AST):

    def __init__(self) -> None:
        super().__init__()


class Eq(cmpop):

    def __init__(self) -> None:
        super().__init__()


class Gt(cmpop):

    def __init__(self) -> None:
        super().__init__()


class GtE(cmpop):

    def __init__(self) -> None:
        super().__init__()


class In(cmpop):

    def __init__(self) -> None:
        super().__init__()


class Is(cmpop):

    def __init__(self) -> None:
        super().__init__()


class IsNot(cmpop):

    def __init__(self) -> None:
        super().__init__()


class Lt(cmpop):

    def __init__(self) -> None:
        super().__init__()


class LtE(cmpop):

    def __init__(self) -> None:
        super().__init__()


class NotEq(cmpop):

    def __init__(self) -> None:
        super().__init__()


class NotIn(cmpop):

    def __init__(self) -> None:
        super().__init__()


class comprehension(AST):
    target: "expr"
    iter: "expr"
    ifs: list["expr"]
    is_async: int

    def __init__(
        self, target: "expr", iter: "expr", ifs: list["expr"], is_async: int
    ) -> None:
        super().__init__()
        self.target = target
        self.iter = iter
        self.ifs = ifs
        self.is_async = is_async


class excepthandler(AST):

    def __init__(self) -> None:
        super().__init__()


class ExceptHandler(excepthandler):
    type: Optional["expr"]
    name: Optional[str]
    body: list["stmt"]

    def __init__(
        self, type: Optional["expr"], name: Optional[str], body: list["stmt"]
    ) -> None:
        super().__init__()
        self.type = type
        self.name = name
        self.body = body


class arguments(AST):
    args: list["arg"]
    vararg: Optional["arg"]
    kwonlyargs: list["arg"]
    kw_defaults: list[Optional["expr"]]
    kwarg: Optional["arg"]
    defaults: list["expr"]
    posonlyargs: list["arg"]

    def __init__(
        self,
        args: list["arg"],
        vararg: Optional["arg"],
        kwonlyargs: list["arg"],
        kw_defaults: list[Optional["expr"]],
        kwarg: Optional["arg"],
        defaults: list["expr"],
        posonlyargs: list["arg"],
    ) -> None:
        super().__init__()
          
        # pylint: disable=too-many-arguments

        self.kw_defaults = kw_defaults
        self.kwonlyargs = kwonlyargs
        self.kwarg = kwarg

        self.posonlyargs = posonlyargs
        self.defaults = defaults
        self.vararg = vararg
        self.args = args


class arg(AST):
    arg: str
    annotation: Optional["expr"]
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int

    def __init__(
        self,
        arg: str,
        annotation: Optional["expr"],
        lineno: int,
        col_offset: int,
        end_lineno: int,
        end_col_offset: int,
    ) -> None:
        super().__init__()
        self.arg = arg
        self.annotation = annotation
        self.lineno = lineno
        self.col_offset = col_offset
        self.end_lineno = end_lineno
        self.end_col_offset = end_col_offset


class keyword(AST):
    arg: Optional[str]
    value: "expr"

    def __init__(self, arg: Optional[str], value: "expr") -> None:
        super().__init__()
        self.arg = arg
        self.value = value


class alias(AST):
    name: str
    asname: Optional[str]

    def __init__(self, name: str, asname: Optional[str]) -> None:
        super().__init__()
        self.name = name
        self.asname = asname


class withitem(AST):
    context_expr: "expr"
    optional_vars: Optional["expr"]

    def __init__(self, context_expr: "expr", optional_vars: Optional["expr"]) -> None:
        super().__init__()
        self.context_expr = context_expr
        self.optional_vars = optional_vars
