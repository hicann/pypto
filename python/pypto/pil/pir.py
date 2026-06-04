# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import threading
import enum
import textwrap
import inspect
from typing import Optional, Any, Union, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

from pypto import ir
from .op_registry import dispatch


@dataclass
class LoopRange:
    start: ir.Expr
    stop: ir.Expr
    step: ir.Expr


@dataclass(frozen=True)
class Value:
    id: int

    def __str__(self) -> str:
        return f"%{self.id}"


Operand = Union[Value, Any]


class Formatter:
    def __init__(self):
        self.blocks = []

    def format(self, x: Operand) -> str:
        if isinstance(x, Block):
            self.blocks.append(x)
            return f"^{x.id}"
        elif isinstance(x, list):
            return f"[{', '.join(self.format(x) for x in x)}]"
        elif isinstance(x, tuple):
            return f"({', '.join(self.format(x) for x in x)})"
        elif callable(x):
            return f"@{x.__name__}"
        else:
            return f"{x}"


class Jump(enum.Enum):
    END_BRANCH = 1  # used in if_else
    RETURN = 2
    CONTINUE = 3
    BREAK = 4


class LoopKind(enum.Enum):
    FOR = 0
    WHILE = 1


@dataclass
class Call:
    result: Optional[Value]
    callee: Operand
    args: tuple[Operand, ...]
    kwargs: dict[str, Operand]
    span: ir.Span

    def __str__(self) -> str:
        fmt = Formatter()
        result_str = f"{self.result} = " if self.result else ""
        line_info = f"  # Line {self.span.begin_line}"
        args_all = [fmt.format(x) for x in self.args] + [
            f"{k}={fmt.format(v)}" for k, v in self.kwargs
        ]
        args_str = ", ".join(args_all)
        block_str = "".join(textwrap.indent(f"\n{b}", "  ") for b in fmt.blocks)
        return (
            f"{result_str}{fmt.format(self.callee)}({args_str}){line_info}{block_str}"
        )


@dataclass
class Block:
    id: int
    args: tuple[Operand, ...]
    calls: list[Call]
    result: Operand
    jump: Optional[Jump] = None
    jump_loc: ir.Span = ir.Span.unknown()
    store_names: set[str] = field(default_factory=set)
    load_names: set[str] = field(default_factory=set)
    span: ir.Span = ir.Span.unknown()

    def __str__(self):
        arg_str = ", ".join(str(p) for p in self.args)
        stmt_str = "".join(f"\n{c}" for c in self.calls)
        if self.jump is not None:
            stmt_str += f"\n{self.jump.name}  # Line {self.jump_loc.begin_line}"
        stmt_str = textwrap.indent(stmt_str, "    ")
        return f"^{self.id}({arg_str}):{stmt_str}"


@dataclass
class Function:
    # Function name
    name: str

    # source location
    span: ir.Span

    # Function signature
    signature: inspect.Signature

    # Function body
    body: Block

    # Store variables
    load_vars: tuple[str, ...]

    # Store variables
    store_vars: tuple[str, ...]

    # Global variables
    global_vars: tuple[str, ...]

    # Global values
    global_values: tuple[Any, ...]


class ReturnSignal(Exception):
    pass


class BuildContext(ir.IRBuilder):
    def __init__(self, span: ir.Span):
        super().__init__()
        self.parent = None
        self.span = span

    def __enter__(self):
        self.parent = _current.build_context
        _current.build_context = self
        return self

    def __exit__(self, exc_type, exc, tb):
        _current.build_context = self.parent

    @staticmethod
    def current() -> "BuildContext":
        if _current.build_context is None:
            raise ValueError("BuildContext is not initialized")
        return _current.build_context

    @contextmanager
    def change_span(self, span: ir.Span):
        old_span, self.span = self.span, span
        try:
            yield
        finally:
            self.span = old_span

    def create_var(self, name: str, type_: Optional[ir.Type], span: ir.Span):
        return ir.Var(name, type_, span=span)

    def create_none(self, span: ir.Span):
        return ir.Var("None", ir.UnknownType.get(), span=span)


class InsertPoint:
    ctx: BuildContext

    def __init__(self, body: ir.SeqStmts):
        self.insert_point = ir.InsertPoint(body)
        self.ctx: BuildContext

    def __enter__(self):
        self.ctx = BuildContext.current()
        self.ctx.set_insert_point(self.insert_point)

    def __exit__(self, exc_type, exc, tb):
        self.ctx.clear_insert_point()


class Scope:
    def __init__(self, local_vars: list[str], parent: Optional["Scope"] = None):
        self.locals: dict[str, Union[ir.Var, Any]] = {name: None for name in local_vars}
        self.parent = parent
        self.varmap = {}
        self.eval = True

    def __getitem__(self, name: str) -> ir.Var:
        var = self.locals[name]
        if var is None:
            if self.parent:
                var = self.parent[name]
            else:
                raise ValueError(f"Variable {name} is not initialized")
        return var

    def __setitem__(self, key: str, value: Union[ir.Var, Any]):
        self.locals[key] = value

    @staticmethod
    def store(name: str, value: Union[ir.Var, Any]):
        scope = Scope.current()
        scope[name] = value

    @staticmethod
    def load(name: str) -> ir.Var:
        scope = Scope.current()
        return scope[name]

    @staticmethod
    def current() -> "Scope":
        if _current.scope is None:
            raise ValueError("Scope is not initialized")
        return _current.scope

    @contextmanager
    def make_current(self):
        old, _current.scope = _current.scope, self
        try:
            yield
        finally:
            _current.scope = old


class LoopStatus:
    def __init__(self, mode: bool = True, names: Optional[list[str]] = None):
        if names is None:
            names = []
        self.static_eval = mode
        self.return_var_names = names
        self.jump: Optional[Jump] = None

    @staticmethod
    def current() -> "LoopStatus":
        return _current.loop_status

    @contextmanager
    def make_current(self):
        old, _current.loop_status = _current.loop_status, self
        try:
            yield self
        finally:
            _current.loop_status = old


class _Current(threading.local):
    scope: Optional[Scope] = None
    build_context: Optional[BuildContext] = None
    loop_status: LoopStatus = LoopStatus()


_current = _Current()


def _resolve(val, scope: Scope):
    if isinstance(val, Value):
        return scope.varmap[val.id]
    elif isinstance(val, tuple):
        return tuple(_resolve(v, scope) for v in val)
    elif isinstance(val, list):
        return list(_resolve(v, scope) for v in val)
    return val


def dispatch_call(call: Call, scope: Scope, ctx: BuildContext):
    callee = _resolve(call.callee, scope)
    args = tuple(_resolve(v, scope) for v in call.args)
    kwargs = {k: _resolve(v, scope) for k, v in call.kwargs}
    ret = dispatch(callee, ctx, *args, **kwargs)
    if call.result is not None:
        scope.varmap[call.result.id] = ret
    ctx.emit_tensor_stmts()


def dispatch_block(block: Block):
    scope = Scope.current()
    ctx = BuildContext.current()
    for call in block.calls:
        with ctx.change_span(call.span):
            dispatch_call(call, scope, ctx)
        lstatus = LoopStatus.current()
        if lstatus.jump is not None:
            break
    # Propagate the block's own jump if no sub-call already set one
    lstatus = LoopStatus.current()
    if lstatus.static_eval and lstatus.jump is None and block.jump is not Jump.END_BRANCH:
        lstatus.jump = block.jump
