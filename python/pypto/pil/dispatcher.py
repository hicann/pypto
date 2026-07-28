# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from functools import lru_cache
import inspect
import logging
import os
import sys
import sysconfig
import typing

from .. import pypto_impl
from .op_registry import dispatch
from .parser import ast2pil
from .pir import (
    Block,
    BreakSignal,
    BuildContext,
    Call,
    CollectContext,
    ContinueSignal,
    DoubleStarred,
    Function,
    Jump,
    LoopRange,
    ReturnSignal,
    Scope,
    _current,
)

# used to skip "std python package".
_STDLIB_DIRS = (sysconfig.get_paths().get("stdlib", ""), sysconfig.get_paths().get("platstdlib", ""))


def _is_stdlib(mod_name: str) -> bool:
    mod = sys.modules.get(mod_name)
    if mod is None:
        return False
    f = getattr(mod, "__file__", "")
    if not f:
        # built-in module (no source file) -> skip
        return True
    f = os.path.normpath(f)
    return any(f.startswith(d) for d in _STDLIB_DIRS if d)


def _is_pypto(mod_name: str) -> bool:
    return mod_name == "pypto" or mod_name.startswith("pypto.")


@lru_cache(maxsize=None)
def _get_or_compile(pyfunc):
    """Return a cached pil Function for a user-defined helper, else `None`."""
    if not inspect.isfunction(pyfunc):
        return None

    mod = getattr(pyfunc, "__module__", "")
    if not mod or _is_pypto(mod) or _is_stdlib(mod):
        return None

    try:
        return ast2pil(pyfunc, entry_point=False)
    except Exception as e:
        logging.error("Failed to compile %s: %s", pyfunc.__name__, e)
        return None


def dispatch_call(call: Call, scope: Scope, ctx: BuildContext, callback=None):
    callee = scope.resolve(call.callee)
    args = tuple(scope.resolve(call.args))

    kwargs = {}
    for k, v in call.kwargs:
        if isinstance(v, DoubleStarred):
            kwargs.update(scope.resolve(v.value))
        else:
            kwargs[k] = scope.resolve(v)

    if isinstance(callee, Function):
        ret = call_function(callee, args, kwargs, ctx)
    else:
        func = _get_or_compile(callee)
        if func is not None:
            ret = call_function(func, args, kwargs, ctx)
        else:
            if callback is None:
                ret = dispatch(callee, ctx, *args, **kwargs)
            else:
                ret = callback(*args, **kwargs)
    if call.result is not None:
        scope.varmap[call.result.id] = ret

    pypto_impl.SetSpan(call.span.filename, call.span.begin_line)
    ctx.emit_tensor_stmts()
    pypto_impl.ClearSpan()


def call_function(func: Function, args: tuple, kwargs: dict, ctx: BuildContext):
    caller = Scope.current()
    if func.global_vars:
        # Standalone function (built by ast2pil from a real Python function
        root = Scope(list(func.global_vars))
        for name, val in zip(func.global_vars, func.global_values):
            root[name] = val
    else:
        # Inline nested def/lambda/comprehension: lowered within the caller, so
        # its free names resolve lexically through the caller's scope chain.
        root = caller

    local_names = sorted(set(func.load_vars) | set(func.store_vars))
    scope = Scope(local_names, parent=root)
    with scope.make_current():
        supplied = set()
        for name, val in zip(func.params, args):
            scope[name] = val
            supplied.add(name)
        for name, val in kwargs.items():
            scope[name] = val
            supplied.add(name)
        # apply default values for any params not supplied by the call
        for name, defval in zip(func.params, func.param_defaults):
            if name not in supplied and defval is not None:
                scope[name] = caller.resolve(defval)
        try:
            if isinstance(ctx, CollectContext):
                collect(func.body)
            else:
                dispatch_block(func.body, True)
        except ReturnSignal as sig:
            return sig.value
    return None


def block_jump(scope, ctx, block: Block):
    if block.jump is Jump.CONTINUE:
        raise ContinueSignal
    if block.jump is Jump.BREAK:
        raise BreakSignal
    if block.jump is Jump.RETURN:
        raise ReturnSignal(ctx.wrap(scope["$retval"]))
    if block.jump is Jump.END_BRANCH and block.result is not None:
        return scope.resolve(block.result)


def dispatch_block(block: Block, is_static: bool):
    scope = Scope.current()
    ctx = BuildContext.current()

    for call in block.calls:
        with ctx.change_span(call.span):
            dispatch_call(call, scope, ctx)

    return block_jump(scope, ctx, block) if is_static else None


class Collector:
    def __init__(self):
        self.scope = Scope.current()
        self.ctx = BuildContext.current()

    def visit_call(self, call: Call, block: Block):
        callee = call.callee
        if callee == "pil.loop":
            dispatch_call(call, self.scope, self.ctx, self.visit_for)
        elif callee == "pil.if_else":
            dispatch_call(call, self.scope, self.ctx, self.visit_if)
        else:
            dispatch_call(call, self.scope, self.ctx)
            if callee == "pil.store":
                name = typing.cast(str, call.args[0])
                block.store_names.add(name)
            elif callee == "pil.load":
                name = typing.cast(str, call.args[0])
                block.load_names.add(name)

    def visit(self, block: Block, is_static: bool = True):
        prev_block = _current.collector_block
        _current.collector_block = block

        try:
            for call in block.calls:
                self.visit_call(call, block)
        finally:
            _current.collector_block = prev_block

        if prev_block:
            prev_block.store_names.update(block.store_names)
            prev_block.load_names.update(block.load_names)

        return block_jump(self.scope, self.ctx, block) if is_static else None

    def _visit_for(self, body: Block, iterator):
        scope = Scope.current()
        loop_var = body.args[0]
        for item in iterator:
            scope.varmap[loop_var.id] = item
            try:
                self.visit(body)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _visit_while(self, body: Block):
        while True:
            try:
                self.visit(body)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _visit_dyn_for(self, body: Block, loop: LoopRange):
        scope = Scope.current()
        scope["_loop_begin"] = loop.start
        scope["_loop_end"] = loop.stop

        loop_val = body.args[0]
        loop_var = self.ctx.create_scalar_var("idx")

        if loop.batch:
            start, stop = loop.start, loop.stop
            for factor in loop.unroll_list:
                if factor == 1:
                    loop.stop = stop
                else:
                    step = factor * loop.step
                    loop.stop = start + (stop - start) // step * step
                    scope.varmap[loop_val.id] = (loop_var, step)
                    self.visit(body, is_static=False)
                    loop.start = loop.stop
        else:
            scope.varmap[loop_val.id] = loop_var
            self.visit(body, is_static=False)

    def visit_for(self, body: Block, loop):
        if isinstance(loop, LoopRange):
            self._visit_dyn_for(body, loop)
        elif loop is not None:
            self._visit_for(body, loop)
        else:
            self._visit_while(body)

    def visit_if(self, cond, then_body, else_body):
        if isinstance(cond, pypto_impl.SymbolicScalar):
            cond = cond.simplify()
            if cond.is_concrete():
                body = then_body if cond.concrete() else else_body
                return self.visit(body)
            else:
                self.visit(then_body, is_static=False)
                self.visit(else_body, is_static=False)
                return None
        else:
            body = then_body if cond else else_body
            return self.visit(body)

    @classmethod
    def mark_store(cls, tensor):
        block = _current.collector_block
        if block is None:
            return
        s = Scope.current()
        for name, val in s.locals.items():
            if val is tensor:
                block.store_names.add(name)


def collect(block: Block):
    # A function invocation starts a fresh collection tree. Reset the
    # enclosing collector block so the callee's store/load names don't
    # propagate into the caller's block
    prev_block = _current.collector_block
    _current.collector_block = None
    try:
        collector = Collector()
        collector.visit(block)
    finally:
        _current.collector_block = prev_block
