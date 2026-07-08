# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from .pir import Call, Block, Function, Scope, BuildContext, Jump
from .pir import ReturnSignal, BreakSignal, ContinueSignal
from .op_registry import dispatch
from .. import pypto_impl


def dispatch_call(call: Call, scope: Scope, ctx: BuildContext):
    callee = scope.resolve(call.callee)
    args = tuple(scope.resolve(v) for v in call.args)
    kwargs = {k: scope.resolve(v) for k, v in call.kwargs}
    if isinstance(callee, Function):
        ret = call_function(callee, args, kwargs, ctx)
    else:
        ret = dispatch(callee, ctx, *args, **kwargs)
    if call.result is not None:
        scope.varmap[call.result.id] = ret

    pypto_impl.SetSpan(call.span.filename, call.span.begin_line)
    ctx.emit_tensor_stmts()
    pypto_impl.ClearSpan()


def call_function(func: Function, args: tuple, kwargs: dict, ctx: BuildContext):
    parent = Scope.current()
    local_names = sorted(set(func.load_vars) | set(func.store_vars))
    scope = Scope(local_names, parent=parent)
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
                scope[name] = parent.resolve(defval)
        try:
            dispatch_block(func.body, True)
        except ReturnSignal as sig:
            return sig.value
    return None


def dispatch_block(block: Block, is_static: bool):
    scope = Scope.current()
    ctx = BuildContext.current()
    for call in block.calls:
        with ctx.change_span(call.span):
            dispatch_call(call, scope, ctx)
    if is_static:
        if block.jump is Jump.CONTINUE:
            raise ContinueSignal
        if block.jump is Jump.BREAK:
            raise BreakSignal
        if block.jump is Jump.RETURN:
            raise ReturnSignal(ctx.wrap(scope["$retval"]))
