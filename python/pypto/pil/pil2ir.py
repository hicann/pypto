# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import inspect
import pypto

from .parser import ast2pil
from .pir import Function, BuildContext, InsertPoint, Scope, ReturnSignal
from .dispatcher import dispatch_block
from .op_registry import dispatch
from . import ops
from ..ir import SeqStmts


def pil2ir(func: Function, args: dict):
    scope = Scope(sorted(set(func.load_vars) | set(func.global_vars)))

    # Pre-populate globals into scope (all of them, so nested functions can
    # resolve module globals/builtins via the scope parent chain)
    for name, val in zip(func.global_vars, func.global_values):
        scope[name] = val

    func_args = []
    body = SeqStmts(func.span)
    with BuildContext(func.span) as ctx, InsertPoint(body), scope.make_current():
        # Store function arguments
        for key, val in args.items():
            if isinstance(val, pypto.Tensor):
                var = ctx.create_var_like(key, val.logical_tensor())
                func_args.append(var)
            dispatch('pil.store', ctx, key, val)
        try:
            dispatch_block(func.body, True)
        except ReturnSignal:
            pass

    return ctx.create_function(func.name, func_args, [], body, func.span)


def compile(pyfunc, *args, **kwargs):
    sig = inspect.signature(pyfunc)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    # c++ addOperation still depends on function
    func = ast2pil(pyfunc)

    all_args = {}
    tensor_args = []
    for key, val in bound.arguments.items():
        all_args[key] = val
        if isinstance(val, pypto.Tensor):
            tensor_args.append(val)

    pypto.pypto_impl.Reset()
    with pypto.function("__entry__", *tensor_args):
        return pil2ir(func, all_args)
