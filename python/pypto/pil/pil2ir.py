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


def pil2ir(func: Function, args: dict, tensor_args: list[pypto.Tensor]):
    # make global scope
    root = Scope(list(func.global_vars))

    for name, val in zip(func.global_vars, func.global_values):
        root[name] = val

    scope = Scope(list(func.load_vars), parent=root)
    params = [x.logical_tensor() for x in tensor_args]
    body = SeqStmts(func.span)
    with BuildContext(func.span) as ctx, InsertPoint(body), scope.make_current():
        # Store function arguments
        for key, val in args.items():
            dispatch('pil.store', ctx, key, val)
        try:
            dispatch_block(func.body, True)
        except ReturnSignal:
            pass
        stmt = ctx.create_return_stmt([ctx.unwrap(t) for t in tensor_args], func.span)
        ctx.emit(stmt)
    return ctx.create_function(func.name, params, [], body, func.span)


def compile(pyfunc, *args, **kwargs):
    pypto.pypto_impl.Reset()

    sig = inspect.signature(pyfunc)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    # c++ addOperation still depends on function
    func = ast2pil(pyfunc)

    all_args = {}
    tensor_args = []
    for key, val in bound.arguments.items():
        if isinstance(val, pypto.Tensor):
            val = val.copy(name=key)
            tensor_args.append(val)
        all_args[key] = val

    with pypto.function("__entry__", *tensor_args):
        func_def = pil2ir(func, all_args, tensor_args)
        # funtion input args still need to be valid, it'll be used later by tensor slot
        setattr(func_def, "__args__", tensor_args)
        return func_def
