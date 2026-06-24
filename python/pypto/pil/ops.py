# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import operator
from typing import Any, Optional, Union

import pypto
from pypto import ir

from .pir import Value, Block, LoopRange, Jump
from .pir import BuildContext, InsertPoint, Scope, LoopStatus, dispatch_block, ReturnSignal
from .op_registry import impl


def has_scalar(values: list) -> bool:
    return any(isinstance(v, ir.Var) and isinstance(v.type, ir.ScalarType) for v in values)


def _to_ir_type(val: Any, ctx: BuildContext) -> ir.Expr:
    if val is None:
        return ctx.none()
    if isinstance(val, int):
        return ctx.create_const_int(val).as_expr()
    elif isinstance(val, pypto.SymbolicScalar):
        return val.as_expr()
    elif isinstance(val, pypto.Tensor):
        if (val.is_empty()):
            return ctx.none()
        return val.logical_tensor()
    elif isinstance(val, (list, tuple)):
        return ir.MakeTuple([_to_ir_type(v, ctx) for v in val], ir.Span.unknown())
    else:
        raise TypeError(f"Invalid type {type(val)} for wrap")


def _from_ir_type(val: Any):
    if isinstance(val, (ir.ConstInt, ir.ConstFloat, ir.ConstBool)):
        return val.value
    if isinstance(val, ir.Expr):
        if isinstance(val.type, ir.ScalarType):
            return pypto.SymbolicScalar(val)
        elif isinstance(val.type, ir.LogicalTensorType):
            return pypto.Tensor.from_logical_tensor(val)
        elif isinstance(val.type, ir.UnknownType):
            return None
        else:
            raise TypeError(f"Invalid type {type(val)} for unwrap")
    return val

# ---- Compile-time ops ----


@impl("pil.const")
def const_impl(ctx, value):
    return value


@impl("pil.load")
def load_impl(ctx, name):
    return Scope.load(name)


@impl("pil.store")
def store_impl(ctx, name, val):
    Scope.store(name, val)


@impl(operator.sub, partial=True)
@impl(operator.add, partial=True)
@impl(operator.mul, partial=True)
@impl(operator.truediv, partial=True)
@impl(operator.mod, partial=True)
@impl(operator.pow, partial=True)
@impl(operator.floordiv, partial=True)
@impl(operator.or_, partial=True)
@impl(operator.xor, partial=True)
@impl(operator.and_, partial=True)
@impl(operator.lshift, partial=True)
@impl(operator.rshift, partial=True)
@impl(operator.matmul, partial=True)
def binary_impl(ctx, op, x, y):
    return op(x, y)


@impl(operator.neg, partial=True)
@impl(operator.pos, partial=True)
@impl(operator.invert, partial=True)
@impl(operator.not_, partial=True)
def unary_impl(ctx, op, x):
    return op(x)


# ---- Comparison ----

@impl(operator.eq, partial=True)
@impl(operator.ne, partial=True)
@impl(operator.lt, partial=True)
@impl(operator.le, partial=True)
@impl(operator.gt, partial=True)
@impl(operator.ge, partial=True)
def compare_impl(ctx, op, x, y):
    return op(x, y)


# ---- Attribute / index ----

@impl(getattr)
def getattr_impl(ctx, obj, attr):
    return getattr(obj, attr)


# ---- Collection construction ----

@impl(list)
def list_impl(ctx, items):
    return list(items)


@impl(tuple)
def tuple_impl(ctx, items):
    return tuple(items)


@impl(operator.getitem)
def getitem_impl(ctx, obj, key):
    return obj[key]


@impl(min)
def min_impl(ctx, *args):
    if has_scalar(list(args)) and len(args) == 2:
        return pypto.min(args[0], args[1])
    return min(args)


@impl(max)
def max_impl(ctx, *args):
    if has_scalar(list(args)) and len(args) == 2:
        return pypto.max(args[0], args[1])
    return max(args)

# ---- Control flow ----


def _resolve(val, scope):
    if isinstance(val, Value):
        return scope.varmap.get(val.id)
    elif isinstance(val, tuple):
        return tuple(_resolve(v, scope) for v in val)
    elif isinstance(val, list):
        return list(_resolve(v, scope) for v in val)
    return val


@impl(pypto.loop)
def pypto_loop_impl(ctx, *args, name: str = ""):
    nargs = len(args)
    if nargs == 1:
        start, stop, step = 0, args[0], 1
    elif nargs == 2:
        start, stop, step = args[0], args[1], 1
    elif nargs == 3:
        start, stop, step = args
    else:
        raise TypeError(
            f"loop() takes 1 to 3 positional arguments but {nargs} were given")
    return LoopRange(
        start=_to_ir_type(start, ctx),
        stop=_to_ir_type(stop, ctx),
        step=_to_ir_type(step, ctx),
    )


def _add_jump_stmt(ctx: BuildContext, jump, operands: Optional[list[ir.Expr]] = None):
    if operands is None:
        operands = []
    scope = Scope.current()
    names = LoopStatus.current().return_var_names

    if jump == Jump.BREAK:
        operands = [_to_ir_type(scope[name], ctx) for name in names]
        stmt = ctx.create_break_stmt(operands, ctx.span)
    elif jump == Jump.CONTINUE:
        operands = [_to_ir_type(scope[name], ctx) for name in names]
        stmt = ctx.create_continue_stmt(operands, ctx.span)
    elif jump == Jump.RETURN:
        stmt = ctx.create_return_stmt(operands, ctx.span)
    elif jump == Jump.END_BRANCH:
        stmt = ctx.create_yield_stmt(operands, ctx.span)
    else:
        raise ValueError(f"Unknown jump type: {jump}")

    ctx.emit(stmt)


def _static_for(body: Block, iterator):
    scope = Scope.current()
    loop_var = body.args[0]
    loop_status = LoopStatus(True)
    with loop_status.make_current() as lstatus:
        for item in iterator:
            scope.varmap[loop_var.id] = item
            lstatus.jump = None
            dispatch_block(body)
            if lstatus.jump == Jump.CONTINUE:
                continue
            if lstatus.jump == Jump.BREAK:
                break
            if lstatus.jump == Jump.RETURN:
                raise ReturnSignal


def _static_while(body: Block):
    loop_status = LoopStatus(True)
    with loop_status.make_current() as lstatus:
        while True:
            lstatus.jump = None
            dispatch_block(body)
            if lstatus.jump == Jump.CONTINUE:
                continue
            elif lstatus.jump == Jump.BREAK:
                break
            elif lstatus.jump == Jump.RETURN:
                raise ReturnSignal


def _for_stmt(body: Block, iterator: LoopRange, ctx: BuildContext):
    scope = Scope.current()
    loop_val = body.args[0]

    # Create loop variable Varß
    loop_var = ctx.create_scalar_var()
    scope.varmap[loop_val.id] = loop_var

    # Save initial values as iterArgs, convert to Vars in scope
    iter_args = []
    return_var_names = []
    for name in body.store_names:
        val = scope.locals.get(name)
        var = ctx.create_var_like(name, _to_ir_type(val, ctx))
        iter_arg = ctx.create_iter_arg(var, initValue=_to_ir_type(val, ctx))
        scope.store(name, _from_ir_type(var))
        iter_args.append(iter_arg)
        return_var_names.append(name)

    # Compile body into Stmt tree via nested IRBuilder
    body_stmt = ir.SeqStmts(body.span)
    loop_status = LoopStatus(False, return_var_names)
    with InsertPoint(body_stmt), ctx.change_span(body.span), loop_status.make_current():
        dispatch_block(body)
        _add_jump_stmt(ctx, body.jump)

    return_vars = []
    for name in return_var_names:
        var = ctx.create_var_like(name, _to_ir_type(scope[name], ctx))
        return_vars.append(var)
        scope.store(name, _from_ir_type(var))

    for_stmt = ctx.create_for_stmt(
        loop_var.as_var(), iterator.start, iterator.stop, iterator.step, iter_args, body_stmt, return_vars, ctx.span,
    )
    ctx.emit(for_stmt)


@impl("pil.loop")
def loop_impl(ctx, body: Block, iterator):
    if isinstance(iterator, LoopRange):
        _for_stmt(body, iterator, ctx)
    elif iterator is not None:
        _static_for(body, iterator)
    else:
        _static_while(body)


def _if_else_stmt(cond, then_block: Block, else_block: Block, ctx: BuildContext):
    scope = Scope.current()

    saved = dict(scope.locals)
    yield_var_names = then_block.store_names | else_block.store_names

    then_body = ir.SeqStmts(then_block.span)
    with InsertPoint(then_body), ctx.change_span(then_block.span):
        dispatch_block(then_block)
        then_yield_vars = [_to_ir_type(scope[name], ctx) for name in yield_var_names]
        _add_jump_stmt(ctx, then_block.jump, list(then_yield_vars))

    scope.locals = dict(saved)
    else_body = ir.SeqStmts(else_block.span)
    with InsertPoint(else_body), ctx.change_span(else_block.span):
        dispatch_block(else_block)
        else_yield_vars = [_to_ir_type(scope[name], ctx) for name in yield_var_names]
        _add_jump_stmt(ctx, else_block.jump, list(else_yield_vars))

    scope.locals = dict(saved)
    yield_vars = []
    for i, name in enumerate(sorted(yield_var_names)):
        if ir.type_equal(then_yield_vars[i], else_yield_vars[i]):
            var = ctx.create_var_like(name, then_yield_vars[i])
        elif isinstance(then_yield_vars[i].type, ir.UnknownType):
            var = ctx.create_var_like(name, else_yield_vars[i])
        elif isinstance(else_yield_vars[i].type, ir.UnknownType):
            var = ctx.create_var_like(name, then_yield_vars[i])
        else:
            raise ValueError(f"Var({name}) then_type={then_yield_vars[i].type}, else_type={else_yield_vars[i].type}")
        yield_vars.append(var)
        scope.store(name, _from_ir_type(var))

    if_stmt = ctx.create_if_stmt(_to_ir_type(cond, ctx), then_body, else_body, yield_vars, ctx.span)
    ctx.emit(if_stmt)


@impl("pil.if_else")
def if_else_impl(ctx, cond, then_block: Block, else_block: Block):
    # Concrete condition: interpret one branch, return early
    if isinstance(cond, pypto.SymbolicScalar):
        cond = cond.simplify()
        if cond.is_concrete():
            block = then_block if cond.concrete() else else_block
            dispatch_block(block)
        else:
            _if_else_stmt(cond, then_block, else_block, ctx)
    else:
        block = then_block if cond else else_block
        dispatch_block(block)


@impl("pil.fstring")
def format_impl(ctx, joined_str):
    ss = []
    for val in joined_str:
        if isinstance(val, tuple):
            val, conv, spec = val
            # Apply conversion first (mirrors Python f-string semantics)
            if conv == ord('a'):
                val = ascii(val)
            elif conv == ord('s'):
                val = str(val)
            elif conv == ord('r'):
                val = repr(val)
            # conv == -1 means no conversion; val stays as-is
            # Then apply format spec (may be "" for no spec)
            val = format(val, spec)
        ss.append(val)
    return "".join(ss)


@impl("pil.assert")
def assert_impl(ctx, cond, msg):
    if msg:
        assert cond, msg
    else:
        assert cond
