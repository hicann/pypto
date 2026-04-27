#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Build PIL (Python Intermediate Language) for pypto frontend. Compared with the full python ast,
PIL is a simplified version of python ast, which only contains the necessary information for code generation.
The main purpose of PIL is to simplify the code generation process and improve the performance of code generation.

Simplify rule:
1.  All expr should be replaced by identifier as much as possible
2.  When assigned, only multiple names with starred, single attribute, single subscript
    are allowed in the assignment's lhs and for's target
3.  Lambda is converted to a named FunctionDef
4.  Comprehensions (ListComp, SetComp, GeneratorExp, DictComp) are converted to generator FunctionDef + call
5.  BoolOp, IfExp, Compare chains are converted to if/assign sequences
6.  AugAssign is converted to explicit load + BinOp + store
7.  AnnAssign is converted to annotation expr eval + optional assignment
8.  Assert is converted to if + Assert (preserving __debug__ guard)
9.  With items' optional_vars assignment is placed inside the body
10. Try handlers are unified into a single except Exception with isinstance dispatch

stmt = FunctionDef(identifier name, arguments args,
                    stmt* body, identifier* decorator_list)
        | Assign(expr target, expr value, string? type_comment)
          # target only allow for identifier, attribute and subscript
        | Return(identifier? value)

        | For(identifier target, identifier iter, stmt* body, stmt* orelse, string? type_comment)
        | While(identifier test, stmt* body, stmt* orelse)
        | If(identifier test, stmt* body, stmt* orelse)

        | With(identifier* items, stmt* body, string? type_comment)  # items are context managers, no as-binding
        | Raise(identifier? exc, identifier? cause)
        | Try(stmt* body, (identifier exc_var, stmt* handler_body), stmt* orelse, stmt* finalbody)

        | Assert(identifier test, identifier? msg)

        | Import(alias* names)
        | ImportFrom(identifier? module, alias* names, int? level)

        | Global(identifier* names)
        | Nonlocal(identifier* names)
        | Pass
        | Break
        | Continue

        -- col_offset is the byte offset in the utf8 string the parser uses
        attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

expr =  BinOp(identifier left, operator op, identifier right)
        | UnaryOp(unaryop op, identifier operand)
        | Dict(identifier?* keys, identifier* values)
        | Set(expr* elts) # only allow for identifier and starred
        | Compare(identifier left, cmpop ops, identifier comparators)
        | Call(identifier func, identifier* args, keyword* keywords)

        | Yield(identifier? value)
        | YieldFrom(identifier value)

        | JoinedStr(expr* values)  # only allow for identifier and constant (string)

        | Constant(constant value, string? kind)

        | Name(identifier id)
        | Attribute(identifier value, identifier attr)
        | Subscript(identifier value, expr *slice)

        | Starred(identifier value)
        | List(expr* elts) # Only allow for identifier and starred
        | Tuple(expr* elts) # Only allow for slice and identifier, starred

        | Slice(identifier? lower, identifier? upper, identifier? step)

operator = Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift
                | RShift | BitOr | BitXor | BitAnd | FloorDiv

unaryop = Invert | Not | UAdd | USub

cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
"""

import ast
import re
from collections.abc import Mapping
from typing import Any, Callable, Optional, Union

from pypto.error import FeError

PILExpr = Union[str, ast.Constant]
PILSlice = Union[tuple[Optional[PILExpr], Optional[PILExpr], Optional[PILExpr]], PILExpr]
PILExprOrNone = Optional[PILExpr]

PIL_DEFAULT_PREFIX = '_pil_'


class PILContext:

    def __init__(self, prefix=PIL_DEFAULT_PREFIX):
        self._continue_stack = []
        self._temp_count = 0
        self._prefix = prefix

    @property
    def continue_stack(self) -> list[Optional[tuple[ast.expr, str]]]:
        return self._continue_stack

    def create_temp_identifier(self, *_args, **_kwargs) -> str:
        name = f"{self._prefix}{self._temp_count}"
        self._temp_count += 1
        return name


class PILAttr(Mapping):
    ATTR_LIST = [
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset',
    ]

    def __init__(self, node):
        self._data = {
            attr: 0 if node is None else getattr(node, attr, 0)
            for attr in self.ATTR_LIST
        }

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

NOATTR = PILAttr(None)


class PILBuilder(ast.NodeVisitor):

    def __init__(self, ctx: PILContext = None):
        super().__init__()
        if ctx is None:
            ctx = PILContext()
        self._ctx = ctx

    @property
    def continue_stack(self) -> list[Optional[tuple[ast.expr, str]]]:
        return self._ctx.continue_stack

    def create_temp_identifier(self) -> str:
        return self._ctx.create_temp_identifier()

    def create_attribute(self, node) -> dict:
        result = {
            'lineno': node.lineno,
            'col_offset': node.col_offset,
            'end_lineno': node.end_lineno,
            'end_col_offset': node.end_col_offset,
        }
        return result

    def create_pil_expr(self,
         value: PILExpr,
         ctx: ast.expr_context = ast.Load(),
         node_attr: PILAttr = NOATTR) -> Union[ast.Name, ast.Constant]:
        if isinstance(value, ast.Constant):
            return value
        if isinstance(value, str):
            return self.create_pil_name(value, ctx)
        raise FeError(TypeError(f"Expected ast.Constant or str, but got {type(value).__name__}"))

    def create_pil_maybe_starred(self,
         expr: PILExpr,
         starred: bool,
         ctx=ast.Load(),
         node_attr: PILAttr = NOATTR) -> ast.expr:
        if starred:
            if not isinstance(expr, str):
                raise FeError(TypeError(f"Expected str for starred expr, but got {type(expr).__name__}"))
            return self.create_pil_starred(expr, ctx)
        return self.create_pil_expr(expr, ctx)

    def create_pil_slice(self, slices: list[PILSlice]) -> ast.expr:
        result_slice_tuple = []
        for pil_slice in slices:
            if isinstance(pil_slice, tuple):
                result_slice_expr = ast.Slice(
                    lower=self.create_pil_expr(pil_slice[0]) if pil_slice[0] is not None else None,
                    upper=self.create_pil_expr(pil_slice[1]) if pil_slice[1] is not None else None,
                    step=self.create_pil_expr(pil_slice[2]) if pil_slice[2] is not None else None)
                result_slice_tuple.append(result_slice_expr)
            else:
                index = self.create_pil_expr(pil_slice)
                result_slice_tuple.append(index)

        if len(result_slice_tuple) == 1:
            result_slice = result_slice_tuple[0]
        else:
            result_slice = ast.Tuple(elts=result_slice_tuple, ctx=ast.Load())
        return result_slice

    def create_pil_assign_name(self,
        targets: Union[str, list[tuple[str, bool]], tuple[tuple[str, bool]]],
        value: ast.expr,
        node_attr: PILAttr = NOATTR) -> ast.Assign:
        """
        Parameter:
            targets:
                1. identifier
                2. list of maybe starred identifier
                3. tuple of maybe starred identifier
            value: python native expression
        Emit code format 1:
            targets = value                                    # targets: str
        Emit code format 2:
            [targets_0, *targets_1, targets_2] = value         # targets: list[tuple[str, bool]]
        Emit code format 3:
            (targets_0, *targets_1, targets_2) = value         # targets: tuple[tuple[str, bool]]
        """
        if isinstance(targets, str):
            result_targets = [ast.Name(id=targets, ctx=ast.Store())]
            return ast.Assign(targets=result_targets, value=value, **node_attr)
        else:
            if isinstance(targets, list):
                result_targets = ast.List(
                    [self.create_pil_maybe_starred(name, starred, ctx=ast.Store()) for name, starred in targets],
                    ctx=ast.Store())
            else:
                result_targets = ast.Tuple(
                    [self.create_pil_maybe_starred(name, starred, ctx=ast.Store()) for name, starred in targets],
                    ctx=ast.Store())
            return ast.Assign(targets=[result_targets], value=value, **node_attr)

    def create_pil_assign_identifier(self,
         target_name: str,
         source_expr: PILExpr,
         node_attr: PILAttr = NOATTR) -> ast.Assign:
        """
        Parameter:
            target_name: identifier
            source_expr: identifier or constant
        Emit code:
            target_name = source_expr
        """
        if not isinstance(target_name, str):
            raise FeError(TypeError(f"Expected str for target_name, but got {type(target_name).__name__}"))
        return ast.Assign(
            targets=[ast.Name(id=target_name, ctx=ast.Store())],
            value=self.create_pil_expr(source_expr),
            **node_attr)

    def create_pil_assign_attribute(self,
         target_expr: PILExpr,
         attr_name: str,
         source_expr: PILExpr,
         node_attr: PILAttr = NOATTR) -> ast.Assign:
        """
        Parameter:
            target_name: identifier of the object
            attr_name: identifier (attribute name)
            source_expr: identifier or constant
        Emit code:
            target_name.attr_name = source_expr
        """
        return ast.Assign(
            targets=[ast.Attribute(value=self.create_pil_expr(target_expr),
                                   attr=attr_name, ctx=ast.Store())],
            value=self.create_pil_expr(source_expr),
            **node_attr)

    def create_pil_assign_subscript(self,
         target_expr: PILExpr,
         slices: list[PILSlice],
         source_expr: PILExpr,
         node_attr: PILAttr = NOATTR) -> ast.Assign:
        """
        Parameter:
            target_name: identifier of the object
            slices:
                1. identifier or constant            -- index
                2. tuple (lower, upper, step)        -- slice, each element is identifier, constant or None
            source_expr: identifier or constant
        Emit code:
            target_name[slices] = source_expr
        """
        result_slice = self.create_pil_slice(slices)
        return ast.Assign(
            targets=[ast.Subscript(value=self.create_pil_expr(target_expr),
                                   slice=result_slice, ctx=ast.Store())],
            value=self.create_pil_expr(source_expr),
            **node_attr)

    def create_pil_function_def(self,
         name: str,
         args: ast.arguments,
         body: list[ast.stmt],
         decorator_list: list[PILExpr],
         returns: Optional[ast.expr],
         type_comment: Optional[str],
         node_attr: PILAttr = NOATTR) -> ast.FunctionDef:
        """
        Parameter:
            name: identifier (function name)
            args: arguments node
            body: list of statements
            decorator_list: list of identifier or constant
            returns: return annotation expression or None
            type_comment: type comment string or None
        Emit code example:

            @decorator_list[0]
            @decorator_list[1]
            def name(args) -> returns:
                body
        """
        return ast.FunctionDef(name=name,
             args=args,
             body=body,
             decorator_list=[self.create_pil_expr(d) for d in decorator_list],
             returns=returns,
             type_comment=type_comment,
             **node_attr)

    def create_pil_return(self, expr: PILExprOrNone, node_attr: PILAttr = NOATTR) -> ast.Return:
        """
        Parameter:
            expr: identifier, constant, or None
        Emit code format 1:
            return expr    # expr is not None
        Emit code format 2:
            return         # expr is None
        """
        return ast.Return(value=self.create_pil_expr(expr) if expr is not None else None, **node_attr)

    def create_pil_delete_identifier(self, name: str, node_attr: PILAttr = NOATTR) -> ast.Delete:
        """
        Parameter:
            name: identifier to delete
        Emit code format:
            del name
        """
        return ast.Delete(targets=[ast.Name(id=name, ctx=ast.Del())], **node_attr)

    def create_pil_delete_attribute(self, target_expr: PILExpr, attr: str, node_attr: PILAttr = NOATTR) -> ast.Delete:
        """
        Parameter:
            target_expr: identifier or constant (object whose attribute is deleted)
            attr: attribute name string
        Emit code format:
            del target_expr.attr
        """
        return ast.Delete(targets=[ast.Attribute(value=self.create_pil_expr(target_expr),
                     attr=attr,
                     ctx=ast.Del())],
             **node_attr)

    def create_pil_delete_subscript(self,
         target_expr: PILExpr,
         slice_expr: PILExpr,
         node_attr: PILAttr = NOATTR) -> ast.Delete:
        """
        Parameter:
            target_expr: identifier or constant (object whose item is deleted)
            slice_expr: identifier or constant (subscript key)
        Emit code format:
            del target_expr[slice_expr]
        """
        return ast.Delete(targets=[ast.Subscript(value=self.create_pil_expr(target_expr),
                     slice=self.create_pil_expr(slice_expr),
                     ctx=ast.Del())],
             **node_attr)

    def create_pil_for(self,
         target_name: str,
         iter_expr: PILExpr,
         body: list[ast.stmt],
         orelse: list[ast.stmt],
         type_comment: Optional[str],
         node_attr: PILAttr = NOATTR) -> ast.For:
        """
        Parameter:
            target_name: identifier (loop variable)
            iter_expr: identifier or constant (iterable)
            body: list of statements
            orelse: list of statements (else branch)
            type_comment: type comment string or None
        Emit code:
            for target_name in iter_expr:
                body
            else:
                orelse
        """
        if not isinstance(target_name, str):
            raise FeError(TypeError(f"Expected str for target_name, but got {type(target_name).__name__}"))
        return ast.For(target=ast.Name(id=target_name,
                 ctx=ast.Store()),
             iter=self.create_pil_expr(iter_expr),
             body=body,
             orelse=orelse,
             type_comment=type_comment,
             **node_attr)

    def create_pil_while(self,
         test_expr: PILExpr,
         body: list[ast.stmt],
         orelse: list[ast.stmt],
         node_attr: PILAttr = NOATTR) -> ast.While:
        """
        Parameter:
            test_expr: identifier or constant (loop condition)
            body: list of statements
            orelse: list of statements (else branch)
        Emit code:
            while test_expr:
                body
            else:
                orelse
        """
        return ast.While(test=self.create_pil_expr(test_expr), body=body, orelse=orelse, **node_attr)

    def create_pil_if(self,
         test_expr: PILExpr,
         body: list[ast.stmt],
         orelse: list[ast.stmt],
         node_attr: PILAttr = NOATTR) -> ast.If:
        """
        Parameter:
            test_expr: identifier or constant (condition)
            body: list of statements (then branch)
            orelse: list of statements (else branch)
        Emit code:
            if test_expr:
                body
            else:
                orelse
        """
        return ast.If(test=self.create_pil_expr(test_expr), body=body, orelse=orelse, **node_attr)

    def create_pil_assert(self,
         test_expr: PILExpr,
         msg_value: PILExprOrNone,
         node_attr: PILAttr = NOATTR) -> ast.Assert:
        """
        Parameter:
            test_expr: identifier or constant (assertion condition)
            msg_value: identifier, constant, or None (error message)
        Emit code format 1:
            assert test_expr, msg_value    # msg_value is not None
        Emit code format 2:
            assert test_expr               # msg_value is None
        """
        return ast.Assert(test=self.create_pil_expr(test_expr),
             msg=self.create_pil_expr(msg_value) if msg_value is not None else None,
             **node_attr)

    def create_pil_yield(self, expr: PILExprOrNone, node_attr: PILAttr = NOATTR) -> ast.Yield:
        """
        Parameter:
            expr: identifier, constant, or None
        Emit code format 1:
            yield expr    # expr is not None
        Emit code format 2:
            yield         # expr is None
        """
        if expr is not None:
            return ast.Yield(value=self.create_pil_expr(expr), **node_attr)
        else:
            return ast.Yield(value=None, **node_attr)

    def create_pil_yield_from(self, expr: PILExpr, node_attr: PILAttr = NOATTR) -> ast.YieldFrom:
        """
        Parameter:
            expr: identifier or constant (iterable to delegate to)
        Emit code:
            yield from expr
        """
        return ast.YieldFrom(value=self.create_pil_expr(expr), **node_attr)

    def create_pil_with(self,
         items: list[PILExpr],
         body: list[ast.stmt],
         type_comment: Optional[str],
         node_attr: PILAttr = NOATTR) -> ast.With:
        """
        Parameter:
            items: list of identifier or constant (context managers, no as-binding)
            body: list of statements
            type_comment: type comment string or None
        Emit code example:
            with items[0], items[1]:
                body
        """
        pil_items = [ast.withitem(context_expr=self.create_pil_expr(item), optional_vars=None) for item in items]
        return ast.With(items=pil_items, body=body, type_comment=type_comment, **node_attr)

    def create_pil_raise(self, exc: PILExprOrNone, cause: PILExprOrNone, node_attr: PILAttr = NOATTR) -> ast.Raise:
        """
        Parameter:
            exc: identifier, constant, or None (exception to raise)
            cause: identifier, constant, or None (chained exception)
        Emit code format 1:
            raise exc from cause    # exc and cause are not None
        Emit code format 2:
            raise exc               # cause is None
        Emit code format 3:
            raise                   # exc is None (bare re-raise)
        """
        return ast.Raise(
            exc=self.create_pil_expr(exc) if exc is not None else None,
            cause=self.create_pil_expr(cause) if cause is not None else None,
            **node_attr)

    def create_pil_try(self,
         body: list[ast.stmt],
         handlers: tuple[str,
             list[ast.stmt]],
         orelse: list[ast.stmt],
         finalbody: list[ast.stmt],
         node_attr: PILAttr = NOATTR) -> ast.Try:
        """
        Parameter:
            body: list of statements (try body)
            handlers: (exc_var, handler_body) - single except Exception as exc_var handler
            orelse: list of statements (else branch)
            finalbody: list of statements (finally branch)
        Emit code:
            try:
                body
            except Exception as exc_var:
                handler_body
            else:
                orelse
            finally:
                finalbody
        """
        exc_var, handler_body = handlers
        pil_handler = ast.ExceptHandler(type=self.create_pil_name('Exception'), name=exc_var, body=handler_body)
        return ast.Try(body=body, handlers=[pil_handler], orelse=orelse, finalbody=finalbody, **node_attr)

    def create_pil_import(self, names: list[ast.alias], node_attr: PILAttr = NOATTR) -> ast.Import:
        """
        Parameter:
            names: list of alias nodes
        Emit code example:
            import names[0], names[1]
        """
        return ast.Import(names=names, **node_attr)

    def create_pil_import_from(self,
         module: Optional[str],
         names: list[ast.alias],
         level: Optional[int],
         node_attr: PILAttr = NOATTR) -> ast.ImportFrom:
        """
        Parameter:
            module: identifier (module name) or None
            names: list of alias nodes
            level: relative import level or None
        Emit code example:
            from module import names[0], names[1]
        """
        return ast.ImportFrom(module=module, names=names, level=level, **node_attr)

    def create_pil_global(self, names: list[str], node_attr: PILAttr = NOATTR) -> ast.Global:
        """
        Parameter:
            names: list of identifiers
        Emit code example:
            global names[0], names[1]
        """
        return ast.Global(names=names, **node_attr)

    def create_pil_nonlocal(self, names: list[str], node_attr: PILAttr = NOATTR) -> ast.Nonlocal:
        """
        Parameter:
            names: list of identifiers
        Emit code example:
            nonlocal names[0], names[1]
        """
        return ast.Nonlocal(names=names, **node_attr)

    def create_pil_pass(self, node_attr: PILAttr = NOATTR) -> ast.Pass:
        """
        Emit code:
            pass
        """
        return ast.Pass(**node_attr)

    def create_pil_break(self, node_attr: PILAttr = NOATTR) -> ast.Break:
        """
        Emit code:
            break
        """
        return ast.Break(**node_attr)

    def create_pil_continue(self, node_attr: PILAttr = NOATTR) -> ast.Continue:
        """
        Emit code:
            continue
        """
        return ast.Continue(**node_attr)

    def create_pil_bin_op(self,
         left_expr: PILExpr,
         op: ast.operator,
         right_expr: PILExpr,
         node_attr: PILAttr = NOATTR) -> ast.BinOp:
        """
        Parameter:
            left_expr: identifier or constant
            op: operator (Add, Sub, Mult, ...)
            right_expr: identifier or constant
        Emit code:
            left_expr op right_expr
        """
        return ast.BinOp(left=self.create_pil_expr(left_expr),
             op=op,
             right=self.create_pil_expr(right_expr),
             **node_attr)

    def create_pil_unary_op(self, op: ast.unaryop, operand_expr: PILExpr, node_attr: PILAttr = NOATTR) -> ast.UnaryOp:
        """
        Parameter:
            op: unary operator (Invert, Not, UAdd, USub)
            operand_expr: identifier or constant
        Emit code:
            op operand_expr
        """
        return ast.UnaryOp(op=op, operand=self.create_pil_expr(operand_expr), **node_attr)

    def create_pil_dict(self,
         keys: list[PILExprOrNone],
         values: list[PILExpr],
         node_attr: PILAttr = NOATTR) -> ast.Dict:
        """
        Parameter:
            keys: list of identifier, constant, or None (None means dict unpacking **value)
            values: list of identifier or constant
        Emit code example:
            {keys[0]: values[0], **values[1], keys[2]: values[2]}
        """
        return ast.Dict(
            keys=[self.create_pil_expr(key) if key is not None else None for key in keys],
            values=[self.create_pil_expr(value) for value in values],
            **node_attr)

    def create_pil_set(self, elts: list[tuple[PILExpr, bool]], node_attr: PILAttr = NOATTR) -> ast.Set:
        """
        Parameter:
            elts: list of (identifier or constant, is_starred)
        Emit code example:
            {elts[0], *elts[1], elts[2]}
        """
        return ast.Set(
            elts=[self.create_pil_maybe_starred(elt[0], elt[1]) for elt in elts],
            **node_attr)

    def create_pil_compare(self,
         left_expr: PILExpr,
         op: ast.cmpop,
         comparator_expr: PILExpr,
         node_attr: PILAttr = NOATTR) -> ast.Compare:
        """
        Parameter:
            left_expr: identifier or constant
            op: comparison operator (Eq, NotEq, Lt, ...)
            comparator_expr: identifier or constant
        Emit code:
            left_expr op comparator_expr
        """
        return ast.Compare(left=self.create_pil_expr(left_expr),
             ops=[op],
             comparators=[self.create_pil_expr(comparator_expr)],
             **node_attr)

    def create_pil_call(self,
         func_expr: PILExpr,
         args: list[PILExpr],
         keywords: list[tuple[Optional[str],
                 PILExpr]],
         node_attr: PILAttr = NOATTR) -> ast.Call:
        """
        Parameter:
            func_expr: identifier or constant (callable)
            args: list of identifier or constant (positional arguments)
            keywords: list of (arg_name_or_None, identifier_or_constant) pairs;
                      arg_name is None for **dict expansion
        Emit code example:
            func_expr(args[0], args[1], key=keywords[0][1])
        """
        return ast.Call(
            func=self.create_pil_expr(func_expr),
            args=[self.create_pil_expr(arg) for arg in args],
            keywords=[ast.keyword(arg=kw_arg, value=self.create_pil_expr(kw_value)) for kw_arg, kw_value in keywords],
            **node_attr)

    def create_pil_constant(self, value: object, kind: Optional[str], node_attr: PILAttr = NOATTR) -> ast.Constant:
        """
        Parameter:
            value: constant value (int, float, str, bytes, bool, None, ...)
            kind: string kind marker or None (e.g. 'u' for u-strings)
        Emit code:
            value
        """
        return ast.Constant(value=value, kind=kind, **node_attr)

    def create_pil_name(
        self,
        identifier: str,
        ctx: ast.expr_context = ast.Load(),
        node_attr: PILAttr = NOATTR,
    ) -> ast.Name:
        """
        Parameter:
            identifier: identifier
            ctx: Load, Store, or Del
        Emit code:
            identifier
        """
        return ast.Name(id=identifier, ctx=ctx, **node_attr)

    def create_pil_attribute(self,
         value_expr: PILExpr,
         attr_name: str,
         ctx: ast.expr_context = ast.Load(),
         node_attr: PILAttr = NOATTR) -> ast.Attribute:
        """
        Parameter:
            value_expr: identifier or constant (object)
            attr_name: identifier (attribute name)
            ctx: Load, Store, or Del
        Emit code:
            value_expr.attr_name
        """
        return ast.Attribute(value=self.create_pil_expr(value_expr), attr=attr_name, ctx=ctx, **node_attr)

    def create_pil_subscript(self,
        value_expr: PILExpr,
        slices: list[PILSlice],
         ctx: ast.expr_context = ast.Load(),
         node_attr: PILAttr = NOATTR) -> ast.Subscript:
        """
        Parameter:
            value_expr: identifier or constant (object)
            slices: list of (lower, upper, step) tuples, each element is identifier, constant or None
            ctx: Load, Store, or Del
        Emit code:
            value_expr[lower:upper:step, ...]
        """
        result_slice_tuple = []
        for pil_slice in slices:
            if isinstance(pil_slice, tuple):
                result_slice_expr = ast.Slice(
                    lower=self.create_pil_expr(pil_slice[0]) if pil_slice[0] is not None else None,
                    upper=self.create_pil_expr(pil_slice[1]) if pil_slice[1] is not None else None,
                    step=self.create_pil_expr(pil_slice[2]) if pil_slice[2] is not None else None)
            else:
                result_slice_expr = self.create_pil_expr(pil_slice)
            result_slice_tuple.append(result_slice_expr)
        if len(slices) == 1:
            result_slice = result_slice_tuple[0]
        else:
            result_slice = ast.Tuple(elts=result_slice_tuple, ctx=ast.Load())
        return ast.Subscript(value=self.create_pil_expr(value_expr), slice=result_slice, ctx=ast.Load(), **node_attr)

    def create_pil_starred(self,
         value_expr: PILExpr,
         ctx: ast.expr_context = ast.Load(),
         node_attr: PILAttr = NOATTR) -> ast.Starred:
        """
        Parameter:
            value_expr: identifier or constant
            ctx: Load or Store
        Emit code:
            *value_expr
        """
        return ast.Starred(self.create_pil_expr(value_expr), ctx=ctx, **node_attr)

    def create_pil_list(self,
         elts: list[tuple[PILExpr,
                 bool]],
         ctx: ast.expr_context = ast.Load(),
         node_attr: PILAttr = NOATTR) -> ast.List:
        """
        Parameter:
            elts: list of (identifier or constant, is_starred)
            ctx: Load or Store
        Emit code example:
            [elts[0], *elts[1], elts[2]]
        """
        return ast.List(
            elts=[self.create_pil_maybe_starred(elt[0], elt[1]) for elt in elts],
            ctx=ctx,
            **node_attr)

    def create_pil_tuple(self,
         elts: list[tuple[PILExpr,
                 bool]],
         ctx: ast.expr_context = ast.Load(),
         node_attr: PILAttr = NOATTR) -> ast.Tuple:
        """
        Parameter:
            elts: list of (identifier or constant, is_starred)
            ctx: Load or Store
        Emit code example:
            (elts[0], *elts[1], elts[2])
        """
        return ast.Tuple(
            elts=[self.create_pil_maybe_starred(elt[0], elt[1]) for elt in elts],
            ctx=ctx,
            **node_attr)


class PythonParser(PILBuilder, ast.NodeVisitor):

    def __init__(self, ctx: PILContext):
        super().__init__(ctx)

    @staticmethod
    def _node_name_to_visitor_suffix(name: str) -> str:
        return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()

    def visit_slice_values(
        self,
        slice_expr: Union[ast.Slice, tuple[ast.Slice]],
    ) -> tuple[list[ast.stmt], list[PILSlice]]:
        slice_list = slice_expr.elts if isinstance(slice_expr, ast.Tuple) else [slice_expr]

        slice_stmt_list = []
        pil_slice_list = []
        for s in slice_list:
            if isinstance(s, ast.Slice):
                lower_expr = upper_expr = step_expr = None
                if s.lower is not None:
                    stmts, lower_expr = self.visit(s.lower)
                    slice_stmt_list.extend(stmts)
                if s.upper is not None:
                    stmts, upper_expr = self.visit(s.upper)
                    slice_stmt_list.extend(stmts)
                if s.step is not None:
                    stmts, step_expr = self.visit(s.step)
                    slice_stmt_list.extend(stmts)
                pil_slice_list.append((lower_expr, upper_expr, step_expr))
            else:
                subscript_stmt_list, subscript_expr = self.visit(s)
                slice_stmt_list.extend(subscript_stmt_list)
                pil_slice_list.append(subscript_expr)
        return slice_stmt_list, pil_slice_list

    def visit_lhs(self, target: ast.expr, source_expr: PILExprOrNone) -> list[ast.stmt]:
        if isinstance(target, ast.Name):
            assign_stmts = [self.create_pil_assign_identifier(target.id,
                     source_expr)] if source_expr is not None else []
            return assign_stmts

        elif isinstance(target, ast.Attribute):
            obj_stmts, obj_expr = self.visit(target.value)
            assign_stmts = [self.create_pil_assign_attribute(obj_expr,
                     target.attr,
                     source_expr)] if source_expr is not None else []
            return obj_stmts + assign_stmts

        elif isinstance(target, ast.Subscript):
            obj_stmts, obj_expr = self.visit(target.value)
            slice_stmt_list, pil_slice_list = self.visit_slice_values(target.slice)
            assign_stmts = [self.create_pil_assign_subscript(obj_expr,
                     pil_slice_list,
                     source_expr)] if source_expr is not None else []
            return obj_stmts + slice_stmt_list + assign_stmts

        elif isinstance(target, (ast.Tuple, ast.List)):
            # Step 1: allocate one temp per element, preserving starred-ness
            elt_temps_data = [(self.create_temp_identifier(), isinstance(elt, ast.Starred)) for elt in target.elts]
            if isinstance(target, ast.List):
                elt_temps = elt_temps_data
            else:
                elt_temps = tuple(elt_temps_data)

            # Step 2: emit the first unpack into per-element temporaries.
            unpack_stmts = [self.create_pil_assign_name(elt_temps,
                     self.create_pil_expr(source_expr))] if source_expr is not None else []
            # Step 3: recursively handle each element with its temp
            result_stmts = unpack_stmts
            for (temp_name, starred), elt in zip(elt_temps, target.elts):
                if isinstance(elt, ast.List):
                    result_stmts.extend(self.visit_lhs(elt, temp_name))
                elif isinstance(elt, ast.Tuple):
                    result_stmts.extend(self.visit_lhs(elt, temp_name))
                else:
                    actual_elt = elt.value if starred else elt
                    result_stmts.extend(self.visit_lhs(actual_elt, temp_name))
            return result_stmts

        raise FeError(NotImplementedError(f"LHS target type {type(target).__name__} is not supported"))

    def visit_function_def(
        self,
        name: str,
        args: ast.arguments,
        body: list[ast.stmt],
        decorator_list: list[ast.expr],
        returns: Optional[ast.expr],
        type_comment: Optional[str],
        node_attr: PILAttr = NOATTR,
        **kwargs,
    ) -> tuple[list[ast.stmt], PILExprOrNone]:
        """
        Case 1 (no decorators):
            Python:

                def name(args):
                    body
            PIL:

                def name(args):
                    body
        Case 2 (with decorators):
            Python:

                @dec_expr
                def name(args):
                    body
            PIL:
                _tmp_0 = dec_expr

                @_tmp_0
                def name(args):
                    body
        """
        decorator_stmts = []
        decorator_names = []
        for dec in decorator_list:
            dec_stmts, dec_name = self.visit(dec)
            decorator_stmts.extend(dec_stmts)
            decorator_names.append(dec_name)
        body_stmt_list, _ = self.visit_stmts(body)
        func_def = self.create_pil_function_def(name,
             args,
             body_stmt_list,
             decorator_names,
             returns,
             type_comment,
             node_attr=node_attr)
        return decorator_stmts + [func_def], None

    def visit_async_function_def(self,
         name: str,
         args: ast.arguments,
         body: list[ast.stmt],
         decorator_list: list[ast.expr],
         returns: Optional[ast.expr],
         type_comment: Optional[str],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        raise FeError(NotImplementedError("AsyncFunctionDef is not supported"))

    def visit_class_def(self,
         name: str,
         bases: list[ast.expr],
         keywords: list[ast.keyword],
         body: list[ast.stmt],
         decorator_list: list[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        raise FeError(NotImplementedError("ClassDef is not supported"))

    def visit_return(self,
         value: Optional[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Case 1:
            Python:
                return
            PIL:
                return
        Case 2:
            Pythoin:
                return id0
            PIL:
                return id0
        Case 3:
            Python:
                return expr
            PIL:
                _tmp_0 = expr
                return _tmp_0
        """
        if value is not None:
            value_stmt_list, value_name = self.visit(value)
            result_stmt_list = [*value_stmt_list, self.create_pil_return(value_name, node_attr=node_attr)]
        else:
            result_stmt_list = [self.create_pil_return(None, node_attr=node_attr)]
        return result_stmt_list, None

    def visit_delete(self,
         targets: list[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Case 1 (delete name):
            Python:
                del name
            PIL:
                del name
        Case 2 (delete attribute):
            Python:
                del obj.attr
            PIL:
                _tmp_0 = obj
                del _tmp_0.attr
        Case 3 (delete subscript):
            Python:
                del obj[key]
            PIL:
                _tmp_0 = obj
                _tmp_1 = key
                del _tmp_0[_tmp_1]
        Case 4 (delete tuple/list - recursive):
            Python:
                del (a, obj.attr, obj[key])
            PIL:
                del a
                _tmp_0 = obj
                del _tmp_0.attr
                _tmp_1 = obj
                _tmp_2 = key
                del _tmp_1[_tmp_2]
        Case 5 (multiple targets):
            Python:
                del a, obj.attr, obj[key]
            PIL:
                del a
                _tmp_0 = obj
                del _tmp_0.attr
                _tmp_1 = obj
                _tmp_2 = key
                del _tmp_1[_tmp_2]
        """
        result_stmts = []
        for target in targets:
            if isinstance(target, ast.Name):
                result_stmts.append(self.create_pil_delete_identifier(target.id, node_attr=node_attr))
            elif isinstance(target, ast.Attribute):
                obj_stmts, obj_expr = self.visit(target.value)
                result_stmts.extend(obj_stmts)
                result_stmts.append(self.create_pil_delete_attribute(obj_expr, target.attr, node_attr=node_attr))
            elif isinstance(target, ast.Subscript):
                obj_stmts, obj_expr = self.visit(target.value)
                result_stmts.extend(obj_stmts)
                slice_stmts, slice_expr = self.visit(target.slice)
                result_stmts.extend(slice_stmts)
                result_stmts.append(self.create_pil_delete_subscript(obj_expr, slice_expr, node_attr=node_attr))
            elif isinstance(target, (ast.Tuple, ast.List)):
                nested_stmts, _ = self.visit_delete(target.elts, node_attr=node_attr)
                result_stmts.extend(nested_stmts)
            else:
                raise FeError(NotImplementedError(f"Delete target type {type(target).__name__} is not supported"))
        return result_stmts, None

    def visit_assign(self,
         targets: list[ast.expr],
         value: ast.expr,
         type_comment: Optional[str],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Case 1:
            Python:
                a = b = id0
            PIL:
                a = id0
                b = id0
        Case 2:
            Python:
                a = b = expr
            PIL:
                _tmp_0 = expr
                a = _tmp_0
                b = _tmp_0
        """
        value_stmt_list, value_name = self.visit(value)
        result_stmt_list = value_stmt_list
        for target in targets:
            target_stmt_list = self.visit_lhs(target, value_name)
            result_stmt_list.extend(target_stmt_list)
        return result_stmt_list, None

    def visit_aug_assign(self,
         target: ast.expr,
         op: ast.operator,
         value: ast.expr,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Case 1 (name target):
            Python:
                name op= expr
            PIL:
                _tmp_0 = expr
                _tmp_1 = name op _tmp_0
                name = _tmp_1
        Case 2 (attribute target):
            Python:
                obj.attr op= expr
            PIL:
                _tmp_0 = obj_expr       # evaluate obj
                _tmp_1 = _tmp_0.attr    # load
                _tmp_2 = expr           # evaluate rhs
                _tmp_3 = _tmp_1 op _tmp_2
                _tmp_0.attr = _tmp_3    # store
        Case 3 (subscript target):
            Python:
                obj[slice] op= expr
            PIL:
                _tmp_0 = obj_expr       # evaluate obj
                _tmp_1 = slice_expr     # evaluate slice
                _tmp_2 = _tmp_0[_tmp_1] # load
                _tmp_3 = expr           # evaluate rhs
                _tmp_4 = _tmp_2 op _tmp_3
                _tmp_0[_tmp_1] = _tmp_4 # store
        """
        if isinstance(target, ast.Name):
            # target is a bare name - no side effects, visit value first is fine
            value_stmt_list, value_name = self.visit(value)
            temp_name = self.create_temp_identifier()
            binop_stmt = self.create_pil_assign_name(temp_name, self.create_pil_bin_op(target.id, op, value_name))
            store_stmt = self.create_pil_assign_identifier(target.id, temp_name)
            return value_stmt_list + [binop_stmt, store_stmt], None

        elif isinstance(target, ast.Attribute):
            # Python evaluates: obj first, then load, then rhs value, then store
            target_stmt_list, target_expr = self.visit(target.value)
            load_temp = self.create_temp_identifier()
            load_stmt = self.create_pil_assign_name(load_temp, self.create_pil_attribute(target_expr, target.attr))
            value_stmt_list, value_name = self.visit(value)
            temp_name = self.create_temp_identifier()
            binop_stmt = self.create_pil_assign_name(temp_name, self.create_pil_bin_op(load_temp, op, value_name))
            store_stmt = self.create_pil_assign_attribute(target_expr, target.attr, temp_name)
            return target_stmt_list + [load_stmt] + value_stmt_list + [binop_stmt, store_stmt], None

        elif isinstance(target, ast.Subscript):
            # Python evaluates: obj first, then slice, then load, then rhs value, then store
            target_stmt_list, target_expr = self.visit(target.value)
            # normalize slice into list of (lower, upper, step) tuples
            slice_stmt_list, pil_slice_list = self.visit_slice_values(target.slice)
            load_temp = self.create_temp_identifier()
            load_stmt = self.create_pil_assign_name(load_temp, self.create_pil_subscript(target_expr, pil_slice_list))
            value_stmt_list, value_name = self.visit(value)
            temp_name = self.create_temp_identifier()
            binop_stmt = self.create_pil_assign_name(temp_name, self.create_pil_bin_op(load_temp, op, value_name))
            store_stmt = self.create_pil_assign_subscript(target_expr, pil_slice_list, temp_name)
            return target_stmt_list + slice_stmt_list + [load_stmt] + value_stmt_list + [binop_stmt, store_stmt], None

        raise FeError(NotImplementedError(f"AugAssign target type {type(target).__name__} is not supported"))

    def visit_ann_assign(self,
         target: ast.expr,
         annotation: ast.expr,
         value: Optional[ast.expr],
         simple: int,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Case 1 (annotation only):
            Python:
                target: annotation
            PIL:
                _tmp_0 = annotation     # evaluate annotation for side effects
        Case 2 (annotation with value):
            Python:
                target: annotation = expr
            PIL:
                _tmp_0 = annotation     # evaluate annotation for side effects
                _tmp_1 = expr
                target = _tmp_1
        """
        result_stmts = []
        if value is not None:
            value_stmts, value_expr = self.visit(value)
        else:
            value_stmts, value_expr = [], None
        ann_stmts, _ = self.visit(annotation)
        return value_stmts + self.visit_lhs(target, value_expr) + ann_stmts, None

    def visit_for(self,
         target: ast.expr,
         iter_expr: ast.expr,
         body: list[ast.stmt],
         orelse: list[ast.stmt],
         type_comment: Optional[str],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            for target in iter_expr:
                body
            else:
                orelse
        PIL:
            _tmp_0 = iter_expr
            for _tmp_1 in _tmp_0:
                target = _tmp_1         # visit_lhs unpacking
                body
            else:
                orelse
        """
        self.continue_stack.append(None)

        iter_stmt_list, iter_name = self.visit(iter_expr)
        target_name = self.create_temp_identifier()
        target_stmt_list = self.visit_lhs(target, target_name)

        body_stmt_list, _ = self.visit_stmts(body)
        result_body_stmt_list = target_stmt_list + body_stmt_list
        orelse_stmt_list, _ = self.visit_stmts(orelse)
        result_stmt_list = iter_stmt_list + [self.create_pil_for(target_name,
                 iter_name,
                 result_body_stmt_list,
                 orelse_stmt_list,
                 type_comment)]
        self.continue_stack.pop()
        return result_stmt_list, None

    def visit_async_for(self,
         target: ast.expr,
         iter_expr: ast.expr,
         body: list[ast.stmt],
         orelse: list[ast.stmt],
         type_comment: Optional[str],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        raise FeError(NotImplementedError("AsyncFor is not supported"))

    def visit_while(self,
         test: ast.expr,
         body: list[ast.stmt],
         orelse: list[ast.stmt],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            while test_expr:
                body
            else:
                orelse
        PIL:
            _tmp_0 = test_expr
            while _tmp_0:
                body
                _tmp_1 = test_expr      # re-evaluate for next iteration
                _tmp_0 = _tmp_1
            else:
                orelse
        """
        test_stmt_list, test_expr = self.visit(test)
        self.continue_stack.append((test, test_expr))
        body_stmt_list, _ = self.visit_stmts(body)
        orelse_stmt_list, _ = self.visit_stmts(orelse)

        if isinstance(test_expr, str):
            # re-evaluate test at the end of each iteration to update test_name for the next check
            reeval_stmt_list, reeval_expr = self.visit(test)
            reeval_stmt_list = reeval_stmt_list + [self.create_pil_assign_identifier(test_expr, reeval_expr)]
            result_body_stmt_list = body_stmt_list + reeval_stmt_list
        else:
            result_body_stmt_list = body_stmt_list
        result_stmt_list = test_stmt_list + [self.create_pil_while(test_expr, result_body_stmt_list, orelse_stmt_list)]
        self.continue_stack.pop()
        return result_stmt_list, None

    def visit_if(self,
         test: ast.expr,
         body: list[ast.stmt],
         orelse: list[ast.stmt],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            if test_expr:
                body
            else:
                orelse
        PIL:
            _tmp_0 = test_expr
            if _tmp_0:
                body
            else:
                orelse
        """
        test_stmt_list, test_name = self.visit(test)
        body_stmt_list, _ = self.visit_stmts(body)
        orelse_stmt_list, _ = self.visit_stmts(orelse)
        result_stmt_list = test_stmt_list + [self.create_pil_if(test_name, body_stmt_list, orelse_stmt_list)]
        return result_stmt_list, None

    def visit_with(self,
         items: list[ast.withitem],
         body: list[ast.stmt],
         type_comment: Optional[str],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            with ctx_expr0 as var0, ctx_expr1 as var1:
                body
        PIL:
            _tmp_0 = ctx_expr0
            _tmp_1 = ctx_expr1
            with _tmp_0, _tmp_1:
                var0 = _tmp_0           # optional_vars assigned inside body
                var1 = _tmp_1
                body
        """
        result_stmts = []
        ctx_names = []
        for item in items:
            ctx_stmts, ctx_name = self.visit(item.context_expr)
            result_stmts.extend(ctx_stmts)
            ctx_names.append(ctx_name)
        # optional_vars assignment happens after __enter__, so goes at the start of body
        body_preamble = []
        for item, ctx_name in zip(items, ctx_names):
            if item.optional_vars is not None:
                body_preamble.extend(self.visit_lhs(item.optional_vars, ctx_name))
        body_stmts, _ = self.visit_stmts(body)
        result_stmts.append(self.create_pil_with(ctx_names,
                 body_preamble + body_stmts,
                 type_comment,
                 node_attr=node_attr))
        return result_stmts, None

    def visit_async_with(self,
         items: list[ast.withitem],
         body: list[ast.stmt],
         type_comment: Optional[str],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        raise FeError(NotImplementedError("AsyncWith is not supported"))

    def visit_match(self,
         subject: ast.expr,
         cases: list,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        raise FeError(NotImplementedError("Match is not supported"))

    def visit_raise(self,
         exc: Optional[ast.expr],
         cause: Optional[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Case 1 (bare re-raise):
            Python:
                raise
            PIL:
                raise
        Case 2 (raise exception):
            Python:
                raise exc_expr
            PIL:
                _tmp_0 = exc_expr
                raise _tmp_0
        Case 3 (raise with cause):
            Python:
                raise exc_expr from cause_expr
            PIL:
                _tmp_0 = exc_expr
                _tmp_1 = cause_expr
                raise _tmp_0 from _tmp_1
        """
        result_stmts = []
        exc_name = None
        if exc is not None:
            exc_stmts, exc_name = self.visit(exc)
            result_stmts.extend(exc_stmts)
        cause_name = None
        if cause is not None:
            cause_stmts, cause_name = self.visit(cause)
            result_stmts.extend(cause_stmts)
        result_stmts.append(self.create_pil_raise(exc_name, cause_name, node_attr=node_attr))
        return result_stmts, None

    def visit_try(self,
         body: list[ast.stmt],
         handlers: list[ast.excepthandler],
         orelse: list[ast.stmt],
         finalbody: list[ast.stmt],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            try:
                body
            except TypeA as e:
                handler_a_body
            except TypeB:
                handler_b_body
            except:
                bare_body
            else:
                orelse
            finally:
                finalbody
        PIL:
            try:
                body
            except Exception as _tmp_0:
                _tmp_1 = TypeA
                _tmp_2 = isinstance(_tmp_0, _tmp_1)
                if _tmp_2:
                    e = _tmp_0
                    handler_a_body
                    del e
                else:
                    _tmp_3 = TypeB
                    _tmp_4 = isinstance(_tmp_0, _tmp_3)
                    if _tmp_4:
                        handler_b_body
                    else:
                        bare_body
            else:
                orelse
            finally:
                finalbody
        """
        body_stmts, _ = self.visit_stmts(body)
        exc_var = self.create_temp_identifier()
        # Build if-elif chain from innermost outward; default else: bare raise
        dispatch_body: list[ast.stmt] = [self.create_pil_raise(None, None)]

        for handler in reversed(handlers):

            handler_body_stmts, _ = self.visit_stmts(handler.body)

            if handler.type is not None:
                handler_type_stmts, handler_type_name = self.visit(handler.type)
                isinstance_temp = self.create_temp_identifier()
                isinstance_stmts = [self.create_pil_assign_name(isinstance_temp,
                         self.create_pil_call('isinstance',
                             [exc_var,
                                 handler_type_name],
                             []))]

                if handler.name is not None:
                    pre_handler_set = [self.create_pil_assign_identifier(handler.name, exc_var)]
                    post_handler_del = [self.create_pil_delete_identifier(handler.name)]
                    result_then_stmts = pre_handler_set + handler_body_stmts + post_handler_del
                else:
                    result_then_stmts = handler_body_stmts
            else:
                handler_type_stmts = []
                isinstance_stmts = []
                isinstance_temp = self.create_pil_constant(True, None)
                result_then_stmts = handler_body_stmts

            dispatch_body = handler_type_stmts + isinstance_stmts + [self.create_pil_if(isinstance_temp,
                     result_then_stmts,
                     dispatch_body)]
        orelse_stmts, _ = self.visit_stmts(orelse)
        finalbody_stmts, _ = self.visit_stmts(finalbody)
        return [self.create_pil_try(body_stmts,
                 (exc_var,
                     dispatch_body),
                 orelse_stmts,
                 finalbody_stmts,
                 node_attr=node_attr)], None

    def visit_try_star(self,
         body: list[ast.stmt],
         handlers: list[ast.excepthandler],
         orelse: list[ast.stmt],
         finalbody: list[ast.stmt],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        raise FeError(NotImplementedError("TryStar is not supported"))

    def visit_assert(self,
         test: ast.expr,
         msg: Optional[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            assert test_expr, msg_expr
        PIL:
            if __debug__:
                _tmp_0 = test_expr
                _tmp_1 = not _tmp_0
                if _tmp_1:
                    _tmp_2 = msg_expr
                    assert _tmp_0, _tmp_2
        """
        test_stmt_list, test_name = self.visit(test)
        # use prefix "not_test" to keep "not test_name"
        not_test_name = self.create_temp_identifier()
        not_test_stmt = self.create_pil_assign_name(not_test_name, self.create_pil_unary_op(ast.Not(), test_name))
        # msg is only evaluated when the assertion fails
        if msg is not None:
            msg_stmt_list, msg_name = self.visit(msg)
        else:
            msg_stmt_list, msg_name = [], None
        fail_body = msg_stmt_list + [self.create_pil_assert(test_name, msg_name)]
        debug_body = test_stmt_list + [not_test_stmt, self.create_pil_if(not_test_name, fail_body, [])]
        return [self.create_pil_if("__debug__", debug_body, [])], None

    def visit_import(self,
         names: list[ast.alias],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        return [self.create_pil_import(names)], None

    def visit_import_from(self,
         module: Optional[str],
         names: list[ast.alias],
         level: Optional[int],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        return [self.create_pil_import_from(module, names, level)], None

    def visit_global(self, names: list[str], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExprOrNone]:
        return [self.create_pil_global(names)], None

    def visit_nonlocal(self, names: list[str], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExprOrNone]:
        return [self.create_pil_nonlocal(names)], None

    def visit_expr(self, value: ast.expr, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExprOrNone]:
        value_stmt_list, value_name = self.visit(value)
        return value_stmt_list, None

    def visit_pass(self, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExprOrNone]:
        return [self.create_pil_pass()], None

    def visit_break(self, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExprOrNone]:
        return [self.create_pil_break()], None

    def visit_continue(self, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExprOrNone]:
        if self.continue_stack[-1] is None:
            # continue in for-loop
            result_stmt_list = [self.create_pil_continue()]
        else:
            # continue in while-loop
            test, test_expr = self.continue_stack[-1]
            if isinstance(test_expr, str):
                reeval_stmt_list, reeval_expr = self.visit(test)
                result_stmt_list = reeval_stmt_list + [self.create_pil_assign_identifier(test_expr,
                         reeval_expr),
                     self.create_pil_continue()]
            else:
                result_stmt_list = [self.create_pil_continue()]
        return result_stmt_list, None

    # expr nodes

    def visit_bool_op(self,
         op: ast.boolop,
         values: list[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Case 1 (and):
            Python:
                a and b and c
            PIL:
                _tmp_0 = a
                if _tmp_0:
                    _tmp_1 = b and c    # recursive
                    _tmp_0 = _tmp_1
                else:
                    _tmp_0 = a
        Case 2 (or):
            Python:
                a or b or c
            PIL:
                _tmp_0 = a
                if _tmp_0:
                    _tmp_0 = a
                else:
                    _tmp_1 = b or c    # recursive
                    _tmp_0 = _tmp_1
        """
        # Base case: single value, just visit it directly
        if len(values) == 1:
            return self.visit(values[0])

        temp_name = self.create_temp_identifier()
        first_stmt_list, first_name = self.visit(values[0])
        rest_stmt_list, rest_name = self.visit_bool_op(op=op, values=values[1:])

        if isinstance(op, ast.And):
            # if temp is truthy, evaluate the rest and update temp
            rest_stmt = self.create_pil_if(
                first_name,
                rest_stmt_list + [self.create_pil_assign_identifier(temp_name, rest_name)],
                [self.create_pil_assign_identifier(temp_name, first_name)])

            result_stmt_list = first_stmt_list + [rest_stmt]
            return result_stmt_list, temp_name

        elif isinstance(op, ast.Or):
            # if temp is falsy, evaluate the rest and update temp
            rest_stmt = self.create_pil_if(
                first_name,
                [self.create_pil_assign_identifier(temp_name, first_name)],
                rest_stmt_list + [self.create_pil_assign_identifier(temp_name, rest_name)])

            result_stmt_list = first_stmt_list + [rest_stmt]
            return result_stmt_list, temp_name

        raise FeError(NotImplementedError(f"BoolOp {type(op).__name__} is not supported"))

    def visit_named_expr(self,
         target: ast.expr,
         value: ast.expr,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            target := expr
        PIL:
            _tmp_0 = expr
            target = _tmp_0
        """
        if not isinstance(target, ast.Name):
            raise FeError(
                TypeError('Python native ast parser should guarantee that the target of NamedExpr is always ast.Name'))
        value_stmt_list, value_name = self.visit(value)
        assign_stmt = self.create_pil_assign_identifier(target.id, value_name)
        result_stmt_list = value_stmt_list + [assign_stmt]
        return result_stmt_list, target.id

    def visit_bin_op(self,
         left: ast.expr,
         op: ast.operator,
         right: ast.expr,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            left_expr op right_expr
        PIL:
            _tmp_0 = left_expr
            _tmp_1 = right_expr
            _tmp_2 = _tmp_0 op _tmp_1
        """
        left_stmt_list, left_name = self.visit(left)
        right_stmt_list, right_name = self.visit(right)

        temp_name = self.create_temp_identifier()
        result_expr = self.create_pil_bin_op(left_name, op, right_name)
        binop_stmt = self.create_pil_assign_name(temp_name, result_expr)
        result_stmt_list = left_stmt_list + right_stmt_list + [binop_stmt]
        return result_stmt_list, temp_name

    def visit_unary_op(self,
         op: ast.unaryop,
         operand: ast.expr,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            op operand_expr
        PIL:
            _tmp_0 = operand_expr
            _tmp_1 = op _tmp_0
        """
        operand_stmt_list, operand_name = self.visit(operand)

        temp_name = self.create_temp_identifier()
        result_expr = self.create_pil_unary_op(op, operand_name)
        unaryop_stmt = self.create_pil_assign_name(temp_name, result_expr)
        result_stmt_list = operand_stmt_list + [unaryop_stmt]
        return result_stmt_list, temp_name

    def visit_lambda(self,
         args: ast.arguments,
         body: ast.expr,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            lambda args: body_expr
        PIL:

            def _tmp_0(args):
                _tmp_1 = body_expr
                return _tmp_1
        """
        body_stmt_list, body_name = self.visit(body)
        return_stmt = self.create_pil_return(body_name)
        func_name = self.create_temp_identifier()
        func_def = self.create_pil_function_def(
            name=func_name,
            args=args,
            body=body_stmt_list + [return_stmt],
            decorator_list=[],
            returns=None,
            type_comment=None,
            node_attr=node_attr)
        return [func_def], func_name

    def visit_if_exp(self,
         test: ast.expr,
         body: ast.expr,
         orelse: ast.expr,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            body_expr if test_expr else orelse_expr
        PIL:
            _tmp_0 = test_expr
            if _tmp_0:
                _tmp_1 = body_expr
                _tmp_2 = _tmp_1
            else:
                _tmp_3 = orelse_expr
                _tmp_2 = _tmp_3
        """
        test_stmt_list, test_name = self.visit(test)
        body_stmt_list, body_name = self.visit(body)
        orelse_stmt_list, orelse_name = self.visit(orelse)

        temp_name = self.create_temp_identifier()
        result_body_stmt_list = body_stmt_list + [self.create_pil_assign_identifier(temp_name, body_name)]
        result_orelse_stmt_list = orelse_stmt_list + [self.create_pil_assign_identifier(temp_name, orelse_name)]
        result_if_stmt = self.create_pil_if(test_name, result_body_stmt_list, result_orelse_stmt_list)
        result_stmt_list = test_stmt_list + [result_if_stmt]
        return result_stmt_list, temp_name

    def visit_dict(self,
         keys: list[Optional[ast.expr]],
         values: list[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            {key0: val0, **val1, key2: val2}
        PIL:
            _tmp_0 = key0
            _tmp_1 = val0
            _tmp_2 = val1
            _tmp_3 = key2
            _tmp_4 = val2
            _tmp_5 = {_tmp_0: _tmp_1, **_tmp_2, _tmp_3: _tmp_4}
        """
        result_stmt_list = []
        key_name_list = []
        value_name_list = []
        for key, value in zip(keys, values):
            if key is not None:
                key_stmt_list, key_name = self.visit(key)
            else:
                key_stmt_list, key_name = [], None
            value_stmt_list, value_name = self.visit(value)

            result_stmt_list.extend(key_stmt_list)
            result_stmt_list.extend(value_stmt_list)
            key_name_list.append(key_name)
            value_name_list.append(value_name)

        temp_name = self.create_temp_identifier()
        result_expr = self.create_pil_dict(key_name_list, value_name_list)
        result_stmt = self.create_pil_assign_name(temp_name, result_expr)
        result_stmt_list.append(result_stmt)
        return result_stmt_list, temp_name

    def visit_set(self, elts: list[ast.expr], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExprOrNone]:
        """
        Python:
            {elt0, *elt1, elt2}
        PIL:
            _tmp_0 = elt0
            _tmp_1 = elt1
            _tmp_2 = elt2
            _tmp_3 = {_tmp_0, *_tmp_1, _tmp_2}
        """
        temp_name = self.create_temp_identifier()
        result_stmt_list, elt_list = self._collect_sequence_elements(elts)
        result_expr = self.create_pil_set(elt_list)
        result_stmt = self.create_pil_assign_name(temp_name, result_expr)
        result_stmt_list.append(result_stmt)
        return result_stmt_list, temp_name

    def visit_comp_generator(self, elt: ast.expr, generators: list[ast.comprehension]) -> list[ast.stmt]:
        gen = generators[0]
        iter_stmts, iter_name = self.visit(gen.iter)

        target_var = self.create_temp_identifier()
        target_stmts = self.visit_lhs(gen.target, target_var)

        if len(generators) == 1:
            elt_stmts, elt_name = self.visit(elt)
            yield_temp = self.create_temp_identifier()
            yield_stmt = self.create_pil_assign_name(yield_temp, self.create_pil_yield(elt_name))
            inner_body = elt_stmts + [yield_stmt]
        else:
            inner_body = self.visit_comp_generator(elt, generators[1:])

        # Apply if guards: wrap inner_body from innermost outward
        for cond in reversed(gen.ifs):
            cond_stmts, cond_name = self.visit(cond)
            inner_body = cond_stmts + [self.create_pil_if(cond_name, inner_body, [])]

        for_stmt = self.create_pil_for(target_var, iter_name, target_stmts + inner_body, [], None)
        return iter_stmts + [for_stmt]

    def visit_comp(self, elt: ast.expr, generators: list[ast.comprehension]) -> tuple[ast.FunctionDef, str]:
        comp_body = self.visit_comp_generator(elt, generators)
        func_name = self.create_temp_identifier()
        func_args = ast.arguments(
            posonlyargs=[], args=[],
            vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
        return self.create_pil_function_def(func_name, func_args, comp_body, [], None, None), func_name

    def visit_list_comp(self,
         elt: ast.expr,
         generators: list[ast.comprehension],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            [elt for x in iter if cond]
        PIL:

            def _tmp_0():
                for _tmp_1 in iter:
                    x = _tmp_1
                    _tmp_2 = cond
                    if _tmp_2:
                        _tmp_3 = elt
                        _tmp_4 = yield _tmp_3
            _tmp_5 = _tmp_0()
            _tmp_6 = [*_tmp_5]
        """
        func_def, func_name = self.visit_comp(elt, generators)
        gen_name = self.create_temp_identifier()
        gen_stmt = self.create_pil_assign_name(gen_name, self.create_pil_call(func_name, [], []))
        result_name = self.create_temp_identifier()
        result_stmt = self.create_pil_assign_name(result_name, self.create_pil_list([(gen_name, True)]))
        return [func_def, gen_stmt, result_stmt], result_name

    def visit_set_comp(self,
         elt: ast.expr,
         generators: list[ast.comprehension],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            {elt for x in iter if cond}
        PIL:

            def _tmp_0():
                for _tmp_1 in iter:
                    x = _tmp_1
                    _tmp_2 = cond
                    if _tmp_2:
                        _tmp_3 = elt
                        _tmp_4 = yield _tmp_3
            _tmp_5 = _tmp_0()
            _tmp_6 = {*_tmp_5}
        """
        func_def, func_name = self.visit_comp(elt, generators)
        gen_name = self.create_temp_identifier()
        gen_stmt = self.create_pil_assign_name(gen_name, self.create_pil_call(func_name, [], []))
        result_name = self.create_temp_identifier()
        result_stmt = self.create_pil_assign_name(result_name, self.create_pil_set([(gen_name, True)]))
        return [func_def, gen_stmt, result_stmt], result_name

    def visit_dict_comp(self,
         key: ast.expr,
         value: ast.expr,
         generators: list[ast.comprehension],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            {key: val for x in iter if cond}
        PIL:

            def _tmp_0():
                for _tmp_1 in iter:
                    x = _tmp_1
                    _tmp_2 = cond
                    if _tmp_2:
                        _tmp_3 = (key, val)
                        _tmp_4 = yield _tmp_3
            _tmp_5 = _tmp_0()
            _tmp_6 = dict(_tmp_5)
        """
        kv_tuple = ast.Tuple(elts=[key, value], ctx=ast.Load())
        func_def, func_name = self.visit_comp(kv_tuple, generators)
        gen_name = self.create_temp_identifier()
        gen_stmt = self.create_pil_assign_name(gen_name, self.create_pil_call(func_name, [], []))
        result_name = self.create_temp_identifier()
        result_stmt = self.create_pil_assign_name(result_name, self.create_pil_call('dict', [gen_name], []))
        return [func_def, gen_stmt, result_stmt], result_name

    def visit_generator_exp(self,
         elt: ast.expr,
         generators: list[ast.comprehension],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            (elt for x in iter if cond)
        PIL:

            def _tmp_0():
                for _tmp_1 in iter:
                    x = _tmp_1
                    _tmp_2 = cond
                    if _tmp_2:
                        _tmp_3 = elt
                        _tmp_4 = yield _tmp_3
            _tmp_5 = _tmp_0()
        """
        func_def, func_name = self.visit_comp(elt, generators)
        result_name = self.create_temp_identifier()
        result_stmt = self.create_pil_assign_name(result_name, self.create_pil_call(func_name, [], []))
        return [func_def, result_stmt], result_name

    def visit_await(
        self,
        value: ast.expr,
        node_attr: PILAttr = NOATTR,
    ) -> tuple[list[ast.stmt], PILExprOrNone]:
        raise FeError(NotImplementedError("Await is not supported"))

    def visit_yield(
        self,
        value: Optional[ast.expr],
        node_attr: PILAttr = NOATTR,
    ) -> tuple[list[ast.stmt], PILExprOrNone]:
        """
        Case 1:
            Python:
                yield expr
            PIL:
                _tmp_0 = expr
                _tmp_1 = yield _tmp_0
        Case 2:
            Python:
                yield
            PIL:
                _tmp_0 = yield
        """
        temp_name = self.create_temp_identifier()
        if value is not None:
            value_stmt_list, value_expr = self.visit(value)
            yield_stmt = self.create_pil_assign_name(temp_name, self.create_pil_yield(value_expr))
            result_stmt_list = value_stmt_list + [yield_stmt]
            return result_stmt_list, temp_name
        else:
            yield_stmt = self.create_pil_assign_name(temp_name, self.create_pil_yield(None))
            return [yield_stmt], temp_name

    def visit_yield_from(self, value: ast.expr, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExprOrNone]:
        """
        Python:
            yield from expr
        PIL:
            _tmp_0 = expr
            _tmp_1 = yield from _tmp_0
        """
        value_stmts, value_name = self.visit(value)
        temp_name = self.create_temp_identifier()
        yield_from_stmt = self.create_pil_assign_name(temp_name, self.create_pil_yield_from(value_name))
        return value_stmts + [yield_from_stmt], temp_name

    def visit_compare(self,
         left: ast.expr,
         ops: list[ast.cmpop],
         comparators: list[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Case 1 (single comparison):
            Python:
                a < b
            PIL:
                _tmp_0 = a
                _tmp_1 = b
                _tmp_2 = _tmp_0 < _tmp_1
        Case 2 (chained comparison):
            Python:
                a < b < c
            PIL:
                _tmp_0 = a
                _tmp_1 = b
                _tmp_2 = _tmp_0 < _tmp_1
                if _tmp_2:
                    _tmp_3 = c
                    _tmp_4 = _tmp_1 < _tmp_3    # b evaluated only once
                    _tmp_2 = _tmp_4
        """
        # Base case: single comparison - visit left and comparator, build compare expr
        left_stmts, left_name = self.visit(left)
        comp_stmts, comp_name = self.visit(comparators[0])
        temp_name = self.create_temp_identifier()
        first_stmt = self.create_pil_assign_name(temp_name, self.create_pil_compare(left_name, ops[0], comp_name))

        if len(ops) == 1:
            return left_stmts + comp_stmts + [first_stmt], temp_name

        # Recursive case: a op0 b op1 c ... => (a op0 b) and (b op1 c ...)
        # comp_name is reused as left of the next comparison (evaluated only once)
        rest_stmt_list, rest_name = self.visit_compare(
            left=ast.Name(id=comp_name, ctx=ast.Load()),
            ops=ops[1:],
            comparators=comparators[1:])
        rest_stmt = self.create_pil_if(
            temp_name,
            rest_stmt_list + [self.create_pil_assign_identifier(temp_name, rest_name)],
            [])
        return left_stmts + comp_stmts + [first_stmt, rest_stmt], temp_name

    def visit_call(self,
         func: ast.expr,
         args: list[ast.expr],
         keywords: list[ast.keyword],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            func_expr(arg0, arg1, key=kw_expr)
        PIL:
            _tmp_0 = func_expr
            _tmp_1 = arg0
            _tmp_2 = arg1
            _tmp_3 = kw_expr
            _tmp_4 = _tmp_0(_tmp_1, _tmp_2, key=_tmp_3)
        """
        result_stmt_list = []

        # func must be an identifier per PIL spec
        func_stmts, func_name = self.visit(func)
        result_stmt_list.extend(func_stmts)

        # visit positional args
        arg_names = []
        for arg in args:
            arg_stmts, arg_name = self.visit(arg)
            result_stmt_list.extend(arg_stmts)
            arg_names.append(arg_name)

        # visit keyword values, rewrite keyword nodes with resolved names
        pil_keywords = []
        for kw in keywords:
            kw_stmts, kw_name = self.visit(kw.value)
            result_stmt_list.extend(kw_stmts)
            pil_keywords.append((kw.arg, kw_name))

        temp_name = self.create_temp_identifier()
        result_expr = self.create_pil_call(func_name, arg_names, pil_keywords)
        result_stmt_list.append(self.create_pil_assign_name(temp_name, result_expr))
        return result_stmt_list, temp_name

    def visit_formatted_value(self,
         value: ast.expr,
         conversion: int,
         format_spec: Optional[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Case 1 (no conversion, no format_spec):
            Python:
                {expr}
            PIL:
                _tmp_0 = expr
        Case 2 (with conversion):
            Python:
                {expr!r}
            PIL:
                _tmp_0 = expr
                _tmp_1 = repr(_tmp_0)
        Case 3 (with format_spec):
            Python:
                {expr:fmt}
            PIL:
                _tmp_0 = expr
                _tmp_1 = fmt_expr
                _tmp_2 = format(_tmp_0, _tmp_1)
        Case 4 (with conversion and format_spec):
            Python:
                {expr!r:fmt}
            PIL:
                _tmp_0 = expr
                _tmp_1 = repr(_tmp_0)
                _tmp_2 = fmt_expr
                _tmp_3 = format(_tmp_1, _tmp_2)
        """
        result_stmts, value_name = self.visit(value)

        # Apply the formatted-value conversion code when one is present.
        if conversion == ord('s'):
            conv_name = self.create_temp_identifier()
            result_stmts += [self.create_pil_assign_name(conv_name, self.create_pil_call('str', [value_name], []))]
            value_name = conv_name
        elif conversion == ord('r'):
            conv_name = self.create_temp_identifier()
            result_stmts += [self.create_pil_assign_name(conv_name, self.create_pil_call('repr', [value_name], []))]
            value_name = conv_name
        elif conversion == ord('a'):
            conv_name = self.create_temp_identifier()
            result_stmts += [self.create_pil_assign_name(conv_name, self.create_pil_call('ascii', [value_name], []))]
            value_name = conv_name

        if format_spec is not None:
            spec_stmts, spec_name = self.visit(format_spec)
            result_stmts += spec_stmts
            temp_name = self.create_temp_identifier()
            result_stmts += [self.create_pil_assign_name(temp_name,
                     self.create_pil_call('format',
                         [value_name,
                             spec_name],
                         []))]
            return result_stmts, temp_name

        return result_stmts, value_name

    def visit_interpolation(self,
         value: ast.expr,
         literal_text: str,
         conversion: int,
         format_spec: Optional[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        raise FeError(NotImplementedError("Interpolation is not supported"))

    def visit_joined_str(self,
         values: list[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Case 1 (single part):
            Python:
                f"{expr}"
            PIL:
                _tmp_0 = expr           # via visit_FormattedValue
        Case 2 (multiple parts):
            Python:
                f"prefix{expr}suffix"
            PIL:
                _tmp_0 = expr           # via visit_FormattedValue
                _tmp_1 = ['prefix', _tmp_0, 'suffix']
                _tmp_2 = ''.join
                _tmp_3 = _tmp_2(_tmp_1)
        """
        result_stmts = []
        part_names = []
        for part in values:
            if isinstance(part, ast.Constant):
                part_names.append(ast.Constant(value=str(part.value)))
            else:
                part_stmts, part_name = self.visit(part)
                result_stmts.extend(part_stmts)
                part_names.append(part_name)
        if len(part_names) == 1:
            return result_stmts, part_names[0]
        parts_list_name = self.create_temp_identifier()
        result_stmts.append(self.create_pil_assign_name(parts_list_name,
                 self.create_pil_list([(n,
                             False) for n in part_names])))
        join_func = self.create_temp_identifier()
        result_stmts.append(self.create_pil_assign_name(join_func,
                 self.create_pil_attribute(ast.Constant(value=''),
                     'join')))
        temp_name = self.create_temp_identifier()
        result_stmts.append(self.create_pil_assign_name(temp_name,
                 self.create_pil_call(join_func,
                     [parts_list_name],
                     [])))
        return result_stmts, temp_name

    def visit_template_str(self,
         values: list[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        raise FeError(NotImplementedError("TemplateStr is not supported"))

    def visit_constant(self,
         value: object,
         kind: Optional[str],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        return [], self.create_pil_constant(value, kind)

    def visit_attribute(self,
         value: ast.expr,
         attr: str,
         ctx: ast.expr_context,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            value_expr.attr
        PIL:
            _tmp_0 = value_expr
            _tmp_1 = _tmp_0.attr
        """
        if not isinstance(ctx, ast.Load):
            raise FeError(TypeError(f"Expected ast.Load for ctx, but got {type(ctx).__name__}"))
        value_stmts, value_name = self.visit(value)
        temp_name = self.create_temp_identifier()
        result_expr = self.create_pil_attribute(value_name, attr)
        result_stmt_list = value_stmts + [self.create_pil_assign_name(temp_name, result_expr)]
        return result_stmt_list, temp_name

    def visit_subscript(self,
         value: ast.expr,
         slice_expr: ast.expr,
         ctx: ast.expr_context,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            value_expr[slice]
        PIL:
            _tmp_0 = value_expr
            _tmp_1 = slice_expr         # lower/upper/step each resolved
            _tmp_2 = _tmp_0[_tmp_1]
        """
        if not isinstance(ctx, ast.Load):
            raise FeError(TypeError(f"Expected ast.Load for ctx, but got {type(ctx).__name__}"))
        value_stmts, value_name = self.visit(value)

        # Normalize slice into a list of ast.Slice nodes
        slice_stmt_list, pil_slice_list = self.visit_slice_values(slice_expr)

        temp_name = self.create_temp_identifier()
        result_expr = self.create_pil_subscript(value_name, pil_slice_list)
        result_stmt_list = value_stmts + slice_stmt_list + [self.create_pil_assign_name(temp_name, result_expr)]
        return result_stmt_list, temp_name

    def visit_starred(self,
         value: ast.expr,
         ctx: ast.expr_context,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        raise FeError(RuntimeError("Starred should not be directly accessed"))

    def visit_name(self,
         identifier: str,
         ctx: ast.expr_context,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        if not isinstance(ctx, ast.Load):
            raise FeError(TypeError(f"Expected ast.Load for ctx, but got {type(ctx).__name__}"))
        return [], identifier

    def visit_list(self,
         elts: list[ast.expr],
         ctx: ast.expr_context,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            [elt0, *elt1, elt2]
        PIL:
            _tmp_0 = elt0
            _tmp_1 = elt1
            _tmp_2 = elt2
            _tmp_3 = [_tmp_0, *_tmp_1, _tmp_2]
        """
        if not isinstance(ctx, ast.Load):
            raise FeError(TypeError(f"Expected ast.Load for ctx, but got {type(ctx).__name__}"))

        return self._visit_sequence_literal(elts, self.create_pil_list)

    def visit_tuple(self,
         elts: list[ast.expr],
         ctx: ast.expr_context,
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        """
        Python:
            (elt0, *elt1, elt2)
        PIL:
            _tmp_0 = elt0
            _tmp_1 = elt1
            _tmp_2 = elt2
            _tmp_3 = (_tmp_0, *_tmp_1, _tmp_2)
        """
        if not isinstance(ctx, ast.Load):
            raise FeError(TypeError(f"Expected ast.Load for ctx, but got {type(ctx).__name__}"))

        return self._visit_sequence_literal(elts, self.create_pil_tuple)

    def visit_slice(self,
         lower: Optional[ast.expr],
         upper: Optional[ast.expr],
         step: Optional[ast.expr],
         node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt],
         PILExprOrNone]:
        raise FeError(NotImplementedError("Slice is not supported"))

    def visit_stmts(self, stmts: list[ast.stmt]) -> tuple[list[ast.stmt], PILExprOrNone]:
        stmt_list = []
        for stmt in stmts:
            result_stmt_list, _ = self.visit(stmt)
            stmt_list.extend(result_stmt_list)
        return stmt_list, None


    def visit(self, node):
        method = 'visit_' + self._node_name_to_visitor_suffix(node.__class__.__name__)
        visitor = getattr(self, method)
        field_dict = {key: value for key, value in ast.iter_fields(node)}
        # Rename fields to match PILAttr
        field_aliases = {
            'id': 'identifier',
            'iter': 'iter_expr',
            'slice': 'slice_expr',
            'str': 'literal_text',
        }
        field_dict = {field_aliases.get(key, key): value for key, value in field_dict.items()}
        node_attr = PILAttr(node)
        return visitor(**field_dict, node_attr=node_attr)

    def parse_func(self, func: ast.FunctionDef) -> ast.FunctionDef:
        body_stmt_list, _ = self.visit_stmts(func.body)
        result_func = ast.FunctionDef(
            func.name, func.args, body_stmt_list, func.decorator_list, func.returns, func.type_comment,
            **self.create_attribute(func))
        return result_func

    def parse_stmts(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        return self.visit_stmts(stmts)[0]



    def _collect_sequence_elements(self,
         elts: list[ast.expr]) -> tuple[list[ast.stmt],
         list[tuple[PILExpr, bool]]]:
        result_stmt_list = []
        elt_list = []
        for elt in elts:
            if isinstance(elt, ast.Starred):
                elt_stmt_list, elt_name = self.visit(elt.value)
                elt_list.append((elt_name, True))
            else:
                elt_stmt_list, elt_name = self.visit(elt)
                elt_list.append((elt_name, False))
            result_stmt_list.extend(elt_stmt_list)
        return result_stmt_list, elt_list

    def _visit_sequence_literal(self,
         elts: list[ast.expr],
         create_expr: Callable[[list[tuple[PILExpr, bool]]], PILExpr]) -> tuple[list[ast.stmt],
         str]:
        temp_name = self.create_temp_identifier()
        result_stmt_list, elt_list = self._collect_sequence_elements(elts)
        result_expr = create_expr(elt_list)
        result_stmt = self.create_pil_assign_name(temp_name, result_expr)
        result_stmt_list.append(result_stmt)
        return result_stmt_list, temp_name


def parse_func(func: ast.FunctionDef, prefix=PIL_DEFAULT_PREFIX) -> ast.FunctionDef:
    ctx = PILContext(prefix=prefix)
    parser = PythonParser(ctx)
    return parser.parse_func(func)


def parse_stmts(stmts: list[ast.stmt], prefix=PIL_DEFAULT_PREFIX) -> list[ast.stmt]:
    ctx = PILContext(prefix=prefix)
    parser = PythonParser(ctx)
    return parser.parse_stmts(stmts)
