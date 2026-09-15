# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Struct parsing helpers for ASTParser (pl.struct / pl.struct_array / pl.make_tuple)."""

from __future__ import annotations

import ast

from pypto.pypto_impl import ir
from pypto_pro.ir.op._op_registry import op_impl

from .diagnostics import ParserSyntaxError

# C++ keywords. A struct type name or field name is emitted verbatim into the
# generated C++ (``class Name { int64_t field; }``, ``s.field``), so a name that
# is a valid Python identifier but a C++ keyword (``int``, ``true``, ``delete``)
# produces uncompilable C++. Reject such names in the frontend with a clear
# diagnostic instead of letting them reach codegen.
_CPP_KEYWORDS = frozenset({
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
    "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
    "class", "compl", "concept", "const", "consteval", "constexpr", "constinit",
    "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype",
    "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
    "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
    "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept",
    "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private", "protected",
    "public", "register", "reinterpret_cast", "requires", "return", "short",
    "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
    "switch", "template", "this", "thread_local", "throw", "true", "try",
    "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual",
    "void", "volatile", "wchar_t", "while", "xor", "xor_eq",
})


class StructParserMixin:
    """Mixin containing ``pl.struct`` / ``pl.struct_array`` / ``pl.make_tuple`` helpers."""

    def _check_cpp_identifier(self, name: str, what: str, span: ir.Span) -> None:
        """Reject a struct type name / field name that is a C++ keyword.

        The name is emitted verbatim into the generated C++, so a C++ keyword
        would produce uncompilable code. Reject it here with a clear diagnostic.
        """
        if name in _CPP_KEYWORDS:
            raise ParserSyntaxError(
                f"{what} '{name}' is a C++ keyword and cannot be used as a struct "
                f"type name or field name",
                span=span,
                hint=f"Rename it, e.g. '{name}_'",
            )

    def _check_array_field_uniform(self, field_name: str, elem: ir.Expr, span: ir.Span) -> None:
        """Reject an empty or mixed-dtype fixed-size array field literal."""
        if not isinstance(elem, ir.MakeTuple):
            return
        elems = elem.elements
        if not elems:
            raise ParserSyntaxError(
                f"array field '{field_name}' is empty; a fixed-size array field "
                f"must have at least one element",
                span=span,
                hint="Provide initial values, e.g. [0, 0, 0, 0]",
            )
        first_dtype = elems[0].type.dtype
        for e in elems[1:]:
            if e.type.dtype != first_dtype:
                raise ParserSyntaxError(
                    f"array field '{field_name}' has mixed element types; "
                    f"all elements must be the same scalar type",
                    span=span,
                    hint="Use a single dtype, e.g. [1.0, 2.5, 3.0] or [1, 2, 3]",
                )

    @op_impl("make_tuple")
    def _parse_pl_make_tuple_expr(self, call: ast.Call) -> ir.Expr:
        span = self.span_tracker.get_span(call)
        if call.args:
            raise ParserSyntaxError(
                "pl.make_tuple() does not accept positional arguments; use keyword args",
                span=span,
            )
        if not call.keywords:
            raise ParserSyntaxError(
                "pl.make_tuple() requires at least one keyword argument (field=value)",
                span=span,
            )
        field_names: list[str] = []
        elements: list[ir.Expr] = []
        for kw in call.keywords:
            if kw.arg is None:
                raise ParserSyntaxError("pl.make_tuple() does not support **kwargs", span=span)
            field_names.append(kw.arg)
            elements.append(self.parse_expression(kw.value, nested=True))
        return self.make_named_tuple(elements, field_names, span)

    @op_impl("struct")
    def _parse_pl_struct_expr(self, call: ast.Call) -> ir.Expr:
        span = self.span_tracker.get_span(call)
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            raise ParserSyntaxError(
                'pl.struct("Name", ...) requires exactly one string struct name as first argument',
                span=span,
                hint='Use pl.struct("Name", field1=val1, ...)',
            )
        struct_name = call.args[0].value
        self._check_cpp_identifier(struct_name, "struct type name", span)
        if not call.keywords:
            raise ParserSyntaxError(
                'pl.struct("Name", ...) requires at least one keyword field',
                span=span,
            )
        field_names: list[str] = []
        elements: list[ir.Expr] = []
        for kw in call.keywords:
            if kw.arg is None:
                raise ParserSyntaxError("pl.struct() does not support **kwargs", span=span)
            self._check_cpp_identifier(kw.arg, "struct field name", span)
            elem = self.parse_expression(kw.value, nested=True)
            if self.named_fields(elem):
                raise ParserSyntaxError(
                    f"pl.struct() field '{kw.arg}' is a nested named tuple/struct, which is not "
                    f"supported; struct fields must be scalars or fixed-size arrays "
                    f"(list literals like [0, 0, 0, 0])",
                    span=span,
                    hint='Flatten the nested fields into this struct, e.g. pl.struct("Name", x=...)',
                )
            self._check_array_field_uniform(kw.arg, elem, span)
            field_names.append(kw.arg)
            elements.append(elem)
        call = ir.create_op_call(
            "struct.create",
            elements,
            {"name": struct_name, "fields": field_names},
            span,
        )
        return self.register_struct_type(call, struct_name, field_names)

    @op_impl("struct_array")
    def _parse_struct_array_expr(self, call: ast.Call) -> ir.Expr:
        """Handle ``var = pl.struct_array(N, "StructName", field1=val1, ...)``.

        Lowers to N ``struct.create`` slots wrapped in a MakeTuple.
        The caller (_parse_name_assignment) emits the final let-binding.
        """
        span = self.span_tracker.get_span(call)
        if not call.args or not isinstance(call.args[0], ast.Constant):
            raise ParserSyntaxError(
                "pl.struct_array() requires an integer size as first argument",
                span=span,
                hint='Use pl.struct_array(N, "Name", field1=0, field2=0, ...)',
            )
        arr_size = call.args[0].value
        if not isinstance(arr_size, int) or arr_size < 1:
            raise ParserSyntaxError(
                f"pl.struct_array() size must be a positive integer, got {arr_size}",
                span=span,
            )
        if not (len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)):
            raise ParserSyntaxError(
                'pl.struct_array(N, "Name", ...) requires a string struct name as second arg',
                span=span,
            )
        struct_name = call.args[1].value
        self._check_cpp_identifier(struct_name, "struct type name", span)
        if len(call.args) > 2:
            raise ParserSyntaxError(
                f"pl.struct_array() accepts exactly two positional arguments "
                f"(size and name), got {len(call.args)}",
                span=span,
            )
        if not call.keywords:
            raise ParserSyntaxError(
                'pl.struct_array(N, "Name", ...) requires at least one keyword field',
                span=span,
            )
        var_name = self.current_target_name
        field_names: list[str] = []
        field_inits: list[ir.Expr] = []
        for kw in call.keywords:
            if kw.arg is None:
                raise ParserSyntaxError("pl.struct_array() does not support **kwargs", span=span)
            self._check_cpp_identifier(kw.arg, "struct field name", span)
            elem = self.parse_expression(kw.value, nested=True)
            if self.named_fields(elem):
                raise ParserSyntaxError(
                    f"pl.struct_array() field '{kw.arg}' is a nested named tuple/struct, which is not "
                    f"supported; struct fields must be scalars or fixed-size arrays "
                    f"(list literals like [0, 0, 0, 0])",
                    span=span,
                    hint='Flatten the nested fields into this struct, e.g. pl.struct_array(N, "Name", x=...)',
                )
            self._check_array_field_uniform(kw.arg, elem, span)
            field_names.append(kw.arg)
            field_inits.append(elem)
        slot_vars: list[ir.Expr] = []
        for i in range(arr_size):
            slot_call = ir.create_op_call(
                "struct.create",
                field_inits,
                {"name": struct_name, "fields": field_names},
                span,
            )
            self.register_struct_type(slot_call, struct_name, field_names)
            slot_var = self.builder.let(f"{var_name}_{i}", slot_call, span=span)
            slot_vars.append(slot_var)
        result = ir.MakeTuple(slot_vars, span)
        self._struct_array_tuples.add(result)
        return result
