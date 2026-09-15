# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser-session metadata for the semantic kinds represented by ``TupleType``."""

from __future__ import annotations

from pypto.pypto_impl import ir

TupleTypeKind = ir.TupleTypeKind
TupleTypeInfo = ir.TupleTypeInfo


_PLAIN_TUPLE_INFO = TupleTypeInfo(TupleTypeKind.TUPLE)


class TupleTypeRegistry:
    """Register and query semantic tuple metadata stored in ``IRDebugInfo``."""

    def __init__(self, debug_info: ir.IRDebugInfo):
        if debug_info is None:
            raise ValueError("TupleTypeRegistry requires a non-null IRDebugInfo")
        self._debug_info = debug_info

    def _register(self, tuple_type: ir.TupleType, info: TupleTypeInfo) -> None:
        if len(info.fields) != len(tuple_type.types):
            raise ValueError(
                f"Tuple field count {len(info.fields)} does not match element count {len(tuple_type.types)}"
            )
        self._debug_info.register_tuple_type_info(tuple_type, info)

    def register_named_tuple(
        self,
        tuple_type: ir.TupleType,
        fields: list[str] | tuple[str, ...],
        *,
        name: str | None = None,
    ) -> None:
        """Register a user or framework named tuple."""
        self._register(tuple_type, TupleTypeInfo(TupleTypeKind.NAMED_TUPLE, name, list(fields)))

    def register_struct(
        self,
        tuple_type: ir.TupleType,
        name: str,
        fields: list[str] | tuple[str, ...],
    ) -> None:
        """Register a C++ struct."""
        self._register(tuple_type, TupleTypeInfo(TupleTypeKind.STRUCT, name, list(fields)))

    def classify(self, tuple_type: ir.TupleType) -> TupleTypeInfo:
        """Return the semantic classification of one exact tuple type."""
        if not isinstance(tuple_type, ir.TupleType):
            raise TypeError(f"Expected TupleType, got {type(tuple_type).__name__}")
        info = self._debug_info.get_tuple_type_info(tuple_type)
        return info if info is not None else _PLAIN_TUPLE_INFO

    def types_equal(self, lhs: ir.Type, rhs: ir.Type) -> bool:
        """Compare IR types and tuple metadata recursively."""
        if not ir.structural_equal(lhs, rhs, enable_auto_mapping=False):
            return False
        if not isinstance(lhs, ir.TupleType) or not isinstance(rhs, ir.TupleType):
            return True
        if self.classify(lhs) != self.classify(rhs):
            return False
        return all(
            self.types_equal(left, right) for left, right in zip(lhs.types, rhs.types, strict=True)
        )
