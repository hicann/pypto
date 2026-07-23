#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile wrapper type for PyPTO Language DSL.

Tile represents a block in unified buffer memory, used for block-level programming.
"""

from __future__ import annotations

__all__ = ["Tile"]


from collections.abc import Sequence

from pypto.pypto_impl.ir import DataType, Expr, MemRef


class Tile:
    """Tile type for PyPTO Language DSL.

    Tile represents a block in unified buffer (UB) memory. It is used for
    block-level programming with operations like load, store, add, mul, etc.

    Annotation mode (used in type hints):
        x: pl.Tile[[64, 64], pl.DT_FP32]

    Runtime mode (wraps IR expressions):
        tile = pl.load(tensor, [0, 0], [64, 64])
        # Returns Tile wrapping the Call expression

    Examples:
        >>> import pypto_pro.language as pl
        >>>
        >>> @pl.function
        ... def my_func(input: pl.Tensor[[64, 64], pl.DT_FP32]) -> pl.Tensor[[64, 64], pl.DT_FP32]:
        ...     tile: pl.Tile[[64, 64], pl.DT_FP32] = pl.load(input, [0, 0], [64, 64])
        ...     result: pl.Tile[[64, 64], pl.DT_FP32] = pl.add(tile, tile)
        ...     return pl.store(result, [0, 0], [64, 64], input)
    """

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        memref: MemRef | None = None,
        _annotation_only: bool = False,
    ):
        """Initialize Tile.

        Args:
            shape: Shape (for annotation mode)
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            memref: Optional memory reference
            _annotation_only: Whether this is annotation-only mode
        """
        if _annotation_only:
            self.shape = shape
            self.dtype = dtype
            self.memref = memref
            self.expr = None
        elif expr is not None:
            self.expr = expr
            self.shape = None
            self.dtype = None
            self.memref = None
            self.valid_shape = None  # (row_expr, col_expr) or None
        else:
            raise ValueError(
                "Tile must be initialized with either (shape, dtype) for annotations or expr for runtime wrapping"
            )

    @classmethod
    def __class_getitem__(cls, item: tuple) -> "Tile":
        """Enable Tile[[shape], dtype] and Tile[[shape], dtype, memref] subscript syntax."""
        if not isinstance(item, tuple) or len(item) not in (2, 3):
            raise TypeError("Tile requires [shape, dtype] or [shape, dtype, memref] notation")

        if len(item) == 3:
            shape, dtype, memref = item
            if not isinstance(memref, MemRef):
                raise TypeError(f"Tile 3rd argument must be a MemRef instance, got {type(memref).__name__}")
            return cls(shape, dtype, memref=memref, _annotation_only=True)
        shape, dtype = item
        return cls(shape, dtype, _annotation_only=True)

    def __repr__(self) -> str:
        """String representation."""
        if self.expr is not None:
            return f"Tile(expr={self.expr})"
        if self.memref is not None:
            return f"Tile[[{self.shape}], {self.dtype}, {self.memref}]"
        return f"Tile[[{self.shape}], {self.dtype}]"

    def unwrap(self) -> Expr:
        """Get underlying IR expression.

        Returns:
            The wrapped Expr/Call object

        Raises:
            ValueError: If called on an annotation-only Tile
        """
        if self.expr is None:
            raise ValueError("Cannot unwrap annotation-only Tile (used in type hints)")
        return self.expr

    def set_expr(self, expr: Expr) -> None:
        """Update the wrapped IR expression for in-place DSL operations."""
        self.expr = expr
