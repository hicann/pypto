# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""make_tile_group parsing helpers for ASTParser.

``pl.make_tile_group(type=, addrs=, mutex_ids=)`` declares a rotating group of
tiles inside a kernel. The parser lowers it to an IR named-tuple handle with
fields ``tiles`` (tuple of tile vars), ``mutex_ids`` (tuple of ConstInt) and a
mutable ``cursor`` struct. The handle's ``next()/current()/previous()`` methods
return a bare tile (not a slot); the tile <-> mutex_id mapping is kept as parser
metadata (``_tile_mutex_meta``) and consumed by auto_mutex, so the runtime never
handles mutex ids or manual lock/unlock.
"""

from __future__ import annotations

import ast
from functools import reduce

from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType

from pypto_pro.ir.op._op_registry import op_impl
from pypto_pro.ir.op.block_ops import make_tile as _ir_make_tile
from pypto_pro.ir.op.block_ops import TileType as _TileType

from .diagnostics import ParserSyntaxError, ParserTypeError


class BufferParserMixin:
    """make_tile_group DSL composite: parse pl.make_tile_group(...) into an IR named tuple."""

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _validate_mutex_ids(mutex_ids, span: ir.Span) -> None:
        if not mutex_ids:
            raise ParserTypeError("make_tile_group() mutex_ids must be a non-empty list", span=span)
        for m in mutex_ids:
            if not isinstance(m, int) or m < 0 or m > 31:
                raise ParserTypeError(f"mutex_ids must be ints in [0,31], got {m!r}", span=span)
        if len(set(mutex_ids)) != len(mutex_ids):
            raise ParserTypeError(f"mutex_ids must be unique, got {mutex_ids}", span=span)

    @staticmethod
    def _tile_type_slot_size(tile_type: _TileType) -> int:
        """Per-tile byte size derived from the TileType static shape and dtype."""
        shape = tuple(tile_type.shape)
        elems = reduce(lambda a, b: a * b, shape, 1)
        return int(elems) * max(1, (int(tile_type.dtype.get_bit()) + 7) // 8)

    # --- factory --------------------------------------------------------------

    @op_impl("make_tile_group")
    def _parse_make_tile_group(self, call: ast.Call) -> ir.Expr:
        span = self.span_tracker.get_span(call)
        if call.args:
            raise ParserSyntaxError(
                "pl.make_tile_group() takes keyword args only (type=, addrs=, mutex_ids=)", span=span)
        kw = {k.arg: k for k in call.keywords if k.arg is not None}
        for required in ("type", "addrs", "mutex_ids"):
            if required not in kw:
                raise ParserTypeError(
                    f"pl.make_tile_group() missing required keyword '{required}'", span=span)

        tile_type = self.parse_expression(kw["type"].value)
        if not isinstance(tile_type, _TileType):
            raise ParserTypeError("pl.make_tile_group() 'type' must be a pl.TileType", span=span)

        mutex_ids = tuple(self.expr_evaluator.eval_expr(kw["mutex_ids"].value))
        self._validate_mutex_ids(mutex_ids, span)
        num = len(mutex_ids)

        slot_size = self._tile_type_slot_size(tile_type)
        # addrs: a single base address -> contiguous tiles (base + i*slot_size);
        # a list -> one explicit address per tile (non-contiguous).
        addrs = self.expr_evaluator.eval_expr(kw["addrs"].value)
        if isinstance(addrs, (list, tuple)):
            tile_addrs = list(addrs)
            if len(tile_addrs) != num:
                raise ParserTypeError(
                    f"make_tile_group() addrs length {len(tile_addrs)} must equal "
                    f"mutex_ids length {num}", span=span)
        else:
            tile_addrs = [addrs + i * slot_size for i in range(num)]

        return self._build_tile_group_ir(tile_type, tile_addrs, mutex_ids, slot_size, span)

    def is_tile_group(self, expr) -> bool:
        return "tiles" in self.named_fields(expr)

    def _build_tile_group_ir(self, tile_type, tile_addrs, mutex_ids, slot_size, span: ir.Span) -> ir.Expr:
        """Build the IR named-tuple handle for a tile group.

        Multi-tile groups carry a rotating ``cursor`` struct ({tiles, mutex_ids,
        cursor}); single-tile groups skip it ({tiles, mutex_ids}) since the only
        tile and its mutex id are statically determined.

        Sets pending metadata ``_tile_group_meta_pending`` consumed by
        ``_consume_nbuf_pending`` after the caller's builder.let.
        """
        var_name = self.current_target_name or "g"
        num = len(mutex_ids)
        tile_vars = []
        for i, addr in enumerate(tile_addrs):
            t = _ir_make_tile(
                shape=tile_type.shape, dtype=tile_type.dtype,
                target_memory=tile_type.target_memory,
                layout=tile_type.layout,
                fractal=tile_type.fractal, pad=tile_type.pad, compact=tile_type.compact,
                valid_shape=tile_type.valid_shape,
                addr=addr, size=slot_size)
            tile_vars.append(self.builder.let(f"_tg_{var_name}_tiles_{i}", t, span=span))
        tiles_tuple = self.builder.let(f"_tg_{var_name}_tiles", ir.MakeTuple(tile_vars, span), span=span)
        mut_tuple = self.builder.let(
            f"_tg_{var_name}_mutex_ids",
            ir.MakeTuple([ir.ConstInt(m, DataType.INDEX, span) for m in mutex_ids], span), span=span)
        self._tile_group_meta_pending = (num, mutex_ids)
        if num == 1:
            # Single tile: no cursor needed; accessors short-circuit to tiles[0].
            return self.make_named_tuple([tiles_tuple, mut_tuple], ["tiles", "mutex_ids"], span)
        cursor_call = ir.create_op_call(
            "struct.create", [ir.ConstInt(num - 1, DataType.INDEX, span)],
            {"name": "_TileGroupCursor", "fields": ["cursor"]}, span)
        self.register_struct_fields(cursor_call, ["cursor"])
        cursor = self.builder.let(f"_tg_{var_name}_cursor", cursor_call, span=span)
        return self.make_named_tuple([tiles_tuple, mut_tuple, cursor], ["tiles", "mutex_ids", "cursor"], span)

    # --- accessors ------------------------------------------------------------

    def _emit_cursor_advance(self, cursor_struct, new_val, span: ir.Span) -> None:
        self.builder.emit(ir.EvalStmt(
            ir.create_op_call("struct.set", [cursor_struct, new_val], {"field": "cursor"}, span), span))

    def _lower_group_accessor(self, group_var, method_name: str, span: ir.Span):
        """Lower group.next()/current()/previous() to a bare tile.

        next() advances the cursor (+1) then returns the tile at the new index;
        current()/previous() leave the cursor unchanged. Records the selected
        tile's mutex id via _tile_mutex_pending for auto_mutex.
        """
        n_slots = self.tile_group_meta.get(id(group_var), (1, None))[0]
        mutex_values = self.tile_group_meta.get(id(group_var), (1, None))[1]
        tiles = self.lower_attr_access(group_var, "tiles", span)
        if n_slots == 1:
            # Single tile: no cursor, no modulo. Select tiles[0] with a const index
            # (folded by ConstFoldAndSimplify) and lock the lone mutex id statically.
            tile_ir = ir.GetItemExpr(tiles, ir.ConstInt(0, DataType.INDEX, span), span)
            self._tile_mutex_pending = (mutex_values[0], mutex_values)
            return tile_ir
        n_const = ir.ConstInt(n_slots, DataType.INDEX, span)
        mut = self.lower_attr_access(group_var, "mutex_ids", span)
        cursor_struct = self.lower_attr_access(group_var, "cursor", span)
        cur_read = self.lower_attr_access(cursor_struct, "cursor", span)
        if method_name == "next":
            new_val = cur_read + ir.ConstInt(1, DataType.INDEX, span)
            self._emit_cursor_advance(cursor_struct, new_val, span)
            raw = new_val % n_const
        elif method_name == "current":
            # next() stores ``cursor = old + 1`` and then (its index re-reads the just-advanced
            # cursor) returns slot ``(cursor + 1) % n``. current() must return that same slot, so it
            # is ``(cursor + 1) % n`` — not ``cursor % n``, which trails next() by one and makes
            # current() read a tile that the preceding next() never selected.
            raw = (cur_read + ir.ConstInt(1, DataType.INDEX, span)) % n_const
        else:  # previous
            raw = cur_read % n_const
        idx = self.builder.let(f"_bufidx_{self._tuple_idx_counter}", raw, span=span)
        self._tuple_idx_counter += 1
        tile_ir = ir.GetItemExpr(tiles, idx, span)
        buf_id_ir = ir.GetItemExpr(mut, idx, span)
        self._tile_mutex_pending = (buf_id_ir, mutex_values)
        return tile_ir
