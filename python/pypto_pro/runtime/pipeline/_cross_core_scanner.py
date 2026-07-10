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

"""Cross-core access scanner for preload pipeline auto-sync.

Pure-AST analysis. Scans the kernel body for cross-core NBuffer declarations
(those configured with cross_core_forward_id / cross_core_backward_id), then
scans each @stage function body to determine, for each cross-core buffer it
touches, the access role (W/R) and the pipe of the op that does the access.

The result drives automatic wait/set_cross_core insertion at stage boundaries.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum

from pypto.pypto_impl.ir import MemorySpace
from pypto_pro.language.parser._op_pipeline import (
    _BLOCK_OP_TILE_ROLES,
    get_move_pipe,
    get_op_pipe,
    get_store_pipe,
)

# MemorySpace attribute name (as written in pl.MemorySpace.<X>) -> enum value
_MEMORY_NAMES = {
    "Vec": MemorySpace.Vec,
    "Mat": MemorySpace.Mat,
    "Left": MemorySpace.Left,
    "Right": MemorySpace.Right,
    "Acc": MemorySpace.Acc,
}


def _extract_memory_from_make_tile_group(call: ast.Call) -> MemorySpace | None:
    """From a pl.make_tile_group(type=pl.TileType(..., target_memory=pl.MemorySpace.X), ...)
    call, extract the MemorySpace X. Returns None if not resolvable."""
    type_node = None
    for kw in call.keywords:
        if kw.arg == "type":
            type_node = kw.value
            break
    if not (isinstance(type_node, ast.Call)):
        return None
    for kw in type_node.keywords:
        if kw.arg == "target_memory":
            v = kw.value
            if isinstance(v, ast.Attribute) and v.attr in _MEMORY_NAMES:
                return _MEMORY_NAMES[v.attr]
    return None


def _is_make_tile_group(call: ast.Call) -> bool:
    """True if call is pl.make_tile_group(...) (or bare make_tile_group(...))."""
    return _get_ctor_name(call) == "make_tile_group"


@dataclass
class CrossCoreBuffer:
    """A cross-core NBuffer's configuration (from kernel-body declaration)."""
    name: str                  # variable name, e.g. "qk_vec_db"
    memory: MemorySpace        # buffer memory space
    fwd_ids_node: ast.expr | None  # AST node for cross_core_forward_id value (e.g. a Name)
    bwd_ids_node: ast.expr | None  # AST node for cross_core_backward_id value
    fwd_slot_count: int        # len(forward id tuple) - for % N indexing
    bwd_slot_count: int        # len(backward id tuple)


class AccessRole(Enum):
    WRITE = "W"
    READ = "R"
    READ_WRITE = "RW"


@dataclass
class CrossCoreAccess:
    """One cross-core buffer access by a stage.

    A buffer may be accessed by multiple ops (same role) on different pipes.
    Sync at stage boundary needs:
      - first_pipe: pipe of the FIRST op  -> used by the *pre* (wait) sync
      - last_pipe:  pipe of the LAST op   -> used by the *post* (set) sync
    For a single-op access first_pipe == last_pipe.
    """
    buffer_name: str
    role: AccessRole
    first_pipe: str            # PipeType name of the first access op (for wait)
    last_pipe: str             # PipeType name of the last access op (for set)
    buffer: CrossCoreBuffer = None  # back-reference to the buffer config
    section_kind: str = ""     # for mix outer accesses: which sub-stage section
                               # ("cube"/"vector") this access happens in, so the
                               # outer sync can be wrapped in the right section.


@dataclass
class CrossCoreSyncContext:
    """Cross-core synchronization context — buffer declarations + access metadata.

    Groups all information needed by the transformer to generate cross-core sync
    (wait/set_cross_core), pre-fire, and lifted-id declarations. Populated by
    _scan_cross_core() during analysis.
    """
    # Buffer declarations: name -> CrossCoreBuffer (only cross-core buffers with fwd/bwd ids)
    buffers: dict = field(default_factory=dict)
    # All buffer memory spaces: name -> MemorySpace (includes local buffers, for pipe calc)
    all_memory: dict = field(default_factory=dict)
    # Inner buffer consumer info for mix pre-fire: {buffer_name: (section_kind, pipe)}
    inner_consumer_map: dict = field(default_factory=dict)
    # Lifted literal fwd_ids/bwd_ids: [(var_name, ast_literal)] to declare as variables
    lifted_ids: list = field(default_factory=list)


def _memory_name(mem: MemorySpace) -> str:
    """MemorySpace enum -> short pipe-table-compatible name. Not used directly;
    pipe is computed via the dynamic helpers or C++ backend metadata."""
    return str(mem)


def scan_all_buffer_memory(kernel_func_def: ast.FunctionDef) -> dict[str, MemorySpace]:
    """Return every make_tile_group declaration's memory by buffer variable name."""
    result: dict[str, MemorySpace] = {}
    for node in ast.walk(kernel_func_def):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        if not _is_make_tile_group(node.value):
            continue
        mem = _extract_memory_from_make_tile_group(node.value)
        if mem is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = mem
    return result


def scan_cross_core_buffers(kernel_func_def: ast.FunctionDef,
                            closure_vars: dict) -> tuple[dict[str, CrossCoreBuffer], list]:
    """Scan the kernel body for cross-core tile-group declarations.

    Returns:
        (dict mapping buffer variable name -> CrossCoreBuffer,
         lifted_ids: list of (var_name, ast_literal) for literal fwd/bwd ids
         that need to be declared as variables before the pipeline loop)
    """
    result: dict[str, CrossCoreBuffer] = {}
    lifted_ids: list[tuple[str, ast.expr]] = []

    for node in ast.walk(kernel_func_def):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not _is_make_tile_group(call):
            continue

        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        fwd_node = kwargs.get("fwd_ids")
        bwd_node = kwargs.get("bwd_ids")
        if fwd_node is None and bwd_node is None:
            continue

        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        bufname = names[0] if names else "<tile_group>"

        memory = _validate_buffer_memory(call, bufname)
        fwd_count, bwd_count = _validate_ids(fwd_node, bwd_node, bufname, closure_vars)
        fwd_node, bwd_node = _lift_literal_ids(fwd_node, bwd_node, bufname, lifted_ids)

        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = CrossCoreBuffer(
                    name=target.id,
                    memory=memory,
                    fwd_ids_node=fwd_node,
                    bwd_ids_node=bwd_node,
                    fwd_slot_count=fwd_count,
                    bwd_slot_count=bwd_count,
                )

    return result, lifted_ids


def _validate_buffer_memory(call: ast.Call, bufname: str) -> MemorySpace:
    """L8: Validate and extract memory space from make_tile_group call."""
    memory = _extract_memory_from_make_tile_group(call)
    if memory is None:
        raise ValueError(
            f"pipeline: cross-core buffer '{bufname}' has no resolvable memory space. "
            f"Its make_tile_group(type=pl.TileType(..., target_memory=pl.MemorySpace.X)) "
            f"must set target_memory to a literal pl.MemorySpace.<X>."
        )
    return memory


def _validate_ids(fwd_node, bwd_node, bufname: str, closure_vars: dict) -> tuple[int, int]:
    """L11: Validate fwd_ids/bwd_ids are resolvable non-empty tuples."""
    fwd_count = _resolve_tuple_len(fwd_node, closure_vars)
    bwd_count = _resolve_tuple_len(bwd_node, closure_vars)
    if fwd_node is not None and fwd_count == 0:
        raise ValueError(
            f"pipeline: cross-core buffer '{bufname}' fwd_ids must be a non-empty tuple "
            f"(literal or a module-level tuple constant); could not resolve its length."
        )
    if bwd_node is not None and bwd_count == 0:
        raise ValueError(
            f"pipeline: cross-core buffer '{bufname}' bwd_ids must be a non-empty tuple "
            f"(literal or a module-level tuple constant); could not resolve its length."
        )
    return fwd_count, bwd_count


def _lift_literal_ids(fwd_node, bwd_node, bufname: str,
                      lifted_ids: list) -> tuple[ast.expr, ast.expr]:
    """Lift literal fwd_ids/bwd_ids to variable names for codegen compatibility."""
    if fwd_node is not None and isinstance(fwd_node, (ast.Tuple, ast.List)):
        var_name = f"_pl_fwd_ids_{bufname}"
        lifted_ids.append((var_name, fwd_node))
        fwd_node = ast.Name(id=var_name, ctx=ast.Load())
    if bwd_node is not None and isinstance(bwd_node, (ast.Tuple, ast.List)):
        var_name = f"_pl_bwd_ids_{bufname}"
        lifted_ids.append((var_name, bwd_node))
        bwd_node = ast.Name(id=var_name, ctx=ast.Load())
    return fwd_node, bwd_node


def _get_slot_accessor_assignment(node: ast.AST, param_names: set[str]) -> tuple[str, str] | None:
    """Return slot variable and source buffer for ``slot = group.next()`` patterns."""
    if not isinstance(node, ast.Assign):
        return None
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None
    if not isinstance(node.value, ast.Call):
        return None

    func = node.value.func
    if not isinstance(func, ast.Attribute) or func.attr not in ("next", "current", "previous"):
        return None
    if not isinstance(func.value, ast.Name) or func.value.id not in param_names:
        return None
    return node.targets[0].id, func.value.id


def _build_slot_to_buffer(stage_func_def: ast.FunctionDef, param_names: set[str]) -> dict[str, str]:
    """Map slot variables from group accessors to their source buffer names."""
    slot_to_buffer: dict[str, str] = {}
    for node in ast.walk(stage_func_def):
        slot_assignment = _get_slot_accessor_assignment(node, param_names)
        if slot_assignment is not None:
            slot_name, buffer_name = slot_assignment
            slot_to_buffer[slot_name] = buffer_name
    return slot_to_buffer


def _scan_all_accesses(
    stage_func_def: ast.FunctionDef,
    slot_to_buffer: dict[str, str],
    cross_buffers: dict[str, CrossCoreBuffer],
    all_buffer_memory: dict[str, MemorySpace],
    vf_func_defs: dict[str, ast.FunctionDef],
    access: dict[str, tuple[str, str, str]],
) -> None:
    """Scan all op calls in SOURCE ORDER for cross-core accesses.

    Uses a single DFS pass (_iter_calls_in_order) to ensure first_pipe/last_pipe
    reflect true source order. Delegates to _handle_block_op / _handle_vf_call.
    """
    for node in _iter_calls_in_order(stage_func_def):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "pl":
            _handle_block_op(node, slot_to_buffer, cross_buffers, all_buffer_memory, access)
        elif isinstance(node.func, ast.Name):
            _handle_vf_call(node, slot_to_buffer, cross_buffers, vf_func_defs, access)


def _handle_block_op(node: ast.Call, slot_to_buffer, cross_buffers, all_buffer_memory, access):
    """Process a single pl.<op>(...) call for cross-core access."""
    op_name = node.func.attr
    roles = _BLOCK_OP_TILE_ROLES.get(op_name)
    if roles is None:
        # C8: unknown op operating on cross-core buffer
        for arg in node.args:
            buf = _tile_arg_buffer(arg, slot_to_buffer)
            if buf is not None and buf in cross_buffers:
                raise ValueError(
                    f"pipeline: op 'pl.{op_name}' operates on cross-core buffer "
                    f"'{buf}' but is not in the op-role table (_BLOCK_OP_TILE_ROLES). "
                    f"The scanner cannot determine whether this is a read or write."
                )
        return
    pipe = None
    for argpos, arg in enumerate(node.args):
        buf = _tile_arg_buffer(arg, slot_to_buffer)
        if buf is None or buf not in cross_buffers or argpos >= len(roles):
            continue
        role = roles[argpos]
        if role is None:
            continue
        if pipe is None:
            pipe = _block_op_pipe(op_name, node, slot_to_buffer, all_buffer_memory)
        _record_access(access, buf, role, pipe)


def _handle_vf_call(node: ast.Call, slot_to_buffer, cross_buffers, vf_func_defs, access):
    """Process a single VF helper call for cross-core access."""
    vf_name = node.func.id
    vf_def = vf_func_defs.get(vf_name)
    if vf_def is None:
        return
    vf_roles = _scan_vf_roles(vf_def, vf_func_defs)
    vf_params = [a.arg for a in vf_def.args.args if a.arg != "self"]
    for argpos, arg in enumerate(node.args):
        buf = _tile_arg_buffer(arg, slot_to_buffer)
        if buf is None or buf not in cross_buffers or argpos >= len(vf_params):
            continue
        role = vf_roles.get(vf_params[argpos])
        if role is None:
            continue
        _record_access(access, buf, role, "V")


def _iter_calls_in_order(node: ast.AST):
    """Yield all ast.Call nodes in SOURCE order (DFS, child order)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Call):
            yield child
        yield from _iter_calls_in_order(child)


def scan_stage_accesses(stage_func_def: ast.FunctionDef,
                        cross_buffers: dict[str, CrossCoreBuffer],
                        vf_func_defs: dict[str, ast.FunctionDef],
                        all_buffer_memory: dict[str, MemorySpace]) -> list[CrossCoreAccess]:
    """Scan a stage function body for cross-core buffer accesses.

    Args:
        stage_func_def: the @stage function AST
        cross_buffers: name -> CrossCoreBuffer (cross-core buffers)
        vf_func_defs: name -> FunctionDef for VF helper functions (for role scan)
        all_buffer_memory: name -> MemorySpace for ALL buffers (for move pipe)

    Returns a list of CrossCoreAccess (one per cross-core buffer the stage touches).
    """
    param_names = [a.arg for a in stage_func_def.args.args if a.arg != "self"]
    param_name_set = set(param_names)
    buf_params = {p for p in param_names if p in cross_buffers}
    if not buf_params:
        return []

    # Map: slot_var_name -> buffer_name, for ALL buffer params (not just cross-core),
    # so move/store pipe can resolve the non-cross-core side's memory.
    # slot = group.next() (or .current()/.previous()); .next() returns a bare tile.
    slot_to_buffer = _build_slot_to_buffer(stage_func_def, param_name_set)

    # L10: validate slot accessor forms for cross-core buffers
    _validate_slot_accessors(stage_func_def, param_name_set, cross_buffers)

    # buffer_name -> (role, first_pipe, last_pipe). A buffer may be accessed by
    # multiple ops in one stage (same role per contract); first/last pipe tracked
    # for pre(wait)/post(set) sync respectively.
    access: dict[str, tuple[str, str, str]] = {}

    # Single source-order pass for both block ops and VF calls
    _scan_all_accesses(stage_func_def, slot_to_buffer, cross_buffers,
                        all_buffer_memory, vf_func_defs, access)

    # Build CrossCoreAccess list
    out: list[CrossCoreAccess] = []
    for buf_name, (role, first_pipe, last_pipe) in access.items():
        out.append(CrossCoreAccess(
            buffer_name=buf_name, role=AccessRole(role),
            first_pipe=first_pipe, last_pipe=last_pipe,
            buffer=cross_buffers[buf_name],
        ))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_slot_accessors(stage_func_def: ast.FunctionDef,
                             param_names: set[str],
                             cross_buffers: dict) -> None:
    """L10: cross-core buffer slot accessors must be `slot = buf.next()` form."""
    accessor_rhs_ids = set()
    for node in ast.walk(stage_func_def):
        sa = _get_slot_accessor_assignment(node, param_names)
        if sa is not None:
            accessor_rhs_ids.add(id(node.value))
    for node in ast.walk(stage_func_def):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("next", "current", "previous"):
            continue
        if not (isinstance(node.func.value, ast.Name)
                and node.func.value.id in cross_buffers):
            continue
        if id(node) in accessor_rhs_ids:
            continue
        raise ValueError(
                f"pipeline: cross-core buffer '{node.func.value.id}' slot accessor "
                f"`.{node.func.attr}()` must be assigned to a simple variable "
                f"(`slot = {node.func.value.id}.{node.func.attr}()`); inline/chained/"
                f"tuple-unpack forms are not supported."
            )


def _get_ctor_name(call: ast.Call) -> str | None:
    """Get the constructor name from a call like pl.UBNBuffer(...) -> 'UBNBuffer'."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _resolve_tuple_len(node: ast.expr | None, closure_vars: dict) -> int:
    """Resolve the length of a tuple-valued node (Name referencing a module-level
    tuple, or a literal Tuple)."""
    if node is None:
        return 0
    if isinstance(node, (ast.Tuple, ast.List)):
        return len(node.elts)
    if isinstance(node, ast.Name):
        val = closure_vars.get(node.id)
        if isinstance(val, (tuple, list)):
            return len(val)
    return 0


def _tile_arg_buffer(arg: ast.expr, slot_to_buffer: dict[str, str]) -> str | None:
    """If arg is a bare slot variable (from group.next()) that maps to a buffer,
    return the buffer name; else None.

    New API: group.next() returns a bare tile, so op args are plain Names
    (e.g. pl.move(qk_left, cur_k)), not `slot.tile` attributes.
    """
    if isinstance(arg, ast.Name):
        return slot_to_buffer.get(arg.id)
    return None


def _block_op_pipe(op_name: str, call: ast.Call,
                   slot_to_buffer: dict[str, str],
                   all_buffer_memory: dict[str, MemorySpace]) -> str:
    """Determine the pipe name for a block op accessing a cross-core buffer."""
    if op_name == "move":
        # move(dst, src): pipe depends on src/dst memory
        dst_mem = _arg_memory(call.args[0] if call.args else None, slot_to_buffer, all_buffer_memory)
        src_mem = _arg_memory(call.args[1] if len(call.args) > 1 else None, slot_to_buffer, all_buffer_memory)
        if src_mem is not None and dst_mem is not None:
            return _pipe_name(get_move_pipe(src_mem, dst_mem))
        return "V"
    if op_name in ("store", "store_tile"):
        src_mem = _arg_memory(call.args[1] if len(call.args) > 1 else None, slot_to_buffer, all_buffer_memory)
        if src_mem is not None:
            return _pipe_name(get_store_pipe(src_mem))
        return "V"
    pipe = get_op_pipe(op_name)
    return _pipe_name(pipe) if pipe is not None else "V"


def _arg_memory(arg, slot_to_buffer, all_buffer_memory) -> MemorySpace | None:
    """Get the MemorySpace of a `slot.tile` arg (any buffer, cross-core or local)."""
    buf = _tile_arg_buffer(arg, slot_to_buffer) if arg is not None else None
    if buf is not None:
        return all_buffer_memory.get(buf)
    return None


def _pipe_name(pipe) -> str:
    """PipeType enum -> short name string used in generated pl.PipeType.<NAME>."""
    # pipe is a PipeType enum; its name attribute gives FIX/V/MTE1/...
    return getattr(pipe, "name", str(pipe).split(".")[-1])


def _record_access(access: dict, buf: str, role: str, pipe: str) -> None:
    """Record a buffer access (called in SOURCE order). Tracks both the first
    op's pipe and the last op's pipe for the buffer.

    Contract: within one stage a buffer has a single role (all W for a producer,
    all R for a consumer — never both). The sync at the stage boundary uses:
      - pre (wait, before stage): the FIRST op's pipe
      - post (set, after stage):  the LAST op's pipe

    Stored as access[buf] = (role, first_pipe, last_pipe). First op sets both;
    subsequent ops (source order) only update last_pipe.
    """
    existing = access.get(buf)
    if existing is None:
        access[buf] = (role, pipe, pipe)   # first==last on first op
        return
    ex_role, first_pipe, _ = existing
    # L9: a stage must be EITHER producer (all W) OR consumer (all R) of a given
    # cross-core buffer, never both.
    if ex_role != role:
        raise ValueError(
            f"pipeline: cross-core buffer '{buf}' is both read and written within a "
            f"single stage (roles {ex_role} and {role}). A stage must be either the "
            f"producer (write-only) or the consumer (read-only) of a cross-core buffer."
        )
    access[buf] = (ex_role, first_pipe, pipe)  # keep first, update last


def _root_name(node):
    """Return the root name of a possibly indexed/attributed expression."""
    cur = node
    while True:
        if isinstance(cur, ast.Name):
            return cur.id
        if isinstance(cur, ast.BinOp):
            cur = cur.left
            continue
        if isinstance(cur, (ast.Attribute, ast.Subscript)):
            cur = cur.value
            continue
        return None


def _merge_vf_role(result: dict[str, str], name: str, role: str | None) -> None:
    """Merge one VF read/write role into the per-param role map."""
    ex = result.get(name)
    if ex is None:
        result[name] = role
    elif ex != role and role is not None:
        result[name] = "RW"


def _record_vf_call_role(sub: ast.Call, param_names: set[str], result: dict[str, str]) -> None:
    """Record the R/W role represented by a single vf load/store call.

    Handles both statement form (vf.load_align(dst_reg, ptr, ...)) and
    assignment form (ptr_arg = vf.load_align(ptr, ...)).
    """
    if not isinstance(sub.func, ast.Attribute):
        return
    # Only vf.* calls are relevant (guards against pl.load etc. in non-VF helpers)
    if not (isinstance(sub.func.value, ast.Name) and sub.func.value.id == "vf"):
        return
    op = sub.func.attr
    if op.startswith("load") and len(sub.args) >= 1:
        root = None
        if len(sub.args) >= 3:
            root = _root_name(sub.args[1])
            if root not in param_names:
                root = None
        if root is None:
            root = _root_name(sub.args[0])
        if root in param_names:
            _merge_vf_role(result, root, "R")
    elif op.startswith("store") and len(sub.args) >= 1:
        root = _root_name(sub.args[0])
        if root in param_names:
            _merge_vf_role(result, root, "W")


def _scan_vf_roles(vf_func_def: ast.FunctionDef, vf_func_defs: dict | None = None) -> dict[str, str]:
    """Scan a VF helper body for param R/W roles.

    Works for both ``@pl.vector_function`` decorated functions (scans
    ``vf.load_*`` / ``vf.store_*`` calls directly) and plain wrapper functions
    that delegate to a known VF function (propagates roles through the call's
    positional argument mapping, recursively for multi-level chains).

    - vf.load_*(reg, ptr, ...) -> ptr param is "R"
    - vf.store_*(ptr, reg, ...) -> ptr param is "W"
    - both -> "RW"
    """
    param_names = {a.arg for a in vf_func_def.args.args if a.arg != "self"}
    result: dict[str, str] = {}

    for sub in ast.walk(vf_func_def):
        if isinstance(sub, ast.Call):
            _record_vf_call_role(sub, param_names, result)
            # Propagate roles through calls to known VF functions (recursive)
            if vf_func_defs and isinstance(sub.func, ast.Name):
                callee_def = vf_func_defs.get(sub.func.id)
                if callee_def is not None and callee_def is not vf_func_def:
                    callee_roles = _scan_vf_roles(callee_def, vf_func_defs)
                    callee_params = [a.arg for a in callee_def.args.args if a.arg != "self"]
                    for argpos, arg in enumerate(sub.args):
                        if argpos >= len(callee_params):
                            break
                        callee_param = callee_params[argpos]
                        role = callee_roles.get(callee_param)
                        if role is None:
                            continue
                        root = _root_name(arg)
                        if root in param_names:
                            _merge_vf_role(result, root, role)
    return result
