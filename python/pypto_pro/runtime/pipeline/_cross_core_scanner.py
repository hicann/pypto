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

from pypto.pypto_impl.ir import MemorySpace
from pypto_pro.language.parser._op_pipeline import (
    _BLOCK_OP_TILE_ROLES,
    _VF_OP_TILE_ROLES,
    get_move_pipe,
    get_op_pipe,
    get_store_pipe,
    op_accesses_buffer,
)

from ._astutil import call_name, slot_accessor

# Highest usable cross-core event id. A hardware limit: ids run 0..15, and an
# out-of-range id only fails once the kernel runs on device. Shared with _sync_graph,
# which allocates from the same pool for address-reuse edges, so it carries no
# module-private underscore.
MAX_EVENT_ID = 15

# MemorySpace attribute name (as written in pl.MemorySpace.<X>) -> enum value
_MEMORY_NAMES = {
    "Vec": MemorySpace.Vec,
    "Mat": MemorySpace.Mat,
    "Left": MemorySpace.Left,
    "Right": MemorySpace.Right,
    "Acc": MemorySpace.Acc,
    "ScaleLeft": MemorySpace.ScaleLeft,
    "ScaleRight": MemorySpace.ScaleRight,
}


def _tile_type_memory(type_node: ast.expr | None) -> MemorySpace | None:
    """The MemorySpace a ``pl.TileType(..., target_memory=pl.MemorySpace.X)`` call declares.

    None when the node is not such a call, or its target_memory is not written as a literal
    ``pl.MemorySpace.<X>``.
    """
    if not isinstance(type_node, ast.Call):
        return None
    for kw in type_node.keywords:
        if kw.arg == "target_memory" and isinstance(kw.value, ast.Attribute):
            if kw.value.attr in _MEMORY_NAMES:
                return _MEMORY_NAMES[kw.value.attr]
    return None


def _is_make_tile_group(call: ast.Call) -> bool:
    """True if call is pl.make_tile_group(...) (or bare make_tile_group(...))."""
    return _get_ctor_name(call) == "make_tile_group"


@dataclass
class CrossCoreBuffer:
    """A cross-core NBuffer's configuration (from kernel-body declaration)."""

    fwd_ids_node: ast.expr | None  # AST node for cross_core_forward_id value (e.g. a Name)
    bwd_ids_node: ast.expr | None  # AST node for cross_core_backward_id value
    # How many ids each direction declares. NOT the buffer's slot count: a buffer may
    # share one id across all its slots (see _validate._check_event_id_counts). This is the
    # modulus of the id pick, `ids[task % id_count]`, while the slots turn over on
    # `task % slot_count`.
    fwd_id_count: int
    bwd_id_count: int


@dataclass
class CrossCoreSyncContext:
    """Buffer declarations and memory layout the sync graph is built from.

    Populated by _scan_cross_core() during analysis.
    """

    # Buffer declarations: name -> CrossCoreBuffer (only cross-core buffers with fwd/bwd ids)
    buffers: dict = field(default_factory=dict)
    # Lifted literal fwd_ids/bwd_ids: [(var_name, ast_literal)] to declare as variables
    lifted_ids: list = field(default_factory=list)
    # Per-buffer slot address ranges: name -> (memory, [(start, end), ...] per slot).
    # Covers ALL buffers (cross-core + local) for address-overlap detection.
    addr_ranges: dict = field(default_factory=dict)
    # buffer name -> tuple of mutex ids. Used to check that co-located buffers hold the
    # same locks (a mutex locks the address, not the variable).
    mutex_ids: dict = field(default_factory=dict)
    # Address-overlapping buffer pairs: [(buf_a, buf_b), ...] (same memory, ranges intersect)
    addr_overlaps: list = field(default_factory=list)
    # Address-reuse sync is derived from the graph on demand, not stored here — see
    # _sync_graph.allocate_reuse_ids.


@dataclass(frozen=True)
class AccessScanTables:
    """The kernel-level tables the per-op ACCESS scan resolves names against.

    Distinct from what the DECLARATION scans produce (addr_ranges, mutex_ids,
    addr_overlaps — those go into CrossCoreSyncContext and model the sync graph). Every
    field here answers one question instead: *a name appears inside a stage body — which
    declared buffer is it, and on which pipe is this op touching it?*

    All seven are resolved once for the whole kernel and only read afterwards, so they
    travel as one value rather than as seven parameters threaded through five call levels.
    That is not only tidier: they used to be positional, untyped and partly defaulted to
    ``None``, so a swapped pair silently mis-attributed accesses and a forgotten argument
    silently scanned nothing — and a missing access is a missing sync.

    What deliberately does NOT live here is the per-stage ``stage_slot_to_buffer``. It has
    a different lifetime (rebuilt per stage) and a different role: it is not a table the
    scan consults but the RESULT of consulting these — see
    _build_slot_to_buffer_from_bindings.
    """

    # Cross-core buffer declarations: name -> CrossCoreBuffer. Membership decides whether
    # an access is worth recording at all.
    cross_buffers: dict
    # name -> MemorySpace for ALL buffers, cross-core or local. A move's pipe follows the
    # memory spaces of both its tiles, so the local side has to resolve too.
    all_buffer_memory: dict
    # Every declared tile group's name, so a call-site argument can be recognised as a group.
    group_names: set
    # Slot variables taken in the KERNEL body (`cur_k = k_db.next()`) -> buffer. Read only
    # by build_binding_map, to resolve a call-site actual argument.
    kernel_slot_to_buffer: dict
    # tuple variable -> {field: source variable}, the one hop that rejoins a tile to its
    # group when the kernel bundles groups with pl.make_tuple.
    tuple_fields: dict
    # ``@pl.vector_function`` name -> FunctionDef. One atomic op each: the scan does not
    # walk into them, it reads their parameter roles (see _scan_vf_roles).
    vf_func_defs: dict
    # Every OTHER plain function a stage may call -> FunctionDef. These are walked through
    # as part of the calling stage (see _scan_function).
    helper_func_defs: dict
    # Buffers sharing a physical region with another buffer. They are recorded even when
    # not cross-core themselves, because a write over shared memory still has to be ordered.
    region_members: set


@dataclass
class TileGroupDecl:
    """One ``x = pl.make_tile_group(...)`` statement, as written.

    Purely syntactic: the keyword nodes are handed over unevaluated, and nothing here is
    validated. Every scan below reads the fields it needs from this and decides for itself
    what to do when a value will not resolve — some skip the buffer, some raise. Keeping
    those decisions in the scans (and their original call order) is what makes the single
    pass a refactor rather than a change in which error a user sees first.

    One field per keyword, rather than a dict of them: the six are the whole of
    make_tile_group's signature and every scan asks for them by literal name, so a dict
    only made the set look open-ended while turning a misspelled key into a silent None.

    ``_node`` marks the fields that are unevaluated AST — deliberately, because
    ``decl.mutex_ids_node`` and ``info.sync.mutex_ids`` are NOT the same thing (one is what
    was written, the other the resolved ints). ``memory`` carries no suffix because it is
    the one field already interpreted.

    Two things are resolved here rather than left to each scan, since each used to be
    re-derived per reader:

    ``type_node`` is the ``pl.TileType(...)`` call itself even when the source wrote a
    variable (``tt = pl.TileType(...)`` … ``type=tt``) — the far more common spelling, and
    the one the API docs use. Every reader wants the tile type, none wants the name it was
    reached through, so the hop is taken once here.

    ``memory`` is read off that type node by pattern-matching ``target_memory=
    pl.MemorySpace.<X>``. Not evaluation: it needs no closure and cannot fail for a reason
    a scan would want to report differently.
    """

    names: list[str]  # every variable this statement binds (targets that are plain Names)
    type_node: ast.expr | None  # the pl.TileType(...) call, resolved through a name if needed
    addrs_node: ast.expr | None
    mutex_ids_node: ast.expr | None
    depth_node: ast.expr | None
    fwd_ids_node: ast.expr | None
    bwd_ids_node: ast.expr | None
    memory: MemorySpace | None  # target_memory, or None when it does not resolve


def scan_tile_group_decls(kernel_func_def: ast.FunctionDef) -> list[TileGroupDecl]:
    """Every ``pl.make_tile_group(...)`` declaration in the kernel body, in AST order.

    The one pass over the kernel body that all buffer-declaration scans share. They used to
    walk it once each, repeating the same "is this an Assign of a make_tile_group call, and
    what does it bind" prologue and re-evaluating the same keywords several times over.
    """
    tile_types = _scan_tile_type_decls(kernel_func_def)
    decls: list[TileGroupDecl] = []
    for node in ast.walk(kernel_func_def):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        if not _is_make_tile_group(call):
            continue
        written = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        # `type=tt` -> the pl.TileType(...) call `tt` was bound to, so every scan below
        # sees the tile type itself rather than the name it arrived under.
        type_node = written.get("type")
        if isinstance(type_node, ast.Name) and type_node.id in tile_types:
            type_node = tile_types[type_node.id]
        decls.append(
            TileGroupDecl(
                names=[t.id for t in node.targets if isinstance(t, ast.Name)],
                type_node=type_node,
                addrs_node=written.get("addrs"),
                mutex_ids_node=written.get("mutex_ids"),
                depth_node=written.get("depth"),
                fwd_ids_node=written.get("fwd_ids"),
                bwd_ids_node=written.get("bwd_ids"),
                # target_memory is a keyword of the TILE TYPE, not of make_tile_group.
                memory=_tile_type_memory(type_node),
            )
        )
    return decls


def _scan_tile_type_decls(kernel_func_def: ast.FunctionDef) -> dict[str, ast.Call]:
    """``{name: the pl.TileType(...) call it was bound to}`` in the kernel body.

    One hop, which is all the spelling needs: a tile type is written once and handed to
    make_tile_group by name. Chains of aliases are not followed — nobody writes them, and
    an unresolved name simply leaves ``kwargs["type"]`` as it was.
    """
    result: dict[str, ast.Call] = {}
    for node in ast.walk(kernel_func_def):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        if _get_ctor_name(node.value) != "TileType":
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = node.value
    return result


def scan_all_tile_group_names(decls: list[TileGroupDecl]) -> set[str]:
    """Every variable bound directly to a ``pl.make_tile_group(...)`` result.

    Deliberately independent of mutex_ids and cross-core ids: a group whose ids are not
    statically resolvable is still a group, and callers that only need to recognise slot
    selection (``g.next()``, ``g[i]``) must not miss it. Aliases are not followed, so
    ``alias = g`` leaves ``alias`` unrecognised.
    """
    return {name for decl in decls for name in decl.names}


def scan_all_buffer_memory(decls: list[TileGroupDecl]) -> dict[str, MemorySpace]:
    """Return every make_tile_group declaration's memory by buffer variable name."""
    return {name: decl.memory for decl in decls if decl.memory is not None for name in decl.names}


def _eval_const(node: ast.expr, closure_vars: dict):
    """Evaluate a constant expression node using closure_vars as namespace.
    Returns the value, or None if it cannot be evaluated (e.g. references a
    runtime variable)."""
    try:
        code = compile(ast.Expression(body=node), "<addr>", "eval")
        return eval(code, {"__builtins__": {}}, dict(closure_vars))
    except Exception:
        return None


def _tile_type_slot_size(type_call: ast.Call, closure_vars: dict) -> int | None:
    """Compute per-slot byte size from a pl.TileType(shape=..., dtype=...) call.
    Returns None if shape/dtype cannot be resolved.

    slot_size = product(shape dims) * ceil(dtype_bits / 8)  (mirrors _buffer_parser).
    """
    shape_node = dtype_node = None
    for kw in type_call.keywords:
        if kw.arg == "shape":
            shape_node = kw.value
        elif kw.arg == "dtype":
            dtype_node = kw.value
    if shape_node is None or dtype_node is None:
        return None
    shape = _eval_const(shape_node, closure_vars)
    dtype = _eval_const(dtype_node, closure_vars)
    if not isinstance(shape, (list, tuple)) or dtype is None:
        return None
    elems = 1
    for d in shape:
        elems *= int(d)
    try:
        bits = int(dtype.get_bit())
    except Exception:
        return None
    return elems * ((bits + 7) // 8)


def scan_buffer_addr_ranges(decls: list[TileGroupDecl], closure_vars: dict) -> dict:
    """Compute each buffer's per-slot address ranges from its declaration.

    Returns: name -> (memory, [(start, end), ...] per slot).
    Buffers whose addrs/shape/dtype cannot be statically resolved are skipped
    (they simply won't participate in overlap detection).

    addrs handling (mirrors _buffer_parser):
      - single value  -> contiguous slots: base + i*slot_size
      - list/tuple     -> one explicit start address per slot
    slot count comes from depth when present, otherwise len(mutex_ids).
    """
    result: dict = {}
    for decl in decls:
        memory = decl.memory
        if memory is None:
            continue

        type_node = decl.type_node
        addrs_node = decl.addrs_node
        mutex_node = decl.mutex_ids_node
        depth_node = decl.depth_node
        if not (isinstance(type_node, ast.Call) and addrs_node is not None):
            continue

        slot_size = _tile_type_slot_size(type_node, closure_vars)
        mutex_ids = _eval_const(mutex_node, closure_vars) if mutex_node is not None else None
        depth = _eval_const(depth_node, closure_vars) if depth_node is not None else None
        addrs = _eval_const(addrs_node, closure_vars)
        if slot_size is None or addrs is None:
            continue
        if mutex_ids is not None and not isinstance(mutex_ids, (list, tuple)):
            continue
        if depth_node is not None:
            if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
                continue
            num = depth
        elif isinstance(mutex_ids, (list, tuple)) and mutex_ids:
            num = len(mutex_ids)
        else:
            continue
        if isinstance(mutex_ids, (list, tuple)) and mutex_ids and len(mutex_ids) != num:
            continue  # malformed; parser will raise the real error

        if isinstance(addrs, (list, tuple)):
            if len(addrs) != num:
                continue  # malformed; skip (parser will raise the real error)
            starts = list(addrs)
        else:
            starts = [addrs + i * slot_size for i in range(num)]
        ranges = [(int(s), int(s) + slot_size) for s in starts]

        for name in decl.names:
            result[name] = (memory, ranges)
    return result


def scan_buffer_mutex_ids(decls: list[TileGroupDecl], closure_vars: dict) -> dict:
    """Each buffer's mutex_ids. Returns name -> tuple of ids.

    Needed to check that buffers sharing an address also share their locks: a mutex locks
    the address, so different ids over one region provide no mutual exclusion at all.
    Buffers whose ids are not statically resolvable are omitted rather than guessed.
    """
    result: dict = {}
    for decl in decls:
        mutex_node = decl.mutex_ids_node
        if mutex_node is None:
            continue
        mutex_ids = _eval_const(mutex_node, closure_vars)
        if not isinstance(mutex_ids, (list, tuple)):
            continue
        for name in decl.names:
            result[name] = tuple(mutex_ids)
    return result


def detect_addr_overlaps(addr_ranges: dict, cross_core_names: set) -> list:
    """Detect address-overlapping buffer pairs relevant to cross-core sync.

    Two buffers overlap if they share the same MemorySpace and any of their slot
    ranges intersect. Only pairs where **at least one side is a cross-core buffer**
    are reported — overlaps between two local buffers (e.g. multiple views of the
    same UB region like p_f16_db / p_f16_main_db) are the user's / auto_mutex's
    concern, not the pipeline cross-core sync's. Returns (buf_a, buf_b) pairs
    (names sorted).

    Constraint: only pairwise overlap is supported. If any buffer overlaps with
    more than one other buffer (counting only cross-core-relevant overlaps),
    raises ValueError.
    """
    names = sorted(addr_ranges.keys())
    overlaps = []
    # buffer -> set of buffers it overlaps with (for the 3+ check)
    overlap_partners: dict = {}

    def _ranges_intersect(ra, rb) -> bool:
        for sa, ea in ra:
            for sb, eb in rb:
                if sa < eb and sb < ea:  # half-open interval intersection
                    return True
        return False

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # Skip if neither is a cross-core buffer (local-local overlap is not
            # a pipeline sync concern).
            if a not in cross_core_names and b not in cross_core_names:
                continue
            mem_a, ranges_a = addr_ranges[a]
            mem_b, ranges_b = addr_ranges[b]
            if mem_a != mem_b:
                continue
            if _ranges_intersect(ranges_a, ranges_b):
                # Slot count must be the same for overlapping buffers (precise
                # per-slot overlap tracking is not yet supported).
                if len(ranges_a) != len(ranges_b):
                    raise ValueError(
                        f"pipeline: address-overlapping buffers '{a}' and '{b}' have "
                        f"different slot counts ({len(ranges_a)} vs {len(ranges_b)}). "
                        f"Overlapping buffers must have the same number of slots."
                    )
                overlaps.append((a, b))
                overlap_partners.setdefault(a, set()).add(b)
                overlap_partners.setdefault(b, set()).add(a)

    for buf, partners in overlap_partners.items():
        if len(partners) > 1:
            raise ValueError(
                f"pipeline: buffer '{buf}' has overlapping addresses with multiple "
                f"buffers {sorted(partners)}. Only pairwise address overlap is "
                f"supported (at most 2 buffers may share a region)."
            )
    return overlaps


def scan_cross_core_buffers(
    decls: list[TileGroupDecl], closure_vars: dict
) -> tuple[dict[str, CrossCoreBuffer], list]:
    """Pick out the cross-core tile-group declarations (those carrying fwd/bwd ids).

    The only declaration scan that raises: a buffer asking for cross-core sync must have a
    resolvable memory space and valid id tuples, or no sync can be generated for it. The
    scans that merely collect layout information skip what they cannot resolve instead.

    Returns:
        (dict mapping buffer variable name -> CrossCoreBuffer,
         lifted_ids: list of (var_name, ast_literal) for literal fwd/bwd ids
         that need to be declared as variables before the pipeline loop)
    """
    result: dict[str, CrossCoreBuffer] = {}
    lifted_ids: list[tuple[str, ast.expr]] = []

    for decl in decls:
        fwd_node = decl.fwd_ids_node
        bwd_node = decl.bwd_ids_node
        if fwd_node is None and bwd_node is None:
            continue

        bufname = decl.names[0] if decl.names else "<tile_group>"

        _validate_buffer_memory(decl, bufname)
        fwd_count, bwd_count = _validate_ids(fwd_node, bwd_node, bufname, closure_vars)
        fwd_node, bwd_node = _lift_literal_ids(fwd_node, bwd_node, bufname, lifted_ids)

        for name in decl.names:
            result[name] = CrossCoreBuffer(
                fwd_ids_node=fwd_node,
                bwd_ids_node=bwd_node,
                fwd_id_count=fwd_count,
                bwd_id_count=bwd_count,
            )

    return result, lifted_ids


def _validate_buffer_memory(decl: TileGroupDecl, bufname: str) -> None:
    """L8: Validate that a make_tile_group call declares a resolvable memory space."""
    if decl.memory is None:
        raise ValueError(
            f"pipeline: cross-core buffer '{bufname}' has no resolvable memory space. "
            f"Its make_tile_group(type=pl.TileType(..., target_memory=pl.MemorySpace.X)) "
            f"must set target_memory to a literal pl.MemorySpace.<X>."
        )


def _validate_ids(fwd_node, bwd_node, bufname: str, closure_vars: dict) -> tuple[int, int]:
    """L11: how many ids each direction declares, once they are known to be usable.

    Checks the spelling and the values. How MANY there must be depends on how many slots
    the buffer rotates through, which is not known here — see
    _validate._check_event_id_counts, which asks the graph.
    """
    counts = []
    for label, node in (("fwd_ids", fwd_node), ("bwd_ids", bwd_node)):
        if node is None:
            counts.append(0)
            continue
        ids = _resolve_event_ids(node, bufname, label, closure_vars)
        if not ids:
            raise ValueError(
                f"pipeline: cross-core buffer '{bufname}' {label} is empty. Drop the "
                f"keyword if this buffer needs no sync in that direction."
            )
        # Cross-core event ids are a hardware resource limited to 0..15; an out-of-range id
        # only fails once the kernel runs on device, so reject it where the source is known.
        bad = [v for v in ids if not 0 <= v <= MAX_EVENT_ID]
        if bad:
            raise ValueError(
                f"pipeline: cross-core buffer '{bufname}' {label} contains out-of-range "
                f"event id(s) {bad}; cross-core event ids must be in 0..{MAX_EVENT_ID}."
            )
        counts.append(len(ids))
    return counts[0], counts[1]


def _lift_literal_ids(fwd_node, bwd_node, bufname: str, lifted_ids: list) -> tuple[ast.expr, ast.expr]:
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


def _is_subview(node: ast.expr) -> bool:
    """Whether ``node`` is ``tile[...]`` with a slice among its indices — a sub-view.

    A sub-view is a window onto the SAME memory as the tile it came from, which gives it
    two properties that have to stay consistent: it resolves to that tile's buffer
    (build_binding_map), and it is not itself an access, because no data moves until
    something uses the view (_handle_tile_subscript). Writing the test out twice would let
    those two drift apart, so both ask here.
    """
    if not isinstance(node, ast.Subscript):
        return False
    index = node.slice
    if isinstance(index, ast.Slice):
        return True
    return isinstance(index, ast.Tuple) and any(isinstance(e, ast.Slice) for e in index.elts)


def _slot_accessor_group_name(value: ast.expr) -> str | None:
    """Group name for a slot accessor expression, or None if it is not one.

    Which spellings count is _astutil.slot_accessor's business; nothing here needs the
    accessor kind, only which group the tile came from.
    """
    accessor = slot_accessor(value)
    return accessor[0] if accessor else None


def _get_slot_accessor_assignment(node: ast.AST, param_names: set[str]) -> tuple[str, str] | None:
    """Return slot variable and source buffer for ``slot = group.next()`` / ``slot = group[i]``."""
    if not isinstance(node, ast.Assign):
        return None
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None

    group_name = _slot_accessor_group_name(node.value)
    if group_name is None or group_name not in param_names:
        return None
    return node.targets[0].id, group_name


def scan_kernel_slot_to_buffer(func_def: ast.FunctionDef, buffer_names: set[str]) -> dict[str, str]:
    """Scan the kernel body for `slot = buf.next()` (or .current()/.previous()),
    mapping slot variable -> buffer name.

    This handles the case where the user takes a slot OUTSIDE the stage function
    (in the pipeline loop) and passes the slot tile as a stage argument.

    Covers EVERY declared buffer, not just the cross-core ones: a local buffer's slot has to
    be traceable too, because an op's pipe is decided by the memory spaces of all its tiles
    — the local side of a `pl.move` included.
    """
    result: dict[str, str] = {}
    for node in ast.walk(func_def):
        acc = _get_slot_accessor_assignment(node, buffer_names)
        if acc is not None:
            slot_name, buffer_name = acc
            result[slot_name] = buffer_name
    return result


def build_binding_map(
    func_def: ast.FunctionDef,
    call_args: list | None,
    tables: AccessScanTables,
) -> dict[str, tuple[str, bool]]:
    """Build the name -> (buffer, is_group) binding map for a stage function body.

    Resolves all names that reference a cross-core buffer (directly or indirectly)
    within this function scope. Handles:
      (a) Formal params bound via call-site positional args (the actual is a declared tile
          group — cross-core or local — or a slot taken from one in the kernel body).
      (b) .next()/.current()/.previous() calls in the body: slot = group.next().
      (c) Pure Name-to-Name alias assignments: tmp = some_known_name.
      (d) Member reads off an aggregate param: grp = agg.field, where the call site passed a
          `pl.make_tuple(field=...)` — rejoins the chain the aggregate broke.

    Steps (b) through (d) iterate until stable (supports alias chains).

    Args:
        func_def: the stage function AST.
        call_args: list of AST args from the call site. If None, returns empty
            (no name-collision guessing).
        tables: the kernel-level lookup tables (see AccessScanTables). Of these, only
            group_names, kernel_slot_to_buffer and tuple_fields are consulted here —
            this is the one place kernel_slot_to_buffer is read at all.

    Returns: {name: (buffer_declared_name, is_group)}
    """
    param_names = [a.arg for a in func_def.args.args if a.arg != "self"]
    result: dict[str, tuple[str, bool]] = {}
    # Aggregate params: {param name: {field: source variable}}. Kept apart from `result`
    # because an aggregate is not a buffer — it is only a route to one.
    aggregates: dict[str, dict[str, str]] = {}

    # (a) Param bindings from call site (positional). Without call args we cannot
    # resolve bindings — name-collision guessing is intentionally NOT done.
    if call_args is None:
        return result
    for pos, arg in enumerate(call_args):
        if pos >= len(param_names) or not isinstance(arg, ast.Name):
            continue
        actual = arg.id
        # A local group is bound under its real name just like a cross-core one. Only
        # cross-core buffers get sync of their own, but a local buffer still has to be
        # traceable: an op's pipe follows the memory spaces of ALL its tiles, so the local
        # side of a `pl.move` needs a name that resolves. Leaving it out here is what made
        # the scan fall back to the FORMAL parameter name, which only ever matched by the
        # convention that call sites name their arguments after the parameters.
        if actual in tables.group_names:
            result[param_names[pos]] = (actual, True)
        elif actual in tables.kernel_slot_to_buffer:
            result[param_names[pos]] = (tables.kernel_slot_to_buffer[actual], False)
        elif actual in tables.tuple_fields:
            aggregates[param_names[pos]] = tables.tuple_fields[actual]

    # (b)(c)(d) Propagate slot accessors (.next() / group[i]), aggregate member reads and
    # alias assignments until stable.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(func_def):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target = node.targets[0].id
            if target in result:
                continue  # already resolved

            # (d) grp = agg.field -> the variable the call site put in that field. Recorded
            # even when that variable is a LOCAL tile group: an op's pipe is decided by the
            # memory spaces of ALL its tiles, so the local side of a `move` has to resolve
            # too. Not being a cross-core buffer keeps it out of the access records.
            if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
                fields = aggregates.get(node.value.value.id)
                if fields is not None:
                    source = fields.get(node.value.attr)
                    if source is not None:
                        result[target] = (source, True)
                        changed = True
                    continue

            if isinstance(node.value, (ast.Call, ast.Subscript)):
                source_name = _slot_accessor_group_name(node.value)
                if source_name is not None and source_name in result:
                    src_buf, src_is_group = result[source_name]
                    # A slot taken from a group, or a sub-view carved out of a tile: either
                    # way the result names memory belonging to src_buf. Only the group case
                    # used to be handled, so `sub = slot[i:, :]` resolved to nothing and
                    # every op on `sub` was dropped from the scan without a word — and with
                    # it any check that works by resolving an argument to a buffer,
                    # _reject_untabled_op included.
                    if src_is_group or _is_subview(node.value):
                        result[target] = (src_buf, False)
                        changed = True
                continue

            if isinstance(node.value, ast.Name) and node.value.id in result:
                result[target] = result[node.value.id]
                changed = True

    return result


def _build_slot_to_buffer_from_bindings(
    func_def: ast.FunctionDef, bindings: dict[str, tuple[str, bool]]
) -> dict[str, str]:
    """This stage's ``name -> buffer`` map, covering ALL params (for pipe resolution).

    Not a second scan of what ``tables.kernel_slot_to_buffer`` already holds, but its
    product: the kernel-body slot a call site passed in arrives here through
    build_binding_map, RE-KEYED from the kernel's variable name to the formal parameter
    name (``cur_k = k_db.next(); compute_qk(cur_k)`` against ``def compute_qk(k_tile)``
    yields ``k_tile -> k_db``, never ``cur_k``). That is the key the stage body's ops are
    written against. On top of it come the slots taken inside this body.

    - Names with is_group=False → directly map to their buffer (they are slots).
    - Names from `slot = param.next()` for ANY param → map to param's resolved name
      (for non-cross-core params, resolved name = param name itself for pipe calc).
    """
    stage_slot_to_buffer: dict[str, str] = {}
    # All resolved slot names (is_group=False) → buffer
    for name, (buf, is_group) in bindings.items():
        if not is_group:
            stage_slot_to_buffer[name] = buf
    # For pipe resolution: also scan .next() on non-cross-core params (local buffers).
    # Their resolved name is just the param name (standard pipe-table lookup).
    all_param_names = {a.arg for a in func_def.args.args if a.arg != "self"}
    for node in ast.walk(func_def):
        acc = _get_slot_accessor_assignment(node, all_param_names)
        if acc is not None:
            slot_name, param_name = acc
            if slot_name not in stage_slot_to_buffer:
                # Resolve: if param is in bindings use its buffer, else param name itself
                if param_name in bindings:
                    stage_slot_to_buffer[slot_name] = bindings[param_name][0]
                else:
                    stage_slot_to_buffer[slot_name] = param_name
    return stage_slot_to_buffer


def _scan_function(
    func_def: ast.FunctionDef, bindings: dict, tables: AccessScanTables, out: list, seen: set
) -> None:
    """Record every buffer access this function makes, in source order, into ``out``.

    Walks the calls in the body and dispatches on what each one is:

        ``pl.<op>(...)``            a block op            -> _handle_block_op
        a ``@pl.vector_function``   one atomic op         -> _handle_vf_call
        any other plain function    part of THIS function -> recurse into it
        a stage                     not reachable here    -> refused by _check_no_nested_stage

    The third case is the reason this takes a whole function rather than a statement list.
    A helper a stage calls is not a separate unit of work — it is the stage, factored out —
    so its accesses belong to the stage's list and are recorded straight into the same
    ``out``. (A nested STAGE would be the opposite: its own node, with its own sync around
    it. That is the mix-stage case and is not supported yet.) Before this, the walk stopped
    at the stage's own body, so a ``pl.move`` inside a helper was scanned by nobody while
    the parser inlined and ran it — a missing sync with nothing to announce it.

    ``bindings`` is this function's ``name -> (buffer, is_group)`` map; each recursion
    re-derives the callee's from the caller's by argument position. ``seen`` holds the names
    already on the current chain, so a cycle terminates instead of recursing forever.

    Control flow is deliberately ignored: an op inside a branch is recorded exactly like
    an unconditional one. The graph then syncs every pipe a buffer is touched on, which
    over-approximates when a path skips some of them but is never short of a sync. Working
    out which pipes a given path really uses would need condition reasoning, and getting
    that wrong drops syncs rather than adding them.
    """
    slot_to_buffer = _build_slot_to_buffer_from_bindings(func_def, bindings)
    for stmt in func_def.body:
        for node, is_aug_target in _iter_accessors_in_order(stmt):
            if isinstance(node, ast.Subscript):
                _handle_tile_subscript(node, is_aug_target, slot_to_buffer, tables, out)
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pl"
            ):
                _handle_block_op(node, slot_to_buffer, tables, out)
                continue
            name = call_name(node)
            if not name:
                continue
            if name in tables.vf_func_defs:
                _handle_vf_call(node, slot_to_buffer, tables, out)
                continue
            callee = tables.helper_func_defs.get(name)
            # Not a known function (a stage, or a name this kernel's scope cannot resolve),
            # or already on this chain.
            if callee is None or name in seen:
                continue
            _reject_tile_group_decl(callee, name)
            _scan_function(
                callee,
                _bind_callee_params(callee, node.args, bindings, slot_to_buffer),
                tables,
                out,
                seen | {name},
            )


def _bind_callee_params(callee: ast.FunctionDef, call_args: list, bindings: dict, slot_to_buffer: dict) -> dict:
    """The callee's ``name -> (buffer, is_group)`` map, from the caller's by argument position.

    Same shape as build_binding_map, different source: that one resolves a call site in the
    KERNEL body against the declarations, this one resolves a call site inside a function
    against what that function already knows. ``slot_to_buffer`` is consulted as well as
    ``bindings`` because it holds the names the accessor scan added on top of them.
    """
    params = [a.arg for a in callee.args.args if a.arg != "self"]
    result: dict[str, tuple[str, bool]] = {}
    for pos, arg in enumerate(call_args):
        if pos >= len(params) or not isinstance(arg, ast.Name):
            continue
        if arg.id in bindings:
            result[params[pos]] = bindings[arg.id]
        elif arg.id in slot_to_buffer:
            result[params[pos]] = (slot_to_buffer[arg.id], False)
    return result


def _reject_tile_group_decl(func_def: ast.FunctionDef, name: str) -> None:
    """A tile group must be declared in the kernel body, not in a function it calls.

    Every table the transform keys on — group names, mutex ids, address ranges, the
    cross-core buffers themselves — is indexed by the variable name the declaration binds,
    and scan_tile_group_decls fills them from the kernel body alone. A group declared in a
    helper is therefore in none of them: accesses to it resolve to nothing and are dropped,
    silently, which is the failure this refuses.

    Supporting it would mean more than scanning further. A factory helper returns its groups
    to the caller under different names, so their identity would have to be traced through
    the return; and one called twice would give a single declaration two identities, which
    a name-keyed model cannot express at all.
    """
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and _is_make_tile_group(node):
            raise ValueError(
                f"pipeline: '{name}' declares a tile group with pl.make_tile_group(), but it "
                f"is a function called from a stage. Tile groups must be declared in the "
                f"kernel body — the transform identifies every buffer by the name its "
                f"declaration binds there, so one declared here is invisible to the sync it "
                f"needs.\nMove the declaration into the kernel and pass the group in."
            )


def _block_op_roles(node: ast.Call, op_name: str) -> list | None:
    """The argument roles for this call, or None when the table does not cover the op.

    An op whose ``mode=`` changes what it reads and writes has one table entry per mode,
    named ``<op>_<mode lowercased>`` — ``fillpad`` carries ``fillpad_inplace`` (its first
    argument becomes RW) and ``fillpad_expand``. Looking that name up, rather than testing
    for one op and one mode as this used to, is what makes the convention usable: a new
    mode-dependent op needs a table entry and nothing here. It also reaches
    ``fillpad_expand``, which the single hard-coded INPLACE test never could.
    """
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Attribute):
            variant = _BLOCK_OP_TILE_ROLES.get(f"{op_name}_{kw.value.attr.lower()}")
            if variant is not None:
                return variant
    return _BLOCK_OP_TILE_ROLES.get(op_name)


def _reject_untabled_op(node: ast.Call, op_name: str, stage_slot_to_buffer: dict, tables) -> None:
    """Refuse an op that touches a tracked buffer but has no roles in the table.

    The table is written by hand and the framework keeps gaining ops, so it runs behind —
    at the time of writing it covers 90 while more than twenty ops that take tiles are
    absent. Reaching one of those used to mean returning quietly: the access went
    unrecorded, no wait/set was generated for it, and nothing said so.

    Refusing only when a resolved argument is a buffer the graph tracks keeps this silent
    for the ops it should be silent for — ``pl.range``, a tile type, anything on purely
    local tiles — and loud exactly where the omission would have cost a sync.

    Same reasoning as _block_op_pipe: there is no safe default. Guessing "read" drops the
    producer's set, guessing "write" drops the consumer's wait; either way the handover is
    unenforced and the kernel is intermittently wrong rather than broken.
    """
    touched = []
    for arg in node.args:
        buf = _tile_arg_buffer(arg, stage_slot_to_buffer)
        if buf is not None and (buf in tables.cross_buffers or buf in tables.region_members):
            touched.append(buf)
    if not touched:
        return
    likely = ["W"] + ["R"] * (len(node.args) - 1)
    raise ValueError(
        f"pipeline: `pl.{op_name}` at line {node.lineno} touches buffer(s) "
        f"{sorted(set(touched))} that cross-core sync tracks, but _BLOCK_OP_TILE_ROLES has "
        f"no entry for it, so the scan cannot tell which arguments it reads and which it "
        f"writes — and which way a wait/set goes depends on exactly that.\n"
        f"Add an entry in pypto_pro/language/parser/_op_pipeline.py. With the "
        f"{len(node.args)} positional argument(s) here the roles are probably {likely}, but "
        f"check the two things a signature cannot show: an argument the op accumulates into "
        f"is 'RW', and a position that is not a tile is None."
    )


def _handle_block_op(node: ast.Call, stage_slot_to_buffer: dict, tables: AccessScanTables, out: list) -> None:
    """Record a single pl.<op>(...) call."""
    op_name = node.func.attr
    # Descriptor-only ops (set_validshape and whatever joins it) rewrite a tile's metadata
    # and never touch its data, so they can neither race nor need a handover. The parser
    # already keeps that set for auto_mutex and says so — "auto_mutex / sync insertion can
    # skip them" — this is the sync-insertion half finally reading it. Asking here, before
    # the role lookup, is also what keeps _reject_untabled_op honest: an op absent from the
    # role table is then genuinely unclassified rather than merely harmless.
    if not op_accesses_buffer(op_name):
        return
    roles = _block_op_roles(node, op_name)
    if roles is None:
        _reject_untabled_op(node, op_name, stage_slot_to_buffer, tables)
        return
    # One record per buffer, not per argument: an op is a single point in time, so a buffer
    # it both reads and writes is one RW access. The pipe is a property of the op (see
    # _block_op_pipe, which is handed the whole call), so merging cannot lose one.
    merged: dict[str, str] = {}
    pipe = None
    for argpos, arg in enumerate(node.args):
        buf = _tile_arg_buffer(arg, stage_slot_to_buffer)
        if buf is None or argpos >= len(roles):
            continue
        role = roles[argpos]
        if role is None:
            continue
        if buf not in tables.cross_buffers and buf not in tables.region_members:
            continue
        if pipe is None:
            pipe = _block_op_pipe(op_name, node, stage_slot_to_buffer, tables.all_buffer_memory)
        _merge_role(merged, buf, role)
    for buf, role in merged.items():
        _record_region_access(out, buf, role, pipe)


def _handle_tile_subscript(
    node: ast.Subscript, is_aug_target: bool, stage_slot_to_buffer: dict,
    tables: AccessScanTables, out: list
) -> None:
    """Record ``tile[i, j]`` / ``tile[i, j] = v`` — one scalar element, on the S pipe.

    A tile is two-dimensional, so an element access is written with two indices. The parser
    requires the index count to equal the container's rank and rejects ``tile[i]`` on a
    rank-2 tile outright, which is what makes skipping every non-tuple index safe here:
    the shapes that get skipped do not compile in the first place.

    A sub-view (a slice among the indices) is not an element access — see _is_subview. It
    resolves to its parent's buffer instead, so ops on it are still scanned.

    A tile-group subscript (``group[i]``) is not an access either — it selects a slot — and
    needs no test of its own: only tiles reach ``stage_slot_to_buffer``, never a group's name.
    """
    if not isinstance(node.value, ast.Name):
        return
    if _is_subview(node) or not isinstance(node.slice, ast.Tuple):
        return
    buf = stage_slot_to_buffer.get(node.value.id)
    if buf is None or (buf not in tables.cross_buffers and buf not in tables.region_members):
        return
    if is_aug_target:
        role, pipe_op = "RW", "setval"
    elif isinstance(node.ctx, ast.Store):
        role, pipe_op = "W", "setval"
    else:
        role, pipe_op = "R", "getval"
    _record_region_access(out, buf, role, _pipe_name(get_op_pipe(pipe_op)))


def _handle_vf_call(node: ast.Call, stage_slot_to_buffer: dict, tables: AccessScanTables, out: list) -> None:
    """Record a single VF helper call, always on pipe V."""
    vf_name = node.func.id
    vf_def = tables.vf_func_defs.get(vf_name)
    if vf_def is None:
        return
    vf_params = [a.arg for a in vf_def.args.args if a.arg != "self"]
    # Which parameters hold a tracked buffer is settled here, where the bindings are, and
    # handed to the scan; it has no way to tell a pointer parameter from a scalar one.
    tile_params: dict[str, str] = {}
    for argpos, arg in enumerate(node.args):
        buf = _tile_arg_buffer(arg, stage_slot_to_buffer)
        if buf is None or argpos >= len(vf_params):
            continue
        if buf in tables.cross_buffers or buf in tables.region_members:
            tile_params[vf_params[argpos]] = buf
    if not tile_params:
        return
    vf_roles = _scan_vf_roles(vf_def, tables.vf_func_defs, tile_params)
    # Merged per buffer for the same reason as a block op: one call is one access, even
    # when the same buffer arrives at two parameters with different roles.
    merged: dict[str, str] = {}
    for param, role in vf_roles.items():
        _merge_role(merged, tile_params[param], role)
    for buf, role in merged.items():
        _record_region_access(out, buf, role, "V")


def _iter_accessors_in_order(node: ast.AST):
    """Yield the ast.Call and ast.Subscript nodes under ``node``, in execution order.

    Subscripts come along because ``t[i]`` and ``t[i] = v`` ARE accesses — the parser
    lowers them to getval/setval — and they are the spelling the docs use, so a scan that
    only looked at calls saw neither.

    An assignment yields its value before its targets, which is the order the machine runs
    them in and the reverse of the field order ast would otherwise give. It matters for
    ``t[i, j] = t[k, l]``: recording the write first would let the read find that write as
    the nearest preceding one and stop there, never reaching the write it depends on.

    Each node comes back paired with a flag saying whether it is the target of an augmented
    assignment. ``t[i, j] += v`` is a read AND a write of that element — the parser rewrites
    it to ``t[i, j] = t[i, j] + v``, materialising a Load of the target — but it does so on
    its own tree, so what is scanned here still carries a lone Store that would otherwise
    look like a plain write.
    """
    if isinstance(node, ast.Assign):
        children = [(node.value, False), *((t, False) for t in node.targets)]
    elif isinstance(node, ast.AugAssign):
        children = [(node.value, False), (node.target, True)]
    else:
        children = [(child, False) for child in ast.iter_child_nodes(node)]
    for child, is_aug_target in children:
        if isinstance(child, (ast.Call, ast.Subscript)):
            yield child, is_aug_target
        yield from _iter_accessors_in_order(child)


def scan_stage_accesses(
    stage_func_def: ast.FunctionDef,
    call_args: list | None,
    tables: AccessScanTables,
) -> list:
    """One stage's buffer accesses, ``[(buffer, role, pipe), ...]`` one entry per op.

    Control flow is ignored (see _scan_function). Returns the list rather than filling one
    handed in: it is the single output, and as an out-parameter it sat in the middle of
    six read-only lookups where nothing marked it as the one thing being written.

    Args:
        stage_func_def: the @stage function AST.
        call_args: the AST args from the call site. None means the bindings cannot be
            resolved, and the scan yields nothing — name-collision guessing is not done.
        tables: the kernel-level lookup tables (see AccessScanTables).
    """
    bindings = build_binding_map(stage_func_def, call_args, tables)
    if not bindings:
        return []

    group_param_names = {p for p, (_b, ig) in bindings.items() if ig}
    _validate_slot_accessors(stage_func_def, group_param_names)

    accesses: list = []
    _scan_function(stage_func_def, bindings, tables, accesses, {stage_func_def.name})
    return accesses


def _validate_slot_accessors(stage_func_def: ast.FunctionDef, group_param_names: set[str]) -> None:
    """L10: cross-core buffer slot accessors must be `slot = param.next()` / `slot = param[i]` form.

    group_param_names: formal params that carry a cross-core buffer GROUP (the ones
    the body calls .next()/.current()/.previous() on, or subscripts)."""
    accessor_rhs_ids = set()
    for node in ast.walk(stage_func_def):
        sa = _get_slot_accessor_assignment(node, group_param_names)
        if sa is not None:
            accessor_rhs_ids.add(id(node.value))
    for node in ast.walk(stage_func_def):
        if not isinstance(node, (ast.Call, ast.Subscript)):
            continue
        group_name = _slot_accessor_group_name(node)
        if group_name is None or group_name not in group_param_names:
            continue
        if id(node) in accessor_rhs_ids:
            continue
        accessor = "[...]" if isinstance(node, ast.Subscript) else f".{node.func.attr}()"
        raise ValueError(
            f"pipeline: cross-core buffer group '{group_name}' slot accessor "
            f"`{accessor}` must be assigned to a simple variable "
            f"(`slot = {group_name}{accessor}`); inline/chained/"
            f"tuple-unpack forms are not supported."
        )


def scan_tuple_fields(kernel_func_def: ast.FunctionDef) -> dict[str, dict[str, str]]:
    """``{tuple variable: {field name: the variable it was built from}}`` for each
    ``x = pl.make_tuple(field=var, ...)`` in the kernel body.

    A stage handed an aggregate reads its members back out as ``agg.field``, which breaks the
    chain from a tile back to its declared tile group. This records the one hop needed to
    rejoin it — see rule (d) in build_binding_map.

    Only keyword members bound to a plain name are recorded: those are the ones a field
    access can be traced through.
    """
    result: dict[str, dict[str, str]] = {}
    for node in ast.walk(kernel_func_def):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        if _get_ctor_name(node.value) != "make_tuple":
            continue
        fields = {kw.arg: kw.value.id for kw in node.value.keywords if kw.arg and isinstance(kw.value, ast.Name)}
        if not fields:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = fields
    return result


def _get_ctor_name(call: ast.Call) -> str | None:
    """Get the constructor name from a call like pl.UBNBuffer(...) -> 'UBNBuffer'."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _const_int(node: ast.expr) -> int | None:
    """The integer a literal element denotes, or None if it is not one.

    A negative literal parses as ``UnaryOp(USub, Constant)`` rather than ``Constant(-n)``,
    so matching only Constant would read ``-1`` as "not an integer" instead of as the
    out-of-range id it is.
    """
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node, sign = node.operand, -1
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return sign * node.value
    return None


def _resolve_event_ids(node: ast.expr, bufname: str, label: str, closure_vars: dict) -> list[int]:
    """The integer event ids behind an ``fwd_ids=`` / ``bwd_ids=`` node.

    Exactly two spellings are supported, and deliberately no more::

        fwd_ids=[0, 1]          # a literal list (or tuple)
        IDS = [0, 1]
        fwd_ids=IDS             # a name bound to one outside the kernel

    Every element must be a compile-time integer. That is stricter than it used to be, and
    the strictness is what makes the ids usable: the list is emitted VERBATIM ahead of the
    loop as ``_pl_fwd_ids_<buffer> = [...]`` and only then subscripted at runtime, so an
    element this scan cannot evaluate is also an element whose meaning in the generated
    kernel is anyone's guess — a name that does not exist there, or a nested list that
    makes ``event_id`` a list. Both used to pass silently.

    Anything else (a slice, a concatenation, a call, a name bound to a non-sequence) is
    refused rather than half-supported: these ids are a handful of numbers the user writes
    once, so the cost of the restriction is a line of source, while accepting an expression
    means the framework has to agree with the parser about how to evaluate it.
    """
    hint = (
        f"Write it as a literal list of integers (`{label}=[0, 1]`), or as a name bound to "
        f"one outside the kernel (`IDS = [0, 1]` ... `{label}=IDS`)."
    )
    if isinstance(node, (ast.List, ast.Tuple)):
        ids = []
        for element in node.elts:
            value = _const_int(element)
            if value is None:
                raise ValueError(
                    f"pipeline: cross-core buffer '{bufname}' {label} contains "
                    f"`{ast.unparse(element)}`, which is not a compile-time integer. {hint}"
                )
            ids.append(value)
        return ids

    if isinstance(node, ast.Name):
        value = closure_vars.get(node.id)
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"pipeline: cross-core buffer '{bufname}' {label} is `{node.id}`, which is "
                f"not bound to a list or tuple of integers outside the kernel. {hint}"
            )
        bad = [v for v in value if not isinstance(v, int) or isinstance(v, bool)]
        if bad:
            raise ValueError(
                f"pipeline: cross-core buffer '{bufname}' {label} is `{node.id}` = {value!r}, "
                f"which holds non-integer element(s) {bad}. {hint}"
            )
        return list(value)

    raise ValueError(
        f"pipeline: cross-core buffer '{bufname}' {label} is `{ast.unparse(node)}`, which is "
        f"not a supported spelling. {hint}"
    )


def _tile_arg_buffer(arg: ast.expr, stage_slot_to_buffer: dict[str, str]) -> str | None:
    """If arg is a bare slot variable (from group.next()) that maps to a buffer,
    return the buffer name; else None.

    New API: group.next() returns a bare tile, so op args are plain Names
    (e.g. pl.move(qk_left, cur_k)), not `slot.tile` attributes.
    """
    if isinstance(arg, ast.Name):
        return stage_slot_to_buffer.get(arg.id)
    return None


def _block_op_pipe(
    op_name: str, call: ast.Call, stage_slot_to_buffer: dict[str, str], all_buffer_memory: dict[str, MemorySpace]
) -> str:
    """Determine the pipe name for a block op accessing a cross-core buffer.

    Raises when the pipe cannot be determined. There is no safe default: the pipe decides
    which queue the wait/set lands on, so guessing puts the sync on a queue the op never
    runs on — and a section only has some of the pipes, so a guess can even name one that
    does not exist on that core. Either way the dependency goes unenforced, silently.
    """
    if op_name == "move":
        # move(dst, src): pipe depends on src/dst memory
        dst_mem = _arg_memory(call.args[0] if call.args else None, stage_slot_to_buffer, all_buffer_memory)
        src_mem = _arg_memory(call.args[1] if len(call.args) > 1 else None, stage_slot_to_buffer, all_buffer_memory)
        if src_mem is None or dst_mem is None:
            unresolved = "destination" if dst_mem is None else "source"
            raise ValueError(
                f"pipeline: cannot determine the pipe of `pl.move` at line {call.lineno}, "
                f"because its {unresolved} tile does not resolve to a declared tile group. "
                f"The move touches a cross-core buffer, so its pipe decides where the sync "
                f"goes. A tile reached through an aggregate (e.g. `tile_groups.x.next()`) is "
                f"not yet traced; pass the tile group to the stage as its own argument."
            )
        return _pipe_name(get_move_pipe(src_mem, dst_mem))
    if op_name in ("store", "store_tile"):
        src_mem = _arg_memory(call.args[1] if len(call.args) > 1 else None, stage_slot_to_buffer, all_buffer_memory)
        if src_mem is None:
            raise ValueError(
                f"pipeline: cannot determine the pipe of `pl.{op_name}` at line {call.lineno}, "
                f"because its source tile does not resolve to a declared tile group. The store "
                f"touches a cross-core buffer, so its pipe decides where the sync goes. A tile "
                f"reached through an aggregate (e.g. `tile_groups.x.next()`) is not yet traced; "
                f"pass the tile group to the stage as its own argument."
            )
        return _pipe_name(get_store_pipe(src_mem))
    pipe = get_op_pipe(op_name)
    if pipe is None:
        raise ValueError(
            f"pipeline: `pl.{op_name}` at line {call.lineno} touches a cross-core buffer, but "
            f"no pipe is registered for it, so the sync has nowhere to go. Register the op's "
            f"pipe (see get_op_pipe) or keep the cross-core buffer out of this op."
        )
    return _pipe_name(pipe)


def _arg_memory(arg, stage_slot_to_buffer, all_buffer_memory) -> MemorySpace | None:
    """Get the MemorySpace of a `slot.tile` arg (any buffer, cross-core or local)."""
    buf = _tile_arg_buffer(arg, stage_slot_to_buffer) if arg is not None else None
    if buf is not None:
        return all_buffer_memory.get(buf)
    return None


def _pipe_name(pipe) -> str:
    """PipeType enum -> short name string used in generated pl.PipeType.<NAME>."""
    # pipe is a PipeType enum; its name attribute gives FIX/V/MTE1/...
    return getattr(pipe, "name", str(pipe).split(".")[-1])


def _record_region_access(out: list, buf: str, role: str, pipe: str) -> None:
    """Append one op-level access.

    Kept per-op rather than collapsed to (first_pipe, last_pipe): the graph needs each
    access as its own node, and a pair of endpoints cannot express a local access sitting
    BETWEEN two cross-core ones. The role comes straight from the op's argument roles — a
    buffer may legitimately be read and written within one stage.
    """
    out.append((buf, role, pipe))


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


def _merge_role(result: dict[str, str], key: str, role: str | None) -> None:
    """Fold one more role for ``key`` into ``result``: differing roles become "RW".

    Used both for a VF's parameters and for the arguments of a single block op. In the
    latter, "differing roles for one key" means the same buffer reached the op at both a
    read and a write position — an in-place call such as ``pl.muls(x, x, 2.0)``. That is
    ONE access to the buffer, not two, and recording it as two is what let its read find
    its own write as the nearest preceding one and stop there, never reaching the producer.
    """
    existing = result.get(key)
    if existing is None:
        result[key] = role
    elif existing != role and role is not None:
        result[key] = "RW"


def _vf_param_aliases(vf_func_def: ast.FunctionDef, param_names: set[str]) -> dict[str, str]:
    """Local names in a VF body that stand for one of its parameters.

    A VF reaches UB only through its parameters, but it rarely names them at the point of
    use. Two spellings cover every alias in the tree: ``src_ub0 = input_tile`` and
    ``src_ub1 = input_tile + TS_HALF`` — a plain rename, or a rename at an offset. Both name
    memory belonging to that parameter, so a load or store through either is a load or store
    of the parameter.

    Chains resolve by iterating to a fixed point. Without this the role was simply dropped:
    `_vf_scatter_backward` writes through two such aliases, which left one parameter with no
    role at all and marked another read-only when it is read AND written.
    """
    aliases: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(vf_func_def):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if not isinstance(node.targets[0], ast.Name):
                continue
            target = node.targets[0].id
            if target in param_names or target in aliases:
                continue
            # _root_name walks a BinOp down its left side, so `param + off` and a bare
            # rename land on the same answer.
            if not isinstance(node.value, (ast.Name, ast.BinOp)):
                continue
            source = _root_name(node.value)
            if source in param_names:
                aliases[target] = source
                changed = True
            elif source in aliases:
                aliases[target] = aliases[source]
                changed = True
    return aliases


def _vf_pointer_param(arg: ast.expr, param_names: set[str], aliases: dict[str, str]) -> str | None:
    """The VF parameter this pointer argument refers to, directly or through an alias."""
    root = _root_name(arg)
    if root is None:
        return None
    if root in param_names:
        return root
    return aliases.get(root)


def _is_vf_command(call: ast.Call) -> bool:
    """``vf.<op>(...)`` — one machine op, as opposed to a call to another vector function."""
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "vf"
    )


def _nested_vf_def(call: ast.Call, vf_func_defs: dict) -> ast.FunctionDef | None:
    """The vector function this call invokes, or None if it does not invoke one.

    Membership of ``vf_func_defs`` is the test, and that set is built from the decorator
    (see _collect_callable_defs) — the surest signal there is, and one a name alone does
    not give.
    """
    if not isinstance(call.func, ast.Name):
        return None
    return vf_func_defs.get(call.func.id)


def _record_vf_call_role(
    call: ast.Call, vf_name: str, param_names: set[str], aliases: dict[str, str],
    tile_params: dict[str, str], result: dict[str, str]
) -> None:
    """Record what one ``vf.<op>(...)`` call does to the tiles it was handed.

    Reads the argument roles from _VF_OP_TILE_ROLES rather than deducing them from the op
    name. There is nothing to deduce from: `gather` reads a tile, `scatter` writes one, and
    the unaligned loads keep their alignment-state register where every other op keeps the
    pointer. This used to guess the pointer's position from the argument count, on the
    premise that the aligned load had a statement form taking the destination register
    first — a form its own documentation says does not exist.

    Only arguments naming a parameter in ``tile_params`` are considered. The rest cannot
    matter: a VF reaches UB solely through its parameters, and the ones missing from that
    map hold scalars at this call site. That is also what makes the refusal below safe to
    raise on the spot — reaching an untabled op with a tile is unambiguous, whereas most vf
    ops take no tile at all and their absence from the table is the normal case.

    ``call`` must be a vf command; the caller decides that (see _is_vf_command).
    """
    op = call.func.attr
    roles = _VF_OP_TILE_ROLES.get(op)
    for argpos, arg in enumerate(call.args):
        param = _vf_pointer_param(arg, param_names, aliases)
        if param is None or param not in tile_params:
            continue
        if roles is None:
            raise ValueError(
                f"pipeline: `vf.{op}` at line {call.lineno}, in vector function "
                f"'{vf_name}', is handed buffer '{tile_params[param]}' (as '{param}'), but "
                f"_VF_OP_TILE_ROLES has no entry for it, so the scan cannot tell whether it "
                f"reads or writes the tile — and which way a cross-core wait/set goes "
                f"depends on that.\n"
                f"If the op touches a UB tile, add it to _VF_OP_TILE_ROLES in "
                f"pypto_pro/language/parser/_op_pipeline.py, giving each argument position "
                f"'R', 'W', 'RW', or None for the positions that are not tiles."
            )
        if argpos < len(roles) and roles[argpos] is not None:
            _merge_role(result, param, roles[argpos])


def _scan_vf_roles(
    vf_func_def: ast.FunctionDef, vf_func_defs: dict, tile_params: dict[str, str], path: tuple = ()
) -> dict[str, str]:
    """R/W roles for the parameters of ``vf_func_def`` that hold a tile at this call site.

    ``tile_params`` maps such a parameter to the buffer it was handed, and is the reason
    this takes a call site at all: which parameters are tiles is not written down anywhere
    in a VF — they are as often scalars — but the caller resolved exactly that before
    getting here, so passing it down beats rediscovering or deferring it.

    A VF called from another VF is part of it, the same way a plain helper is part of the
    stage that calls it: its effect is folded into the caller's parameters by argument
    position, which is the whole of that folding, because a VF touches UB only through its
    parameters. The same positional mapping carries ``tile_params`` inward.

    ``path`` holds the VFs already on this chain. Guarding against the direct self-call
    alone, as this did, leaves A -> B -> A to recurse until the stack runs out.
    """
    param_names = {a.arg for a in vf_func_def.args.args if a.arg != "self"}
    aliases = _vf_param_aliases(vf_func_def, param_names)
    result: dict[str, str] = {}
    path = path + (vf_func_def.name,)

    # A call in a VF body is one of a closed set, and each kind is handled on its own line
    # rather than by falling through the other's guard:
    #
    #   vf.<op>(...)        one machine op                  -> _record_vf_call_role
    #   another VF          part of THIS one                -> recurse
    #   pl.range / pl.min   loop and scalar scaffolding      -> nothing to record
    #
    # Nothing else can appear: the parser refuses a vector function that calls an ordinary
    # helper ("cannot call non-vector inline function"), so there is no fourth case to
    # guess at here.
    for node in ast.walk(vf_func_def):
        if not isinstance(node, ast.Call):
            continue
        if _is_vf_command(node):
            _record_vf_call_role(node, vf_func_def.name, param_names, aliases, tile_params, result)
            continue
        callee_def = _nested_vf_def(node, vf_func_defs)
        if callee_def is None or callee_def.name in path:
            continue
        callee_params = [a.arg for a in callee_def.args.args if a.arg != "self"]
        # Which of the callee's parameters are tiles follows from which of ours are.
        callee_tiles: dict[str, str] = {}
        outer_of: dict[str, str] = {}
        for argpos, arg in enumerate(node.args):
            if argpos >= len(callee_params):
                break
            param = _vf_pointer_param(arg, param_names, aliases)
            if param is None or param not in tile_params:
                continue
            callee_tiles[callee_params[argpos]] = tile_params[param]
            outer_of[callee_params[argpos]] = param
        if not callee_tiles:
            continue
        callee_roles = _scan_vf_roles(callee_def, vf_func_defs, callee_tiles, path)
        for callee_param, role in callee_roles.items():
            _merge_role(result, outer_of[callee_param], role)
    return result
