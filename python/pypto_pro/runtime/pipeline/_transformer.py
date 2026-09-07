# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Transformer: convert serial kernel AST to preload pipeline AST."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from ._analyzer import PipelineInfo, analyze_pipeline
from ._astutil import (
    PL_IS_VALID_FIELD,
    PL_STRUCT_ARG,
    PL_TASK_ID_FIELD,
    call_name,
    slot_accessor,
    slot_index_field,
)
from ._stage import is_pipeline_stage
from ._sync_graph import SyncPlan, build_graph, plan_sync, sites_for
from ._validate import validate_names
from .config import PipelineConfig


def _rewrite_outer_slot_picks(func_body: list[ast.stmt], info: PipelineInfo) -> None:
    """Turn each outside-the-loop slot pick into an explicit index, in place.

    ``x = g.next()`` becomes ``_pl_idx_g = (_pl_idx_g + 1) % slots`` followed by
    ``x = g[_pl_idx_g]``; ``x = g[expr]`` becomes ``_pl_idx_g = expr`` followed by the same
    subscript. Either way the index that selected the slot survives as a plain scalar, so
    ctx can carry it and each stage can re-select the slot for its own beat.

    ``x`` itself stays: the statements around the pick (set_validshape, load, ...) use it,
    and they mean the slot of the CURRENT outer iteration, which is what ``x`` still is.
    Subscripting leaves the group's own cursor untouched, so mixing this with a next()
    elsewhere on the same group does not disturb its rotation.
    """
    if not info.outer_slots:
        return
    by_group = {var: gs for var, gs in info.outer_slots.items()}
    inside = {id(node) for node in ast.walk(info.pipeline_loop)} if info.pipeline_loop else set()

    class _Rewriter(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):  # noqa: N802 - ast API
            if id(node) in inside or len(node.targets) != 1:
                return node
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in by_group:
                return node
            if slot_accessor(node.value) is None:
                # A rename of a slot (``alias = cur``) carries the same entry so that the
                # stage's argument resolves, but only the pick itself may be rewritten:
                # rewriting the rename too would advance the group's index a second time.
                return node
            group, kind, slots = by_group[target.id]
            idx_name = slot_index_field(group)
            if kind == "index":
                index_assign = ast.Assign(
                    targets=[ast.Name(id=idx_name, ctx=ast.Store())],
                    value=copy.deepcopy(node.value.slice),
                    lineno=0,
                )
            else:
                # Only next() advances the group's cursor; current()/previous() read it
                # where it stands (see _buffer_parser._lower_group_accessor), so the
                # counter has to move by the same amount the accessor would.
                step = {"next": 1, "current": 0, "previous": -1}[kind]
                value: ast.expr = ast.Name(id=idx_name, ctx=ast.Load())
                if step:
                    value = ast.BinOp(
                        left=ast.BinOp(left=value, op=ast.Add(), right=ast.Constant(value=step)),
                        op=ast.Mod(),
                        right=ast.Constant(value=slots),
                    )
                index_assign = ast.Assign(
                    targets=[ast.Name(id=idx_name, ctx=ast.Store())],
                    value=value,
                    lineno=0,
                )
            pick = ast.Assign(
                targets=[ast.Name(id=target.id, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id=group, ctx=ast.Load()),
                    slice=ast.Name(id=idx_name, ctx=ast.Load()),
                    ctx=ast.Load(),
                ),
                lineno=0,
            )
            return [index_assign, pick]

    for i, stmt in enumerate(func_body):
        func_body[i] = _Rewriter().visit(stmt)


# ---------------------------------------------------------------------------
# Cross-core sync emission
#
# Everything that turns a SyncPlan into statements. Kept together because the sync path
# is read as a path: plan it (_build_sync_plan), find one stage's sites
# (_sites_for_stage), render them (_sync_stmts_for), and place the two halves that fall
# outside the loop (_emit_prefire_and_drain). These used to sit in four separate parts of
# the file, so following that path meant jumping between them.
#
# Nothing here decides WHERE a stage runs — that is arrangement, below. The one thing the
# two share is the schedule: _schedule_of reads the delays arrangement computed, and the
# graph is built from them.
# ---------------------------------------------------------------------------


def _build_event_id_index(ids_node: ast.expr, slot_count: int, index_expr: ast.expr) -> ast.expr:
    """Build `<ids_node>[<index_expr> % slot_count]`."""
    idx = ast.BinOp(
        left=copy.deepcopy(index_expr),
        op=ast.Mod(),
        right=ast.Constant(value=slot_count),
    )
    return ast.Subscript(
        value=copy.deepcopy(ids_node),
        slice=idx,
        ctx=ast.Load(),
    )


def _build_system_sync_stmt(fn_name: str, pipe_name: str, event_id: ast.expr) -> ast.stmt:
    """Build `pl.system.<fn_name>(pipe=pl.PipeType.<pipe>, event_id=<event_id>)`."""
    call = ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="system", ctx=ast.Load()),
            attr=fn_name,
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[
            ast.keyword(
                arg="pipe",
                value=ast.Attribute(
                    value=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="PipeType", ctx=ast.Load()),
                    attr=pipe_name,
                    ctx=ast.Load(),
                ),
            ),
            ast.keyword(arg="event_id", value=event_id),
        ],
    )
    return ast.Expr(value=call, lineno=0)


# Sync-only's task counter. It plays the role the ctx slot's _pl_task_id plays in the full
# pipeline: the running task number that indexes each buffer's event-id tuple. Kept a plain
# variable because that path has no ctx to carry it in.
_PL_SYNC_ID = "_pl_sync_id"


def _emit_sync_site(site, index_expr: ast.expr) -> ast.stmt:
    """Build the sync statement for one SyncSite."""
    event_id = _build_event_id_index(site.ids_node, site.slot_count, index_expr)
    return _build_system_sync_stmt(site.op, site.pipe, event_id)


def _sync_stmts_for(sites: list, index_expr: ast.expr):
    """(pre_stmts, post_stmts) for one stage from its planned SyncSites.

    Emitted unconditionally: a skewed edge's first few waits are covered by pre-fire rather
    than by a guard, so nothing here has to ask whether the partner task exists.
    """
    pre, post = [], []
    for out, planned in ((pre, sites[0]), (post, sites[1])):
        for site in planned:
            out.append(_emit_sync_site(site, index_expr))
    return pre, post


def _build_sync_plan(info: PipelineInfo):
    """Build the sync graph for the current schedule and plan every instruction from it.

    Called once per transform, at the point the schedule is known. The result travels down
    to the emitters as a value: planning twice would allocate a second event-id group for
    every address-reuse edge and declare it again.
    """
    return plan_sync(build_graph(info, _schedule_of(info)), info)


def _schedule_of(info: PipelineInfo) -> list:
    """Which beat each stage runs on — the one input the sync graph needs from arrangement.

    Read off the stages rather than passed around: the delays ARE the schedule, and a second
    copy travelling alongside ``info`` could only ever disagree with it.

    Sync-only is the degenerate case, not a separate one: _compute_delays never runs there,
    so every delay is still its constructed 0 and this yields an all-zero schedule. Both
    emission paths therefore ask the graph the same question, and the fact that one of them
    has no skew shows up as data rather than as a second code path.
    """
    return [stage.delay for stage in info.stages]


def _sites_for_stage(sync: SyncPlan, stage) -> tuple:
    """(pre_sites, post_sites) for one stage, out of the already-planned in-loop sync."""
    return sites_for(sync.sites, stage.func_name)


def _build_lifted_id_decls(info: PipelineInfo) -> list[ast.stmt]:
    """Declare lifted literal fwd_ids/bwd_ids as variables.

    The buffer's ids_node was rewritten to a Name reference, so sync code uses
    `var[idx]` instead of the unsupported `(1,2)[idx]` subscript. Shared by the
    full-pipeline path (_build_declarations) and the serial one (_build_serial_decls).
    """
    stmts = []
    for var_name, literal_node in info.sync.lifted_ids:
        stmts.append(
            ast.Assign(
                targets=[ast.Name(id=var_name, ctx=ast.Store())],
                value=copy.deepcopy(literal_node),
                lineno=0,
            )
        )
    return stmts


def _build_serial_decls(info: PipelineInfo) -> list[ast.stmt]:
    """Sync-only's pre-loop declarations: the lifted event-id variables + `_pl_sync_id = 0`.

    Declarations only; the caller appends the sync pre-fire, as on the full-pipeline path.
    The lifted ids must stay first either way — the pre-fire sets index those variables."""
    decls = list(_build_lifted_id_decls(info))
    decls.append(
        ast.Assign(targets=[ast.Name(id=_PL_SYNC_ID, ctx=ast.Store())], value=ast.Constant(value=0), lineno=0)
    )
    return decls


def _insert_sync_into_loop_body(body: list[ast.stmt], stage_by_name: dict, sync: SyncPlan) -> None:
    """Insert wait(before)/set(after) cross-core sync around each stage call in
    the loop body. Index base is `_pl_sync_id` (current iteration, no delay).

    A stage call sits inside a `with pl.section_*()` block, so its sync is wrapped in
    that same section.
    """
    index_expr = ast.Name(id=_PL_SYNC_ID, ctx=ast.Load())
    for i, stmt in enumerate(body):
        if not isinstance(stmt, ast.With):
            continue
        # Leaf stage: call inside a section block.
        for j, inner in enumerate(stmt.body):
            if isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Call):
                fname = call_name(inner.value)
                stage = stage_by_name.get(fname)
                if stage is not None:
                    sites = _sites_for_stage(sync, stage)
                    pre, post = _sync_stmts_for(sites, index_expr)
                    stmt.body[j:j + 1] = pre + [inner] + post
                    break


def _emit_prefire_and_drain(sync: SyncPlan) -> tuple[list[ast.stmt], list[ast.stmt]]:
    """Emit the plan's two out-of-loop halves: (sync_prefire, sync_drain).

    Pre-fire supplies the permits a skewed edge's first waits have no partner for; the drain
    consumes both those permits and the sets the same skew strands at the end, so the two
    counts match. Both sit outside the loop, where no task counter exists, so their event ids
    are indexed by a literal.

    Both carry a ``sync_`` prefix at the call sites, because the full-pipeline path emits
    drain BEATS (_build_drain_beats) after the same loop and the two are unrelated: the
    beats let the delayed stages finish their tasks, this balances set/wait counts. The
    prefix marks which of the two concerns a list belongs to — the same split the whole
    transform is organised around — rather than inventing a second word for `sync.drain`.
    """

    def emit(sites) -> list[ast.stmt]:
        by_section: dict[str, list] = {}
        for site in sites:
            by_section.setdefault(site.section, []).append((site.op, site.pipe, site.ids_node, site.id_index))
        out: list[ast.stmt] = []
        for section_kind, entries in by_section.items():
            body = [
                _build_system_sync_stmt(
                    op, pipe, ast.Subscript(value=copy.deepcopy(ids), slice=ast.Constant(value=idx), ctx=ast.Load())
                )
                for op, pipe, ids, idx in entries
            ]
            out.append(_wrap_in_section(section_kind, body))
        return out

    return emit(sync.prefire), emit(sync.drain)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def transform_pipeline(
    func_def: ast.FunctionDef,
    closure_vars: dict,
    config: PipelineConfig,
    if_const_map: dict | None = None,
    var_types: dict | None = None,
    tilingkey_consts: dict | None = None,
    datatype_consts: dict | None = None,
) -> ast.FunctionDef:
    """Transform a serial kernel function into a preload pipeline version.

    Args:
        func_def: The kernel function's AST (will not be mutated; a copy is made)
        closure_vars: Closure variables (contains stage functions, module constants)
        config: Pipeline configuration (depth, dump_generated, etc.)
        var_types: {name: IR type class name} from the parser probe; decides how each
            stage argument reaches its stage (see _derive_ctx_fields).
        if_const_map: {(lineno, col_offset) of test: (is_const, value)} for every ``if``
            in the kernel body, produced by a probe parse (reuses the parser's constant
            folding).
            Drives the compile-time constant branch collapse before analysis.
        tilingkey_consts: This launch key's concrete tilingkey field values.
        datatype_consts: This launch's concrete datatype symbol values.

    Returns:
        New ast.FunctionDef with pipeline transformation applied
    """
    # Work on a deep copy to avoid mutating the original
    new_func = copy.deepcopy(func_def)

    # Same namespace the parser evaluates in (see ASTParser.__init__): a tilingkey field
    # or datatype symbol is a compile-time constant for this key, but lives outside
    # closure_vars, so anything resolving names here needs them folded in. Without them a
    # `dtype=input_dtype` buffer silently loses its address range.
    closure_vars = {
        **(closure_vars or {}),
        **(tilingkey_consts or {}),
        **(datatype_consts or {}),
    }

    # Prune dead compile-time-constant if/elif/else branches that wrap stage calls,
    # so downstream analysis/transform only ever sees plain unconditional stages.
    _prune_const_branches(new_func, closure_vars, if_const_map or {})

    # Analyze the serial structure (raises if no usable stages / pipeline loop)
    info = analyze_pipeline(new_func, closure_vars, var_types)

    if config.preload == 0:
        # Validation mode: keep the serial loop, only auto-insert cross-core sync.
        validate_names(new_func, _serial_var_names())
        _transform_serial(new_func.body, info, _build_sync_plan(info))
        ast.fix_missing_locations(new_func)
        return new_func

    # Compute delays based on preload
    _compute_delays(info, config.preload)

    # The names to check depend on the delays: one ctx variable per distinct delay.
    validate_names(new_func, _pipeline_var_names(info))
    # One gate for every check that needs the schedule (see _validate).

    # ctx ring buffer depth = max_delay + 1
    max_delay = max((s.delay for s in info.stages), default=0)
    depth = max_delay + 1

    # Find and replace the pipeline loop in the function body.
    _transform_body(new_func.body, info, depth, _build_sync_plan(info))

    ast.fix_missing_locations(new_func)
    return new_func


def _prune_const_branches(func_def: ast.FunctionDef, closure_vars: dict, if_const_map: dict) -> None:
    """Splice out compile-time-constant if/else around stage calls, in place.

    The taken branch's body replaces the whole ``if`` so downstream only sees plain
    unconditional stage calls. ``if_const_map`` maps a condition's source position
    ``(lineno, col_offset)`` -> (is_const, value), from a probe parse that reuses the
    parser's constant folding. Raises on a dynamic condition around a stage call.
    """
    stage_func_names = {name for name, val in closure_vars.items() if is_pipeline_stage(val)}
    func_def.body = _prune_stmts(func_def.body, stage_func_names, if_const_map)


def _prune_stmts(stmts: list[ast.stmt], stage_func_names: set, if_const_map: dict) -> list[ast.stmt]:
    """Return a new statement list with dead constant stage-bearing branches pruned.
    Recurses into For/With/If bodies; drops ``with`` sections that become empty."""
    out: list[ast.stmt] = []
    for stmt in stmts:
        if isinstance(stmt, ast.If):
            out.extend(_prune_if(stmt, stage_func_names, if_const_map))
        elif isinstance(stmt, ast.For):
            stmt.body = _prune_stmts(stmt.body, stage_func_names, if_const_map)
            stmt.orelse = _prune_stmts(stmt.orelse, stage_func_names, if_const_map)
            out.append(stmt)
        elif isinstance(stmt, ast.With):
            stmt.body = _prune_stmts(stmt.body, stage_func_names, if_const_map)
            if stmt.body:  # drop a section block emptied by pruning
                out.append(stmt)
        else:
            out.append(stmt)
    return out


def _prune_if(if_stmt: ast.If, stage_func_names: set, if_const_map: dict) -> list[ast.stmt]:
    """Prune a single ``if`` (recursing into elif chains via orelse). Returns the
    replacement statement list for this ``if``.

    Rule: **any compile-time-constant condition is pruned, regardless of what its
    branches contain** (stage calls, buffer declarations like a cross-core
    make_tile_group, plain scalar setup — all of it). Since the condition is known
    at compile time, only the taken branch can run; keeping the dead branch would
    leave downstream scans (cross-core buffer detection, stage arrangement, sync)
    to see declarations/stages that never execute. Pruning gives every downstream
    pass a single, unambiguous AST.

    A runtime (non-constant) condition is left untouched; we only recurse into it
    to prune any nested constant ``if``. But a runtime branch must not wrap a stage
    call — that cannot be auto-arranged/synced → CB3.
    """
    # Keyed by the condition's source position, not its text: two sites can be written
    # the same way and fold to different values (see _record_if_const).
    is_const, value = if_const_map.get(
        (if_stmt.test.lineno, if_stmt.test.col_offset), (False, None)
    )

    if not is_const:
        # Runtime conditional: keep it, but a stage call directly inside a runtime
        # branch is unsupported (dynamic dispatch of stages).
        if _branch_contains_stage(if_stmt, stage_func_names):
            raise ValueError(
                f"pipeline (CB3): branch condition '{ast.unparse(if_stmt.test)}' at line "
                f"{if_stmt.test.lineno} wraps a stage call but is not a compile-time "
                "constant. Dynamic branch stages are unsupported; use a compile-time "
                "constant condition (e.g. a tiling-key field) or insert sync manually."
            )
        if_stmt.body = _prune_stmts(if_stmt.body, stage_func_names, if_const_map)
        if_stmt.orelse = _prune_stmts(if_stmt.orelse, stage_func_names, if_const_map)
        return [if_stmt]

    # Compile-time constant: keep only the taken branch (elif chains are nested If
    # nodes in orelse; a plain else is a stmt list; no else → the if vanishes).
    taken = if_stmt.body if value else if_stmt.orelse
    return _prune_stmts(taken, stage_func_names, if_const_map)


def _branch_contains_stage(if_stmt: ast.If, stage_func_names: set) -> bool:
    """True if any branch of this if (then/elif/else, recursively) calls a stage."""
    for node in ast.walk(if_stmt):
        if isinstance(node, ast.Call) and call_name(node) in stage_func_names:
            return True
    return False


def _build_counter_incr(name: str) -> ast.stmt:
    """Build `<name> = <name> + 1`."""
    return ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())],
        value=ast.BinOp(
            left=ast.Name(id=name, ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(value=1),
        ),
        lineno=0,
    )


def _transform_serial(func_body: list[ast.stmt], info: PipelineInfo, sync: SyncPlan) -> None:
    """Validation mode: keep the serial loop structure, only auto-insert cross-core
    sync. Two placement rules (that's all there is to it):

      1. Stage-to-stage sync goes INSIDE the innermost pipeline loop (the
         pipeline loop): wait/set wrapped around each stage call,
         indexed by `_pl_sync_id % slot_count`.
      2. decls (lifted ids + `_pl_sync_id = 0`) + pre-fire go BEFORE the OUTERMOST
         loop and the sync drain AFTER it, so they run exactly once and `_pl_sync_id`
         advances continuously across all outer iterations.

    No ctx / guard / delay / extra.
    """
    stage_by_name = {s.func_name: s for s in info.stages}
    # Serial: every stage runs in the same beat, so a zero schedule.
    sync_prefire, sync_drain = _emit_prefire_and_drain(sync)

    def transform_loop(site: _LoopSite):
        # Rule 1: stage sync + per-iteration counter, inside the innermost loop.
        # In place (return None); the loop node stays.
        _insert_sync_into_loop_body(site.pipeline_loop.body, stage_by_name, sync)
        site.pipeline_loop.body.append(_build_counter_incr(_PL_SYNC_ID))
        return None

    # Rule 2: decls + pre-fire before the outermost loop, the sync drain after it.
    _place_around_pipeline_loop(
        func_body, info,
        transform_loop=transform_loop,
        before_loop=_build_serial_decls(info) + sync_prefire,
        after_loop=sync_drain,
    )


def _place_around_pipeline_loop(
    func_body: list[ast.stmt],
    info: PipelineInfo,
    transform_loop,
    before_loop: list[ast.stmt],
    after_loop: list[ast.stmt],
) -> None:
    """Shared skeleton for both the serial and full-pipeline transforms.

    Locates the pipeline loop and its enclosing outermost loop, then:
      1. calls ``transform_loop(site)`` for the path-specific loop handling. It
         returns either ``None`` (loop mutated in place — serial) or a list of
         statements to REPLACE the pipeline loop node with (full pipeline). It must
         NOT mutate the tree (indices must stay valid until the splices below).
      2. splices ``before_loop`` BEFORE the outermost loop and ``after_loop`` AFTER it.

    Both lists are named for WHERE they go, not what they hold: the two emission paths put
    different things there (the serial path only balances the sync, the full pipeline also runs
    the drain beats), and naming them after one path's contents misreads the other's.
    """
    site = _locate_pipeline_loops(func_body, info.pipeline_loop)
    if site is None:
        # analyze_pipeline found this very node in this very tree, so failing to find it
        # again means the two searches have drifted apart. Refuse rather than return
        # quietly: a silent no-op emits the serial kernel with neither ctx nor sync.
        raise ValueError(
            "pipeline (internal): the pipeline loop the analyzer found is no longer "
            "reachable in the kernel body — please report."
        )
    replacement = transform_loop(site)

    top_level = site.pipeline_stmts is site.outer_stmts and site.pipeline_idx == site.outer_idx
    if replacement is not None and top_level:
        # Outermost loop IS the pipeline loop, replaced by N stmts. Splice everything in one go.
        site.outer_stmts[site.outer_idx:site.outer_idx + 1] = before_loop + replacement + after_loop
        return

    if replacement is not None:
        # Nested: replace the inner pipeline loop; the outer loop node is untouched,
        # so its (stmts, idx) stay valid for the splices below.
        site.pipeline_stmts[site.pipeline_idx:site.pipeline_idx + 1] = replacement

    # Splice the after-loop part at outer_idx+1 first (so outer_idx is unaffected).
    site.outer_stmts[site.outer_idx + 1:site.outer_idx + 1] = after_loop
    site.outer_stmts[site.outer_idx:site.outer_idx] = before_loop


@dataclass
class _LoopSite:
    """Where the pipeline loop lives, and its enclosing loop nest.

    - pipeline_stmts/pipeline_idx: the pipeline loop — where stage
      sync / loop replacement happens.
    - outer_stmts/outer_idx: the OUTERMOST enclosing loop — where declarations go
      before / the sync drain after (run-once placement). When the pipeline loop is
      top-level, coincides with the pipeline loop.
    """
    pipeline_stmts: list  # statement list that directly holds the pipeline loop
    pipeline_idx: int     # index of the pipeline `for <pipeline_loop_var>` in pipeline_stmts
    outer_stmts: list     # statement list that holds the outermost enclosing loop
    outer_idx: int        # index of the outermost loop in outer_stmts

    @property
    def pipeline_loop(self) -> ast.For:
        return self.pipeline_stmts[self.pipeline_idx]


def _locate_pipeline_loops(func_body: list[ast.stmt], pipeline_loop: ast.For) -> "_LoopSite | None":
    """Locate the pipeline loop and its enclosing outermost loop; return a _LoopSite
    (or None if there is no pipeline loop).

    Scans each statement list for a For; descends into For/With bodies. The first
    For on the path down to the pipeline loop is the outermost loop; the pipeline loop
    itself is where the transform happens. When the pipeline loop is top-level, the two
    coincide.

    Matched by node identity, never by loop-variable name: a kernel may well have several
    loops over the same variable (a prologue loop and the pipelined one both counting
    ``work_id``, say), and matching by name would replace whichever came first — dropping
    that loop's body and emitting the stages against the wrong nest.
    """
    for i, stmt in enumerate(func_body):
        if isinstance(stmt, ast.For):
            if stmt is pipeline_loop:
                return _LoopSite(func_body, i, func_body, i)  # top-level pipeline loop
            inner_site = _find_loop_site(stmt.body, pipeline_loop)
            if inner_site is not None:
                p_stmts, p_idx = inner_site
                return _LoopSite(p_stmts, p_idx, func_body, i)
        elif isinstance(stmt, ast.With):
            inner = _locate_pipeline_loops(stmt.body, pipeline_loop)
            if inner is not None:
                return inner
    return None


def _find_loop_site(stmts: list[ast.stmt], pipeline_loop: ast.For):
    """Return (stmts, index) of ``pipeline_loop`` within ``stmts`` (descending uniformly
    through For/With bodies), or None if not found. Identity match — see
    _locate_pipeline_loops for why the loop variable's name will not do."""
    for i, stmt in enumerate(stmts):
        if stmt is pipeline_loop:
            return stmts, i
        if isinstance(stmt, (ast.For, ast.With)):
            found = _find_loop_site(stmt.body, pipeline_loop)
            if found is not None:
                return found
    return None


def _compute_delays(info, preload: int):
    """Compute delay for each stage based on preload value.

    Rules:
      - Each core maintains a per-core stage counter (starting from 0)
      - First stage of a core: delay = upstream cross-core stage delay + 1
        (or 0 if no upstream)
      - Subsequent stages of same core: delay = same-core previous stage delay + preload
    """
    # Track per-core stage counters and last delay
    core_stage_count = {"cube": 0, "vector": 0}
    core_last_delay = {"cube": -1, "vector": -1}

    for stage in info.stages:
        core = stage.section_kind
        stage_count = core_stage_count.get(core)
        last_delay = core_last_delay.get(core)
        if stage_count is None or last_delay is None:
            raise KeyError(f"Unsupported pipeline section kind: {core}")

        if stage_count == 0:
            # First stage of this core
            if stage is info.stages[0]:
                # Very first stage overall
                stage.delay = 0
            else:
                # First stage of this core: previous stage (cross-core) delay + 1
                prev_idx = info.stages.index(stage) - 1
                stage.delay = info.stages[prev_idx].delay + 1
        else:
            # Subsequent stage of same core: previous same-core delay + preload
            stage.delay = last_delay + preload

        core_last_delay[core] = stage.delay
        core_stage_count[core] = stage_count + 1


def _ctx_var_name(delay: int) -> str:
    """The ctx variable a stage at this delay reads its beat from.

    Delay 0 reads the slot being filled this beat; a delayed stage reads the slot filled
    ``delay`` beats ago, looked up once per beat by _build_stage_ctx_lookup. Everything that
    has to name one of those variables — the lookup that declares it, the guards that read
    fields off it, the collision check that lists framework names — goes through here, so
    the name a stage is given and the name its guard reads cannot come apart.
    """
    return "_pl_ctx_0" if delay == 0 else f"_pl_ctx_neg{delay}"


def _pipeline_var_names(info: PipelineInfo) -> set[str]:
    """The fixed variables the FULL PIPELINE path introduces.

    One entry per distinct delay, because that is how many ctx variables the emitted body
    declares. The serial path emits none of these — see _serial_var_names.
    """
    names = {"_pl_ctx_arr", PL_TASK_ID_FIELD}
    for stage in info.stages:
        names.add(_ctx_var_name(stage.delay))
    return names


def _serial_var_names() -> set[str]:
    """The fixed variables the SYNC-ONLY path introduces: just its task counter.

    No ctx array, no per-delay ctx variables, no drain count — that path keeps the serial
    loop and only threads a counter through the event-id indexing.
    """
    return {_PL_SYNC_ID}


def _transform_body(
    func_body: list[ast.stmt],
    info: PipelineInfo,
    depth: int,
    sync: SyncPlan,
) -> None:
    """Full-pipeline transform. Same two placement rules as the serial path (via the
    shared _place_around_pipeline_loop skeleton):

      1. The innermost pipeline loop is REPLACED by its preload-pipeline version
         (_build_pipeline_loop: ctx ring buffer + delay + is_valid guards).
      2. declarations (_pl_ctx_arr, _pl_task_id, lifted ids, pre-fire) go BEFORE
         the OUTERMOST loop; AFTER it come the drain BEATS that let the delayed stages
         catch up, then the sync DRAIN that balances set(bwd)/wait(bwd). The two are
         unrelated despite both sitting there — see _emit_prefire_and_drain.
    """
    # Make the index behind every outside-the-loop slot pick explicit first, so the ctx can
    # carry it and each stage re-selects its own beat's slot.
    _rewrite_outer_slot_picks(func_body, info)

    def transform_loop(site: _LoopSite):
        # Pure: only build the replacement. Does NOT mutate the tree here, so
        # pipeline_idx stays valid for the skeleton's splice.
        return _build_pipeline_loop(site.pipeline_loop, info, depth, sync)

    sync_prefire, sync_drain = _emit_prefire_and_drain(sync)
    _place_around_pipeline_loop(
        func_body, info,
        transform_loop=transform_loop,
        before_loop=_build_declarations(info, depth) + sync_prefire,
        # Drain beats first, then the sync drain: the beats emit the set(bwd) the last
        # tasks still owe, which the waits in the sync drain then consume.
        after_loop=_build_drain_beats(info, depth, sync) + sync_drain,
    )


def _build_declarations(info: PipelineInfo, depth: int) -> list[ast.stmt]:
    """The full pipeline's pre-loop declarations: `_pl_ctx_arr`, `_pl_task_id`, the
    lifted event-id variables and the outer-slot rotation counters.

    Declarations only. The sync pre-fire that also sits before the loop is appended by the
    caller, so that each of the two things in that block still has a name of its own —
    setting up the arrangement's state, and releasing the sync's opening permits."""
    stmts = []

    keywords = [ast.keyword(arg=f, value=ast.Constant(value=0)) for f in info.ctx_fields]
    ctx_call = ast.Call(
        func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="struct_array", ctx=ast.Load()),
        args=[ast.Constant(value=depth), ast.Constant(value="PipeCtx")],
        keywords=keywords,
    )
    ctx_assign = ast.Assign(
        targets=[ast.Name(id="_pl_ctx_arr", ctx=ast.Store())],
        value=ctx_call,
        lineno=0,
    )
    stmts.append(ctx_assign)

    _pl_task_id_assign = ast.Assign(
        targets=[ast.Name(id=PL_TASK_ID_FIELD, ctx=ast.Store())],
        value=ast.Constant(value=0),
        lineno=0,
    )
    stmts.append(_pl_task_id_assign)

    # Lifted ids (shared with the serial path).
    stmts.extend(_build_lifted_id_decls(info))

    # Rotation counters for slots picked outside the pipeline loop. Starting at
    # `slots - 1` mirrors pl.make_tile_group's own cursor, which starts at the last slot so
    # that the first advance hands out slot 0 (see _buffer_parser._build_tile_group_ir).
    for group, slots in sorted({(g, n) for g, _kind, n in info.outer_slots.values()}):
        stmts.append(
            ast.Assign(
                targets=[ast.Name(id=slot_index_field(group), ctx=ast.Store())],
                value=ast.Constant(value=slots - 1),
                lineno=0,
            )
        )

    return stmts


def _ctx_field_assign(attr: str, value_node: ast.expr) -> ast.Assign:
    """Build `_pl_ctx_0.<attr> = <value_node>`."""
    return ast.Assign(
        targets=[ast.Attribute(value=ast.Name(id=_ctx_var_name(0), ctx=ast.Load()), attr=attr, ctx=ast.Store())],
        value=value_node,
        lineno=0,
    )


def _classify_loop_body_stmts(original_for: ast.For, info: PipelineInfo):
    """Walk the pipeline loop body IN ORDER and return an ordered_body list.

    Each entry is either a ("STAGE", stage_idx) marker for the statement that holds a stage
    call, or one of the user's own statements, kept verbatim. Their order here is the order
    they were written in; _build_pipeline_loop then emits all of the user's ahead of all the
    stages, keeping each group's own order.

    Which statement holds which stage is looked up, not decided again: the analyzer already
    settled it when it built info.stages (see StageCall.source_stmt). Deciding here — every
    ``ast.With`` is one stage — disagreed with the analyzer, which counts only the sections
    that actually hold a stage call. A section holding anything else, or any unrelated
    ``with``, then shifted every stage after it onto the wrong entry, and off the end of the
    list when the shift ran past it.

    Everything that is not a stage call is preserved, including such a section: written in
    the loop body it ran once per iteration, and kept there it runs once per beat, which is
    the same thing. The transform does not try to work out which statements "produce" a
    stage argument, only that they all run before the snapshot takes their values: any
    shape then works — a branch, a helper call, a struct field write, several statements
    building up one value — because nothing here has to model how a value was produced. A
    preserved assignment whose value is also carried in ctx is dead but harmless; codegen
    drops it.
    """
    stage_of_stmt = {
        id(stage.source_stmt): idx
        for idx, stage in enumerate(info.stages)
        if stage.source_stmt is not None
    }

    ordered_body = []
    for stmt in original_for.body:
        stage_idx = stage_of_stmt.get(id(stmt))
        if stage_idx is not None:
            ordered_body.append(("STAGE", stage_idx))
        else:
            ordered_body.append(copy.deepcopy(stmt))

    return ordered_body


def _build_pipeline_loop(
    original_for: ast.For, info: PipelineInfo, depth: int, sync: SyncPlan
) -> list[ast.stmt]:
    """Build the transformed pipeline for-loop."""

    new_for = copy.deepcopy(original_for)

    # Which loop-body statements hold a stage, and which are the user's own. The last stage
    # splits the user's in two, and the split is what the serial code already meant:
    # a statement AHEAD of it feeds a stage of this same iteration, so this beat's snapshot
    # has to see it, while one AFTER every stage is preparing the next iteration — the loop
    # counter being the usual one — and this beat's snapshot must NOT see it.
    ordered_body = _classify_loop_body_stmts(original_for, info)
    last_stage_pos = max(
        (i for i, item in enumerate(ordered_body) if isinstance(item, tuple)), default=-1
    )
    stage_indices = [item[1] for item in ordered_body if isinstance(item, tuple)]
    before_stages = [item for item in ordered_body[:last_stage_pos + 1] if not isinstance(item, tuple)]
    after_stages = list(ordered_body[last_stage_pos + 1:])

    # Assemble new loop body
    new_body = []
    # 1. Pick this beat's ctx slot and set is_valid.
    new_body.extend(_build_ctx_slot_pick(info, depth))
    # 2. The user's statements from before the last stage, in the order they wrote them —
    #    including any that sat BETWEEN two stage calls, which is what lets a value first
    #    assigned there reach the stage that consumes it. Lifting those over the stages is
    #    not an approximation: a stage may not return a value (_check_stage_returns), so its
    #    only effect is on buffers, and a statement touching no buffer cannot observe
    #    whether a stage has run. Leaving them in place is what made such a value fail to
    #    compile — the snapshot reads every ctx source, and this one was not assigned yet.
    new_body.extend(before_stages)
    # 3. Snapshot every ctx field, after those statements and before every stage, so each
    #    beat hands all its stages one consistent set of values.
    new_body.extend(_build_ctx_field_fills(info))
    # 4. The stages, in their original order.
    for stage_idx in stage_indices:
        new_body.extend(_build_stage_call(info.stages[stage_idx], stage_idx, info, depth, sync))
    # 5. Whatever the user wrote after the last stage, still after it — past the snapshot,
    #    so what it changes is picked up by the NEXT beat, which is the beat that uses it.
    new_body.extend(after_stages)
    # 6. task_id increment (frame-level iteration counter, after all stage calls)
    new_body.append(_build_counter_incr(PL_TASK_ID_FIELD))

    new_for.body = new_body
    return [new_for]


def _build_drain_beats(info: PipelineInfo, depth: int, sync: SyncPlan) -> list[ast.stmt]:
    """The beats that run after the outer loop to let the deepest stage catch up.

    A delayed stage is always ``delay`` beats behind, so when the task stream ends its last
    few tasks are still in flight. Those beats used to be appended to the innermost loop of
    the LAST outer iteration (`_pl_extra`), which assumed that iteration reaches the inner
    loop at all — a `continue` above it skipped the drain entirely, leaving a task half done
    and its set(bwd) unsent, so the post-loop wait(bwd) blocked forever.

    Draining here instead ties it to the end of the whole stream, which is what it means.

    The two sizes come from different places, which is what keeps this short: the BEAT count
    is the loop bound (``depth - 1``), while the number of stage blocks is the stage list.
    A deeper pipeline lengthens the loop, not the code. Stages are emitted by the same
    _build_stage_call as inside the pipeline loop, so their shape cannot drift.

    Only stages with a delay are emitted: a delay-0 stage reads the beat's own ctx slot,
    which is marked invalid here, so its guard would never fire.
    """
    if depth <= 1:
        return []
    body: list[ast.stmt] = [
        _build_ctx_slot_assign(depth),
        # No new task this beat. Each stage still checks its OWN (delayed) slot, so it keeps
        # firing while a real task remains behind it.
        _ctx_field_assign(PL_IS_VALID_FIELD, ast.Constant(value=0)),
        _ctx_field_assign(PL_TASK_ID_FIELD, ast.Name(id=PL_TASK_ID_FIELD, ctx=ast.Load())),
    ]
    for stage_idx, stage in enumerate(info.stages):
        if stage.delay:
            body.extend(_build_stage_call(stage, stage_idx, info, depth, sync))
    body.append(_build_counter_incr(PL_TASK_ID_FIELD))

    return [
        ast.For(
            target=ast.Name(id="_pl_drain", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="range", ctx=ast.Load()),
                args=[ast.Constant(value=0), ast.Constant(value=depth - 1)],
                keywords=[],
            ),
            body=body,
            orelse=[],
            lineno=0,
        )
    ]


def _build_ctx_slot_assign(depth: int) -> ast.Assign:
    """`_pl_ctx_0 = _pl_ctx_arr[_pl_task_id % depth]` — this beat's slot."""
    return ast.Assign(
        targets=[ast.Name(id=_ctx_var_name(0), ctx=ast.Store())],
        value=ast.Subscript(
            value=ast.Name(id="_pl_ctx_arr", ctx=ast.Load()),
            slice=ast.BinOp(
                left=ast.Name(id=PL_TASK_ID_FIELD, ctx=ast.Load()),
                op=ast.Mod(),
                right=ast.Constant(value=depth),
            ),
            ctx=ast.Load(),
        ),
        lineno=0,
    )


def _build_ctx_slot_pick(info: PipelineInfo, depth: int) -> list[ast.stmt]:
    """Pick this beat's ctx slot and set is_valid.

    The field values are filled separately (_build_ctx_field_fills), later in the body:
    they snapshot the user's variables and so must run after the statements that set them.
    """
    return [_build_ctx_slot_assign(depth), _build_is_valid_guard(info)]


def _build_is_valid_guard(info: PipelineInfo) -> ast.If:
    """Build `if <loop_var> < <range_end>: _pl_ctx_0._pl_is_valid = 1 else: = 0`.

    The end bound is always present: _find_pipeline_loop rejects a pipeline loop it cannot
    extract one from (L6), long before any of this runs.
    """
    return ast.If(
        test=ast.Compare(
            left=ast.Name(id=info.pipeline_loop_var, ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[copy.deepcopy(info.pipeline_loop_end)],
        ),
        body=[_ctx_field_assign(PL_IS_VALID_FIELD, ast.Constant(value=1))],
        orelse=[_ctx_field_assign(PL_IS_VALID_FIELD, ast.Constant(value=0))],
        lineno=0,
    )


def _build_ctx_field_fills(info: PipelineInfo) -> list[ast.stmt]:
    """Build `_pl_ctx_0.<field> = <value>` for EVERY ctx field (except the validity flag).

    Each field is filled by READING its source, at a point where the user's statements for
    this beat have already run (see _build_pipeline_loop). Snapshot rather than recompute:
    the framework never has to model how a value was produced, so a value assigned in a
    branch, inside a helper function, or across several statements all behave the same.

    A field coming from a struct reads `<struct>.<field>`; a scalar reads its variable
    (under the variable's own name, which may differ from the field name after a collision
    rename); a framework field such as _pl_task_id reads its same-named variable.

    A field whose source lives inside a ``pl.section_*()`` block gets its fill wrapped in
    that same section: an unwrapped fill is emitted for both targets, and the one whose
    parse skipped the block has no such name to fill from.

    A source bound in BOTH sections gets a fill in each. The two are different variables —
    each target compiles to its own function holding its own ctx array — so each fills its
    own copy from the name it can see. Picking just one section would leave the other
    target's copy at its initial zero.
    """
    # Field name -> (fill expression, the sections it must be emitted in; empty = anywhere).
    sources: dict[str, tuple[ast.expr, set]] = {}
    for struct_name, fields in info.struct_args.items():
        sections = info.var_sections.get(struct_name, set())
        for fname in fields:
            sources[fname] = (
                ast.Attribute(value=ast.Name(id=struct_name, ctx=ast.Load()), attr=fname, ctx=ast.Load()),
                sections,
            )
    for var_name, field_name in info.scalar_ctx_names.items():
        sources[field_name] = (ast.Name(id=var_name, ctx=ast.Load()), info.var_sections.get(var_name, set()))
    # A slot index belongs to the section its group is declared in: that is where the
    # counter is advanced, so that is the only target that can fill the field.
    for group, _kind, _slots in info.outer_slots.values():
        field_name = slot_index_field(group)
        sources[field_name] = (ast.Name(id=field_name, ctx=ast.Load()), info.var_sections.get(group, set()))

    plain: list[ast.stmt] = []
    by_section: dict[str, list[ast.stmt]] = {}
    for field_name in info.ctx_fields:
        if field_name == PL_IS_VALID_FIELD:
            continue
        expr, sections = sources.get(field_name, (None, set()))
        source = expr if expr is not None else ast.Name(id=field_name, ctx=ast.Load())
        if not sections:
            plain.append(_ctx_field_assign(field_name, copy.deepcopy(source)))
            continue
        for kind in sections:
            by_section.setdefault(kind, []).append(_ctx_field_assign(field_name, copy.deepcopy(source)))

    return plain + [_wrap_in_section(kind, body) for kind, body in sorted(by_section.items())]


def _build_stage_ctx_lookup(stage, depth: int) -> tuple[str, list[ast.stmt]]:
    """Return the ctx variable name and optional delayed ctx lookup statements."""
    delay = stage.delay
    ctx_var = _ctx_var_name(delay)
    if delay == 0:
        return ctx_var, []

    ctx_assign = ast.Assign(
        targets=[ast.Name(id=ctx_var, ctx=ast.Store())],
        value=ast.Subscript(
            value=ast.Name(id="_pl_ctx_arr", ctx=ast.Load()),
            slice=ast.BinOp(
                left=ast.BinOp(
                    left=ast.Name(id=PL_TASK_ID_FIELD, ctx=ast.Load()),
                    op=ast.Add(),
                    right=ast.Constant(value=depth - delay),
                ),
                op=ast.Mod(),
                right=ast.Constant(value=depth),
            ),
            ctx=ast.Load(),
        ),
        lineno=0,
    )
    return ctx_var, [ctx_assign]


def _build_stage_args(stage, arg_mapping: list, ctx_var: str, info: PipelineInfo) -> list[ast.expr]:
    """Build stage call args, replacing ctx-backed args with ctx.field references.

    A struct argument is replaced by the ctx slot itself: the slot carries the struct's
    fields under their own names, so a stage body reading `ri.ki` reads this beat's `ki`
    with no rewriting of the stage at all.
    """
    new_args = []
    for i, orig_arg in enumerate(stage.args):
        if i < len(arg_mapping) and arg_mapping[i] is not None:
            field_name = arg_mapping[i][0]  # (field_name, fill_expr) tuple
            if field_name == PL_STRUCT_ARG:
                new_args.append(ast.Name(id=ctx_var, ctx=ast.Load()))
                continue
            if isinstance(orig_arg, ast.Name) and orig_arg.id in info.outer_slots:
                # Re-select the slot with this stage's own beat index, so a delayed stage
                # gets the slot from the iteration whose data it is handling.
                group = info.outer_slots[orig_arg.id][0]
                new_args.append(
                    ast.Subscript(
                        value=ast.Name(id=group, ctx=ast.Load()),
                        slice=ast.Attribute(
                            value=ast.Name(id=ctx_var, ctx=ast.Load()),
                            attr=field_name,
                            ctx=ast.Load(),
                        ),
                        ctx=ast.Load(),
                    )
                )
                continue
            new_args.append(
                ast.Attribute(
                    value=ast.Name(id=ctx_var, ctx=ast.Load()),
                    attr=field_name,
                    ctx=ast.Load(),
                )
            )
        else:
            new_args.append(copy.deepcopy(orig_arg))
    return new_args


def _wrap_in_section(section_kind: str, body: list[ast.stmt]) -> ast.With:
    """Wrap statements in `with pl.section_<kind>():`."""
    section_call = ast.Call(
        func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr=f"section_{section_kind}", ctx=ast.Load()),
        args=[],
        keywords=[],
    )
    return ast.With(
        items=[ast.withitem(context_expr=section_call, optional_vars=None)],
        body=body,
        lineno=0,
    )


def _build_guarded_stage_body(stage, call_expr: ast.Call, ctx_var: str, sync: SyncPlan) -> list[ast.stmt]:
    """Build sync + stage-call statements that run under the ctx validity guard."""
    index_expr = ast.Attribute(
        value=ast.Name(id=ctx_var, ctx=ast.Load()),
        attr=PL_TASK_ID_FIELD,
        ctx=ast.Load(),
    )

    # Every cross-core sync for this stage comes from the graph: RAW and WAR,
    # same-buffer and address-reuse alike (see _sync_graph.plan_sync_sites).
    sites = _sites_for_stage(sync, stage)
    auto_pre, auto_post = _sync_stmts_for(sites, index_expr)

    # The user's own statements around the call are emitted verbatim. They are NOT
    # redirected to this stage's ctx slot: they execute on the current beat, so the current
    # value of every name they read is the honest one. Redirecting them would give one name
    # two meanings depending on where it appears — a ctx field inside these statements but
    # the live variable everywhere else — and would still not fix a value defined here and
    # passed to a delayed stage, since the definition runs after the snapshot either way.
    # Only the stage's own arguments read the ctx slot (see _build_stage_args).
    guarded_body = []
    guarded_body.extend(auto_pre)
    guarded_body.extend(copy.deepcopy(s) for s in stage.pre_stmts)
    guarded_body.append(ast.Expr(value=call_expr, lineno=0))
    guarded_body.extend(copy.deepcopy(s) for s in stage.post_stmts)
    guarded_body.extend(auto_post)
    return guarded_body


def _wrap_stage_section(stage, guarded_body: list[ast.stmt], ctx_var: str) -> ast.With:
    """Guard a stage body on the ctx validity flag, and wrap that in its pl.section_*."""
    guarded_call = ast.If(
        test=ast.Attribute(
            value=ast.Name(id=ctx_var, ctx=ast.Load()),
            attr=PL_IS_VALID_FIELD,
            ctx=ast.Load(),
        ),
        body=guarded_body,
        orelse=[],
        lineno=0,
    )
    section_attr = f"section_{stage.section_kind}"
    section_call = ast.Call(
        func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr=section_attr, ctx=ast.Load()),
        args=[],
        keywords=[],
    )
    return ast.With(
        items=[ast.withitem(context_expr=section_call, optional_vars=None)],
        body=[guarded_call],
        lineno=0,
    )


def _build_stage_call(
    stage, stage_idx: int, info: PipelineInfo, depth: int, sync: SyncPlan
) -> list[ast.stmt]:
    """Build a single stage call with delay, ctx lookup, and is_valid guard.

    Preserves pre_stmts (e.g. wait sync) and post_stmts (e.g. set sync)
    from the original section block around the stage call.
    """
    ctx_var, stmts = _build_stage_ctx_lookup(stage, depth)

    # Build the stage function call with args replaced
    arg_mapping = info.stage_arg_mapping[stage_idx]
    call_expr = ast.Call(
        func=ast.Name(id=stage.func_name, ctx=ast.Load()),
        args=_build_stage_args(stage, arg_mapping, ctx_var, info),
        keywords=[],
    )

    # sync + stage call + the user's own surrounding statements, all INSIDE the is_valid
    # guard: a fill or drain beat carries no task, so letting its wait/set run would
    # consume permits belonging to a later real one. The event-id index reads
    # `<ctx_var>._pl_task_id`, i.e. the task THIS stage is handling, not the beat number.
    guarded_body = _build_guarded_stage_body(stage, call_expr, ctx_var, sync)
    stmts.append(_wrap_stage_section(stage, guarded_body, ctx_var))

    return stmts
