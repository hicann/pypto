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

"""Checks on a kernel's pipeline structure, gathered in one place.

Checks used to sit wherever the data happened to be parsed — some inside scanning loops,
some mid-way through analyze_pipeline, some in the transformer. Adding one meant finding
the right scanning function and editing it. They are collected here instead, grouped by
what they need to already know:

``validate_structure(info)``  after the stage sequence, sections, loop info and ctx
                              fields are known, before any sync scanning
``validate_sync(...)``        after the sync graph exists (see _sync_graph)
``validate_names(...)``       once the caller knows which variables it will introduce —
                              both emission paths, each passing its own names

Format checks stay at their point of use: if a buffer's memory space or its id tuple
cannot be resolved, scanning cannot continue, so those raise where they are read.
"""

from __future__ import annotations

import ast

from ._astutil import call_name, get_funcdef


def buffer_users(graph) -> dict:
    """``{buffer: [(stage_idx, stage, section, roles), ...]}``, ordered by stage.

    One entry per stage that touches the buffer, with ``roles`` the set of op-level roles
    it uses there. This is what the sync checks below reason over, and it is also the
    right basis for deciding producer vs consumer: whichever stage comes FIRST produces
    the data, later ones consume it.

    Op-level roles cannot decide that on their own. A consumer may well write — an in-place
    ``pl.muls`` on the buffer shows up as a write yet the stage is still consuming what the
    previous stage left there. "Consumer" means "uses the buffer", not "only reads it".
    """
    by_buf: dict = {}
    for acc in graph.accesses:
        per_stage = by_buf.setdefault(acc.buffer, {})
        entry = per_stage.setdefault(acc.stage_idx, [acc.stage, acc.section, set()])
        entry[2].add(acc.role)
    return {
        buf: [(idx, *per_stage[idx]) for idx in sorted(per_stage)]
        for buf, per_stage in by_buf.items()
    }


def validate_sync(graph, info, cycles) -> None:
    """Checks that need the sync graph: regions, op-level accesses, users, edges.

    Called from _sync_graph once the graph is built.

    ``cycles`` is what ``find_cycles(graph)`` returned. It is passed in rather than
    computed here because the split is real: walking the graph is the graph's job, while
    deciding what a cycle's total distance MEANS is a check. Reaching back for the walk
    would also make this module import a sibling that imports it — the very cycle the
    lazy import used to paper over.
    """
    users = buffer_users(graph)
    _check_cross_core_users(info, users)
    _check_event_id_counts(graph, info)
    _check_reuse_mutex_ids(graph, info)
    _check_stable_distances(graph)
    _check_no_deadlock_cycle(cycles)


def _check_stable_distances(graph) -> None:
    """Every edge must sit at the same distance on every task.

    The whole plan rests on one distance per edge: it decides the skew, and with it how
    many permits pre-fire releases, how many waits drain consumes, and which task each
    ``% slot_count`` event id belongs to. ``build_graph`` expands two full rotations past
    the fill precisely so it has a second sample to compare against; an edge whose distance
    differs between them never settled, which means assumption A1 — each stage advances a
    buffer exactly once per iteration — does not hold for it.

    Emitting anyway would take whichever distance the expansion happened to see last and
    pair every wait with a set belonging to some other task. Reject instead: the flag is
    cheap to compute and there is nothing sound to emit once it is set.
    """
    unstable = [edge for edge in graph.edges if edge.unstable]
    if not unstable:
        return
    detail = "\n".join(
        f"  {edge.kind} {edge.src} -> {edge.dst}   "
        f"(dist={edge.dist:+d}, task_off={edge.task_off:+d}, {edge.slot_count} slots)"
        for edge in unstable
    )
    raise ValueError(
        "pipeline: these cross-core dependencies never reach a steady state — the distance "
        "between their two ends changes from task to task, so no single wait/set placement "
        "covers them:\n"
        f"{detail}\n"
        "This means a stage advances one of these buffers a different number of times per "
        "iteration than the slot timeline (task % slot_count) assumes. Take exactly one "
        "slot per buffer per stage, or give the buffer its own tile_group per handover."
    )


def _check_no_deadlock_cycle(cycles) -> None:
    """A dependency cycle whose distances sum to <= 0 cannot be synchronised at all.

    Every edge means "the destination waits for the source". Going around a cycle and
    ending up at the same task or a later one (total <= 0) asks each side to wait for the
    other, so no wait/set placement can satisfy it — the kernel would hang on hardware.

    A positive total is healthy: the cycle closes on an EARLIER task, which is just a
    cross-iteration pipeline. A single negative edge is also fine on its own — an
    inverse-time edge (wait placed before the matching set) has been verified to work on
    hardware. Only the total around the cycle decides.
    """
    deadlocks = [(path, total) for path, total in cycles if total <= 0]
    if not deadlocks:
        return
    routes = "\n".join(f"  {_format_cycle(path, total)}" for path, total in sorted(deadlocks, key=lambda c: c[1]))
    raise ValueError(
        "pipeline: the stage/buffer dependencies form a cycle that no cross-core sync can "
        "satisfy (its distances sum to <= 0, i.e. each side would wait for the other):\n"
        f"{routes}\n"
        "Break the cycle by reordering the stages, giving one of the buffers its own "
        "memory instead of sharing an address, or adding a slot so the dependency reaches "
        "back to an earlier task."
    )


def _format_cycle(path: list, total: int) -> str:
    """Render one cycle from find_cycles() as a readable route."""
    route = " -> ".join(f"{a.stage}[{a.buffer} {a.role} {a.pipe}]" for a in path)
    return f"{route}   (total distance={total})"


def _check_cross_core_users(info, users: dict) -> None:
    """A buffer declared with fwd/bwd ids must hand data from one core to another.

    Three ways that can fail to hold, none of them currently caught, and all of them
    producing sync that is at best useless:

      one user      nobody on the other side. Its sets pile up unanswered, and the
                    hardware faults once an id is set more than 15 times unmatched.
      same core     both users on cube, or both on vector. No core boundary is crossed,
                    so cross-core sync is not what orders them.
      3+ users      one variable is standing in for several producer/consumer pairs.
                    Ids are resolved per buffer, so two pairs would share one id group
                    and their differently-skewed edges would interleave. Declare a
                    separate tile_group per pair (the same `addrs` may be reused).
    """
    for buf_name, buf in info.sync.buffers.items():
        if buf.fwd_ids_node is None and buf.bwd_ids_node is None:
            continue  # not a cross-core buffer
        entries = users.get(buf_name, [])
        names = [e[1] for e in entries]
        if len(entries) < 2:
            raise ValueError(
                f"pipeline: cross-core buffer '{buf_name}' is used by {len(entries)} stage(s) "
                f"{names}, so there is no cross-core handover to synchronise. Either drop its "
                f"fwd_ids/bwd_ids or have a second stage use it."
            )
        if len(entries) > 2:
            raise ValueError(
                f"pipeline: cross-core buffer '{buf_name}' is used by {len(entries)} stages "
                f"{names}. One buffer variable carries one producer/consumer pair, since its "
                f"event ids are resolved per buffer. Declare one tile_group per pair — they "
                f"may share the same addrs and mutex_ids."
            )
        sections = {e[2] for e in entries}
        if len(sections) == 1:
            raise ValueError(
                f"pipeline: cross-core buffer '{buf_name}' is used only by "
                f"'{sections.pop()}' stages {names}. Cross-core sync orders work across the "
                f"cube/vector boundary; same-core ordering comes from auto_mutex instead."
            )


def _check_event_id_counts(graph, info) -> None:
    """A direction's event-id list must hold one id per slot, or exactly one id.

    Two counters run side by side and they are not the same thing: the slots turn over on
    ``task % slot_count``, while the id for a handover is picked with
    ``ids[task % id_count]``. Only two ratios keep them in step:

      id_count == slot_count   one id per slot; every slot hands over independently.
      id_count == 1            all slots share one id. Consecutive handovers then queue up
                               on it, so a slot cannot be released until the previous one
                               has been — parallelism is lost, correctness is not. This is
                               the intended way to cope with a shortage of ids, and it is
                               why the count is not simply required to equal the slots.

    Anything between (three slots on two ids, say) makes the two cycles slide against each
    other, so a wait pairs with the set of a different slot. Nothing downstream can repair
    it and it surfaces as an intermittent data race rather than a failure, which is exactly
    the shape that is worth refusing up front.

    Buffers no stage touches are skipped — _check_cross_core_users reports those instead.
    """
    for buf_name, buf in info.sync.buffers.items():
        region = graph.regions.get(buf_name)
        if region is None:
            continue
        slots = graph.slots[region]
        for label, id_count in (("fwd_ids", buf.fwd_id_count), ("bwd_ids", buf.bwd_id_count)):
            if id_count and id_count not in (1, slots):
                raise ValueError(
                    f"pipeline: cross-core buffer '{buf_name}' declares {id_count} {label} "
                    f"but rotates through {slots} slots. Give it one id per slot "
                    f"({slots} of them), or a single id shared by all slots — any other "
                    f"count pairs a wait with the set of a different slot."
                )


def _check_reuse_mutex_ids(graph, info) -> None:
    """Buffers sharing a region must share their mutex ids.

    A mutex locks an ADDRESS. Two buffers over the same memory holding different locks do
    not exclude each other at all, so the intra-core ordering auto_mutex is supposed to
    provide silently disappears — and it does not show up as a failure right away, which
    makes it the kind of race that surfaces much later.
    """
    by_region: dict = {}
    for buf_name, region in graph.regions.items():
        by_region.setdefault(region, []).append(buf_name)

    for region, members in sorted(by_region.items()):
        if len(members) < 2:
            continue
        ids_by_buf = {name: info.sync.mutex_ids.get(name) for name in sorted(members)}
        distinct = {tuple(v) for v in ids_by_buf.values() if v is not None}
        if len(distinct) > 1:
            detail = ", ".join(f"{n}={v}" for n, v in ids_by_buf.items())
            raise ValueError(
                f"pipeline: buffers sharing addresses must share mutex_ids, but region "
                f"{region} has {detail}. A mutex locks the address, so different ids over "
                f"one region give no mutual exclusion."
            )


def validate_names(func_def, framework_names: set) -> None:
    """The variables this transform is about to introduce must not already be in use.

    Called by BOTH emission paths, each passing its own names: what a path is allowed to
    collide over is exactly what that path emits. Checking one path's names while running
    the other reports on variables that will never be generated and misses the ones that
    will.

    Only fixed names are worth listing. Lifted id variables (``_pl_fwd_ids_<buffer>``) end
    in a user-chosen buffer name, so a collision needs the user to have declared that exact
    derived name — and the parser catches a genuine redefinition anyway.
    """
    _check_name_collisions(func_def, framework_names)


def _check_name_collisions(func_def, framework_names: set) -> None:
    """The names the transform introduces must not already be taken by the kernel."""
    user_names = {n.id for n in ast.walk(func_def) if isinstance(n, ast.Name)}
    clash = sorted(framework_names & user_names)
    if clash:
        raise ValueError(
            f"pipeline transform: framework variable name(s) {clash} collide with "
            f"user-defined names in the kernel. The '_pl_' prefix is reserved for "
            f"the pipeline transform — please rename the conflicting user variable(s)."
        )


def validate_structure(info, func_def=None, stage_func_names: set | None = None) -> None:
    """Checks that need only the parsed structure — no sync or schedule information.

    Called from analyze_pipeline once the stages, sections and loop info exist, and before
    anything is derived from them: none of the checks below reads the ctx layout or the
    buffer scan, so running first is what lets a malformed stage chain be reported as such
    instead of as whatever the later passes trip over.

    ``func_def`` and ``stage_func_names`` enable the two checks that have to look at the
    whole kernel rather than at the stage list: both catch a shape the transform would
    otherwise SKIP silently, which is worse than refusing it.
    """
    _check_stage_returns(info)
    _check_stage_sections(info)
    _check_alternating_sections(info)
    _check_loop_step(info)
    if func_def is not None and stage_func_names:
        _check_single_pipeline_loop(func_def, stage_func_names)
        _check_no_nested_stage(func_def, info, stage_func_names)


def _loops_holding_stages(func_def, stage_func_names: set) -> list:
    """Every for-loop whose OWN body calls a stage — i.e. every candidate pipeline loop.

    Matches how the stage list is collected (_extract_stages_from_loop looks at a loop's
    direct body only), so an enclosing loop of a pipelined one is not counted: its stages
    sit in the pipeline loop's body, not its own.
    """
    found = []
    for node in ast.walk(func_def):
        if not isinstance(node, ast.For):
            continue
        for stmt in node.body:
            calls = []
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                calls.append(stmt.value)
            elif isinstance(stmt, ast.With):
                calls.extend(
                    s.value for s in stmt.body if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                )
            if any(call_name(c) in stage_func_names for c in calls):
                found.append(node)
                break
    return found


def _check_single_pipeline_loop(func_def, stage_func_names: set) -> None:
    """Only one loop may drive the pipeline.

    The transform builds one ctx ring, one task counter and one drain for the whole kernel,
    so a second loop calling stages of its own would need a second set of all three. Today
    the search stops at the first such loop and the rest are left untouched — no ctx, no
    sync, no drain — which looks like it worked. Refuse instead.

    Nesting is fine and unaffected: an enclosing loop holds the pipelined loop in its body,
    not stage calls, so it is not one of these.
    """
    loops = _loops_holding_stages(func_def, stage_func_names)
    if len(loops) > 1:
        where = ", ".join(f"line {loop.lineno}" for loop in loops)
        raise ValueError(
            f"pipeline: {len(loops)} separate loops call pipeline stages ({where}), but the "
            f"transform pipelines exactly one loop — the others would be left without ctx, "
            f"sync or drain.\n"
            f"Put the stages of one pipeline in a single loop, or split the kernel so each "
            f"loop is its own @pl.jit function."
        )


def _check_no_nested_stage(func_def, info, stage_func_names: set) -> None:
    """A stage must not reach another stage, directly or through the functions it calls.

    Sub-stages need their accesses attributed to the sub-stage rather than the caller, and
    sync placed around the inner call — support for that was removed and is pending a
    redesign. Without it the inner stage reads as an ordinary helper call: its buffer
    accesses never reach the dependency graph, so its handovers go unsynchronised.

    The search follows plain helper calls as well, because a stage buried behind one is
    worse than a directly nested one, not better: the access scan walks helpers but skips
    stage names inside them, and the analyzer only lists the stages called in the pipeline
    loop, so such a stage contributes nothing at all — no accesses, no sync, no beat — and
    says nothing about it.
    """
    stage_defs = {
        node.name: node
        for node in ast.walk(func_def)
        if isinstance(node, ast.FunctionDef) and node.name in stage_func_names
    }
    # The stages the pipeline loop calls already carry their AST; take those for free.
    for stage in info.stages:
        if stage.func_def is not None:
            stage_defs.setdefault(stage.func_name, stage.func_def)
    # A stage decorated but never called by the loop is not among them, and a stage's body
    # usually lives outside the kernel, so the rest are resolved through closure_vars.
    for name in stage_func_names:
        if name in stage_defs:
            continue
        resolved = _try_get_funcdef_for(info, name)
        if resolved is not None:
            stage_defs[name] = resolved

    def visit(caller_def, path: tuple) -> None:
        for node in ast.walk(caller_def):
            if not isinstance(node, ast.Call):
                continue
            callee = call_name(node)
            if not callee or callee in path:
                continue
            if callee in stage_func_names:
                chain = " -> ".join(path + (callee,))
                where = f" via {' -> '.join(path[1:])}," if len(path) > 1 else ""
                raise ValueError(
                    f"pipeline: stage '{path[0]}' reaches stage '{callee}'{where} at line "
                    f"{node.lineno} ({chain}). A stage may not contain another stage — the "
                    f"inner one's buffer accesses would be attributed to the caller and its "
                    f"handovers left unsynchronised.\n"
                    f"Inline '{callee}' into '{path[0]}', or make it a plain helper function "
                    f"(drop @pl.pipeline.stage) if it needs no sync of its own."
                )
            helper_def = _try_get_funcdef_for(info, callee)
            if helper_def is not None:
                visit(helper_def, path + (callee,))

    for outer_name, outer_def in stage_defs.items():
        visit(outer_def, (outer_name,))


def _try_get_funcdef_for(info, name: str):
    """The AST of a stage defined outside the kernel, via the analyzer's closure_vars."""
    return get_funcdef(info.closure_vars.get(name))


def _check_stage_returns(info) -> None:
    """A stage must not return a value: the transform calls it for its side effects and
    has nowhere to put a result."""
    for stage in info.stages:
        func_def = stage.func_def
        if func_def is None:
            continue
        for node in ast.walk(func_def):
            if isinstance(node, ast.Return) and node.value is not None:
                raise ValueError(
                    f"pipeline: stage '{stage.func_name}' returns a value. Stages are "
                    f"called for their effect on buffers; use an output buffer instead."
                )


def _check_stage_sections(info) -> None:
    """Every stage call must sit inside a section block, since that says which core runs it."""
    for stage in info.stages:
        if stage.section_kind == "":
            raise ValueError(
                f"pipeline: stage call '{stage.func_name}' appears directly in the pipeline "
                f"loop body, not inside a `with pl.section_cube()/section_vector()` block. "
                f"Each stage call must be wrapped in a section block."
            )


def _check_alternating_sections(info) -> None:
    """The stage chain has to alternate cube/vector.

    Two consecutive stages on one core would advance that core's delay by the preload
    amount twice in a row, which the delay model has no way to express.
    """
    stages = info.stages
    for i in range(len(stages) - 1):
        cur, nxt = stages[i], stages[i + 1]
        if cur.section_kind == nxt.section_kind:
            raise ValueError(
                f"pipeline: stages '{cur.func_name}' and '{nxt.func_name}' are both "
                f"on the '{cur.section_kind}' core (consecutive same-core stages). The "
                f"delay model requires the stage chain to strictly alternate "
                f"cube/vector (C->V->C->V...)."
            )


def _check_loop_step(info) -> None:
    """The pipeline loop must count upwards.

    Guards that reason about a task some iterations away compute `var + n * step` and
    compare with `<` against the bound; a negative step would need the opposite
    comparison throughout. Rejected rather than half-supported.
    """
    step = info.pipeline_loop_step
    if step is None:
        return  # implicit 1
    if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
        raise ValueError(
            f"pipeline: pipeline loop `for {info.pipeline_loop_var} in ...` steps backwards "
            f"({ast.unparse(step)}). Only forward iteration is supported."
        )
    if isinstance(step, ast.Constant) and isinstance(step.value, int) and step.value <= 0:
        raise ValueError(
            f"pipeline: pipeline loop `for {info.pipeline_loop_var} in ...` has step "
            f"{step.value}. The step must be a positive value."
        )
