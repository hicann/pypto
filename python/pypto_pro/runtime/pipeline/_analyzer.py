# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Analyzer: extract pipeline structure from serial kernel AST."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field

from pypto_pro.language.parser._control_flow_parser import validate_single_tail_return

from ._cross_core_scanner import AccessRole, CrossCoreSyncContext
from ._stage import is_pipeline_stage


@dataclass
class StageCall:
    """Info about a single stage call in the serial loop body."""

    func_name: str  # e.g. "compute_qk"
    section_kind: str  # "cube" or "vector" (for mix: inferred from last sub-stage)
    args: list  # list of ast.expr nodes (call arguments)
    delay: int  # derived from call order: 0, 1, 2, ...
    pre_stmts: list = field(default_factory=list)  # statements before stage call in same section
    post_stmts: list = field(default_factory=list)  # statements after stage call in same section
    cross_access: list = field(default_factory=list)  # list[CrossCoreAccess] for this stage
    # Local (non-cross-core) buffers that address-overlap a cross-core buffer:
    # {buffer_name: (first_pipe, last_pipe)}. Used for scenario 2/3 reverse sync.
    local_access: dict = field(default_factory=dict)
    is_mix: bool = False  # True if this stage's body calls other @stage functions
    sub_stages: list = field(default_factory=list)  # for mix: list of sub-stage func_names in order
    inner_buffers: set = field(default_factory=set)  # for mix: buffer names internal (both W+R inside)
    outer_accesses: list = field(default_factory=list)  # for mix: CrossCoreAccess list for outer buffers


@dataclass
class PipelineInfo:
    """Extracted pipeline structure from the serial kernel."""

    stages: list[StageCall] = field(default_factory=list)
    ctx_fields: list[str] = field(default_factory=list)
    # Maps: stage arg position -> (field_name, fill_expr) | None
    stage_arg_mapping: list[list] = field(default_factory=list)
    inner_loop_var: str = ""  # inner loop variable (e.g. "ki")
    inner_loop_range_end: ast.expr | None = None  # e.g. ast node for "skv_tiles"
    pre_loop_stmts: list = field(default_factory=list)
    # Variables that change across iterations (ctx field candidates)
    loop_changing_vars: set = field(default_factory=set)
    # Mid-loop assignments: name -> rhs AST for changing variables assigned inside
    # the pipeline loop body (e.g. p_offset = ki). Used by _derive_ctx_fields to
    # generate ctx fill with the rhs expression instead of the variable name.
    mid_loop_assigns: dict = field(default_factory=dict)
    # Cross-core sync context (buffers, memory, lifted ids, inner consumer map)
    sync: CrossCoreSyncContext = field(default_factory=CrossCoreSyncContext)
    # Synced mix function defs (name -> ast.FunctionDef with inner sync inserted),
    # used to flatten (inline) the mix body into the main loop at the call site.
    # Populated lazily on first flatten; acts as a cache.
    mix_synced_funcs: dict = field(default_factory=dict)
    # Reference to closure_vars for mix-body flatten (avoids threading it through
    # every call in the transform chain).
    closure_vars: dict = field(default_factory=dict)
    # Reference to the kernel function AST (for kernel-body slot scanning at
    # transform time, e.g. slots taken outside a mix stage).
    func_def: ast.FunctionDef | None = None


def analyze_pipeline(func_def: ast.FunctionDef, closure_vars: dict) -> PipelineInfo:
    """Analyze a serial kernel function AST to extract pipeline structure.

    Looks for the innermost for-loop that contains stage calls inside
    section blocks, and extracts stage ordering, ctx fields, etc.

    Args:
        func_def: The kernel function's AST node
        closure_vars: Closure variables (module-level constants, stage functions)

    Returns:
        PipelineInfo with all extracted information
    """
    info = PipelineInfo()
    info.closure_vars = closure_vars
    info.func_def = func_def

    # Find all stage functions.
    stage_func_names = set()
    for name, val in closure_vars.items():
        if is_pipeline_stage(val):
            stage_func_names.add(name)
            stage_func_def = _try_get_funcdef(val)
            if stage_func_def is not None:
                _check_stage_void_return_only(name, stage_func_def)

    # Walk the AST to find the main loop structure.
    # Seed with ALL loop variables (incl. outer loops) for changing-var analysis.
    all_loop_vars = _collect_loop_vars(func_def.body)
    _find_pipeline_loop(func_def.body, info, stage_func_names, closure_vars, all_loop_vars)

    # L3/L5: pipeline enabled but no usable stages found.
    if not info.stages:
        if not stage_func_names:
            raise ValueError(
                "pipeline: no @pl.pipeline.stage functions found, but pipeline=... was "
                "set on the kernel. Decorate your stage functions with @pl.pipeline.stage."
            )
        raise ValueError(
            "pipeline: no pipeline loop found — @pl.pipeline.stage functions exist but "
            "none are called inside a for-loop's `with pl.section_*()` blocks. The "
            "pipeline loop must contain stage calls wrapped in section blocks."
        )

    # Identify mix stages (body calls other @stage functions) and infer section_kind
    _identify_mix_stages(info, stage_func_names, closure_vars)

    # L1 (deferred): non-mix stages with empty section_kind = bare call without section
    for stage in info.stages:
        if not stage.is_mix and stage.section_kind == "":
            raise ValueError(
                f"pipeline: stage call '{stage.func_name}' appears directly in the pipeline "
                f"loop body, not inside a `with pl.section_cube()/section_vector()` block. "
                f"Each non-mix stage call must be wrapped in a section block."
            )

    # C5: stage chain must strictly alternate cube/vector
    _check_alternating(info)

    # Derive ctx fields from stage arguments
    _derive_ctx_fields(info, closure_vars)

    # Scan cross-core buffer accesses for auto-sync
    _scan_cross_core(info, func_def, closure_vars)

    # C7: each cross-core buffer must have at most one producer and one consumer
    _check_single_producer_consumer(info)

    return info


def _check_stage_void_return_only(func_name: str, func_def: ast.FunctionDef) -> None:
    """Reject value-returning or early-returning @pl.pipeline.stage functions.

    Pipeline stage calls are transformed as statement calls with delayed ctx
    arguments, so there is no caller-side value target. Bare `return` is allowed
    only as one top-level final statement.
    """
    return_error = validate_single_tail_return(func_def, f"@pl.pipeline.stage function '{func_name}'")
    if return_error is not None:
        _, message, hint = return_error
        raise ValueError(f"pipeline: {message} {hint}")

    if func_def.returns is not None and not (
        isinstance(func_def.returns, ast.Constant) and func_def.returns.value is None
    ):
        raise ValueError(
            f"pipeline: @pl.pipeline.stage function '{func_name}' only supports "
            "a None return annotation; returning values is not supported. Hint: "
            "Do not write `return <value>`; only use `return` or `return None`. "
            "Pass output Tensor/Tile/buffer parameters for data results."
        )
    for node in ast.walk(func_def):
        if not isinstance(node, ast.Return):
            continue
        if node.value is None:
            continue
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            continue
        raise ValueError(
            f"pipeline: @pl.pipeline.stage function '{func_name}' only supports "
            "bare return or return None; returning values is not supported. "
            "Hint: Do not write `return <value>`; only use `return` or `return None`. "
            "Pass output Tensor/Tile/buffer parameters for data results."
        )


def _scan_cross_core(info: PipelineInfo, func_def: ast.FunctionDef, closure_vars: dict):
    """Scan cross-core buffers + each stage's accesses, store into info."""
    from ._cross_core_scanner import (
        detect_addr_overlaps,
        scan_all_buffer_memory,
        scan_buffer_addr_ranges,
        scan_cross_core_buffers,
        scan_kernel_slot_to_buffer,
        scan_stage_accesses,
    )

    cross_buffers, lifted_ids = scan_cross_core_buffers(func_def, closure_vars)
    info.sync.buffers = cross_buffers
    info.sync.lifted_ids = lifted_ids
    if not cross_buffers:
        return

    all_mem = scan_all_buffer_memory(func_def)
    info.sync.all_memory = all_mem

    # Detect address overlaps involving cross-core buffers (for auto-sync of
    # address-reused buffers). Local-local overlaps are ignored.
    info.sync.addr_ranges = scan_buffer_addr_ranges(func_def, closure_vars)
    info.sync.addr_overlaps = detect_addr_overlaps(info.sync.addr_ranges, set(cross_buffers.keys()))

    vf_func_defs = _collect_vf_func_defs(info, closure_vars)

    # Slots taken from cross-core buffers in the kernel body (pipeline loop), for
    # stages that receive a pre-taken slot instead of the buffer group itself.
    kernel_slot_to_buffer = scan_kernel_slot_to_buffer(func_def, cross_buffers)

    # All @stage function defs (name -> FunctionDef), for recursive sub-stage descent.
    stage_func_defs = {}
    for s in info.stages:
        sfd = _try_get_funcdef(closure_vars.get(s.func_name))
        if sfd is not None:
            stage_func_defs[s.func_name] = sfd
    # Also include sub-stages (not top-level in info.stages) referenced by mix stages.
    for s in info.stages:
        for sub_name in s.sub_stages:
            if sub_name not in stage_func_defs:
                sfd = _try_get_funcdef(closure_vars.get(sub_name))
                if sfd is not None:
                    stage_func_defs[sub_name] = sfd

    overlaps = info.sync.addr_overlaps
    for stage in info.stages:
        if stage.is_mix:
            _scan_mix_stage(
                stage, info, closure_vars, cross_buffers, vf_func_defs, all_mem, kernel_slot_to_buffer, stage_func_defs
            )
        else:
            fd = stage_func_defs.get(stage.func_name)
            if fd is not None:
                stage.cross_access = []
                stage.local_access = {}
                scan_stage_accesses(
                    fd,
                    cross_buffers,
                    vf_func_defs,
                    all_mem,
                    overlaps,
                    stage.cross_access,
                    stage.local_access,
                    call_args=stage.args,
                    caller_bindings={},
                    kernel_slot_map=kernel_slot_to_buffer,
                    stage_func_defs=stage_func_defs,
                )

    # Scenario 2/3: build reverse syncs for address-overlapping buffers in different stages
    if overlaps:
        _build_overlap_reverse_syncs(info, closure_vars)


def _build_overlap_reverse_syncs(info: PipelineInfo, closure_vars: dict):
    """Identify scenario 2/3 overlap pairs and build reverse sync descriptors.

    Scenario 2/3: two address-overlapping buffers used in DIFFERENT stages need a
    backward sync so the earlier-user cannot overwrite the shared region before the
    later-user finishes. Given a pair (A, B) where A is used before B:
      - wait: at A's earliest-use stage, using A's first-op pipe there
      - set:  at B's latest-use stage,  using B's last-op pipe there

    For a cross-core buffer, earliest use = producer stage (W), latest = consumer
    stage (R). For a local buffer, they are simply the first/last stage it appears
    in (its ops there are read/written without a producer/consumer split).
    """
    # buffer_name -> usage span: (earliest_idx, first_pipe, latest_idx, last_pipe)
    usage = _build_buffer_usage(info)

    reverse_sync_pairs = []
    for buf_a, buf_b in info.sync.addr_overlaps:
        ua = usage.get(buf_a)
        ub = usage.get(buf_b)
        if ua is None or ub is None:
            continue

        # Order the pair by earliest use: `first` is used before `last`.
        if ua[0] <= ub[0]:
            (first_idx, wait_pipe, _, _), first_stage = ua, info.stages[ua[0]]
            (_, _, last_idx, set_pipe), last_stage = ub, info.stages[ub[2]]
        else:
            (first_idx, wait_pipe, _, _), first_stage = ub, info.stages[ub[0]]
            (_, _, last_idx, set_pipe), last_stage = ua, info.stages[ua[2]]

        # Same stage on both ends → scenario 1 (handled by _extend_last_pipe)
        if first_idx == last_idx:
            continue
        if wait_pipe is None or set_pipe is None:
            continue

        # slot_count: both sides have the same count (validated earlier)
        slot_count = len(info.sync.addr_ranges[buf_a][1])

        reverse_sync_pairs.append(
            {
                "first_stage": first_stage.func_name,
                "last_stage": last_stage.func_name,
                "wait_pipe": wait_pipe,
                "set_pipe": set_pipe,
                "set_section_kind": last_stage.section_kind,
                "slot_count": slot_count,
                "stage_gap": last_idx - first_idx,
            }
        )

    # Step 2: allocate event ids for each reverse sync
    _allocate_overlap_event_ids(info, reverse_sync_pairs)

    # Step 3: lift event ids to variables and build OverlapReverseSync dataclass objects
    from ._cross_core_scanner import OverlapReverseSync

    syncs = []
    for i, p in enumerate(reverse_sync_pairs):
        var_name = f"_pl_overlap_ids_{i}"
        literal_node = ast.List(elts=[ast.Constant(value=v) for v in p["event_ids"]], ctx=ast.Load())
        info.sync.lifted_ids.append((var_name, literal_node))
        syncs.append(
            OverlapReverseSync(
                first_stage=p["first_stage"],
                last_stage=p["last_stage"],
                wait_pipe=p["wait_pipe"],
                set_pipe=p["set_pipe"],
                set_section_kind=p["set_section_kind"],
                slot_count=p["slot_count"],
                stage_gap=p["stage_gap"],
                event_ids=p["event_ids"],
                event_ids_var=var_name,
            )
        )
    info.sync.overlap_reverse_syncs = syncs


def _build_buffer_usage(info: PipelineInfo) -> dict:
    """Map each buffer touched by a stage to its usage span across stages:
        buffer_name -> (earliest_idx, first_pipe, latest_idx, last_pipe)
    where first_pipe is the pipe of its first op at the earliest stage, and
    last_pipe is the pipe of its last op at the latest stage.

    Cross-core buffers report pipes from cross_access; local overlapping buffers
    from local_access. A buffer appearing in multiple stages spans earliest→latest.
    """
    usage: dict[str, tuple[int, str, int, str]] = {}
    for idx, stage in enumerate(info.stages):
        # cross-core accesses: (buffer_name, first_pipe, last_pipe)
        touched = [(acc.buffer_name, acc.first_pipe, acc.last_pipe) for acc in stage.cross_access]
        # local overlapping accesses: buffer_name -> (first_pipe, last_pipe)
        touched += [(name, fp, lp) for name, (fp, lp) in stage.local_access.items()]

        for name, first_pipe, last_pipe in touched:
            existing = usage.get(name)
            if existing is None:
                usage[name] = (idx, first_pipe, idx, last_pipe)
            else:
                e_idx, e_fp, l_idx, l_lp = existing
                if idx < e_idx:
                    e_idx, e_fp = idx, first_pipe
                if idx > l_idx:
                    l_idx, l_lp = idx, last_pipe
                usage[name] = (e_idx, e_fp, l_idx, l_lp)
    return usage


def _collect_used_event_ids(info: PipelineInfo) -> set:
    """Collect all event ids already used by cross-core buffers' fwd/bwd ids.

    The real literal id lists live in info.sync.lifted_ids (fwd/bwd nodes on the
    buffers were replaced with variable names)."""
    used = set()
    for _var, node in info.sync.lifted_ids:
        if isinstance(node, (ast.List, ast.Tuple)):
            for e in node.elts:
                if isinstance(e, ast.Constant) and isinstance(e.value, int):
                    used.add(e.value)
    return used


def _allocate_overlap_event_ids(info: PipelineInfo, reverse_sync_pairs: list):
    """Allocate event ids (0-15) for each reverse sync, filling each pair's
    'event_ids' key with a list of ints.

    Strategy:
      1. Each pair needs slot_count ids by default.
      2. Allocate from unused ids (0-15) in order.
      3. If not enough: degrade pairs (share 1 id across all slots) from smallest
         slot_count up, logging a warning each time.
      4. If all degraded to 1 id each and still not enough: raise.
    """
    if not reverse_sync_pairs:
        return

    max_event_id = 16
    used = _collect_used_event_ids(info)
    free = [i for i in range(max_event_id) if i not in used]

    # How many ids each pair wants (default: slot_count). May be degraded to 1.
    wants = [p["slot_count"] for p in reverse_sync_pairs]

    # Degrade (smallest slot_count first) until total demand fits in free ids.
    # Sort indices by slot_count ascending for degradation order.
    order = sorted(range(len(reverse_sync_pairs)), key=lambda i: wants[i])
    deg_ptr = 0
    while sum(wants) > len(free) and deg_ptr < len(order):
        idx = order[deg_ptr]
        if wants[idx] > 1:
            import logging

            logging.warning(
                f"pipeline: not enough event ids for address-overlap reverse sync; "
                f"degrading pair ({reverse_sync_pairs[idx]['first_stage']} -> "
                f"{reverse_sync_pairs[idx]['last_stage']}) from {wants[idx]} ids to 1 "
                f"(slots will serialize, correctness preserved)."
            )
            wants[idx] = 1
        deg_ptr += 1

    if sum(wants) > len(free):
        raise ValueError(
            f"pipeline: not enough free event ids (0-{max_event_id - 1}) for "
            f"address-overlap reverse syncs. Need {sum(wants)}, have {len(free)} free "
            f"(used: {sorted(used)}). Reduce cross-core buffer id usage or overlaps."
        )

    # Allocate from free ids in order
    cursor = 0
    for p, n in zip(reverse_sync_pairs, wants):
        p["event_ids"] = free[cursor:cursor + n]
        cursor += n


def _collect_vf_func_defs(info: PipelineInfo, closure_vars: dict) -> dict[str, ast.FunctionDef]:
    """Collect VF helper func defs.

    Includes both @pl.vector_function decorated functions AND plain wrapper
    functions that transitively call a @pl.vector_function (any depth of
    indirection). This ensures the cross-core scanner can see through chains
    like ``stage a → b → c → @pl.vector_function d``.
    """
    stage_names = {s.func_name for s in info.stages}
    vf_func_defs: dict[str, ast.FunctionDef] = {}
    all_func_defs: dict[str, ast.FunctionDef] = {}
    for name, val in closure_vars.items():
        if callable(val) and name not in stage_names and not is_pipeline_stage(val):
            fd = _try_get_funcdef(val)
            if fd is not None:
                all_func_defs[name] = fd
                if _is_vf_function(fd):
                    vf_func_defs[name] = fd
    # Iteratively include plain functions that call any already-known VF function,
    # repeating until no new additions (transitive closure).
    changed = True
    while changed:
        changed = False
        for name, fd in all_func_defs.items():
            if name in vf_func_defs:
                continue
            if _calls_any_vf_function(fd, vf_func_defs):
                vf_func_defs[name] = fd
                changed = True
    return vf_func_defs


def _scan_mix_stage(
    stage,
    info: PipelineInfo,
    closure_vars: dict,
    cross_buffers: dict,
    vf_func_defs: dict,
    all_mem: dict,
    kernel_slot_to_buffer: dict,
    stage_func_defs: dict,
):
    """Scan a mix stage's sub-stages, classify buffers as inner/outer.

    Uses the binding mechanism so cross-core buffers are tracked regardless of
    formal parameter names, and through any nesting / slot / alias forms:
      - mix_bindings: resolved from the mix stage's own call site (main loop)
      - each sub-stage's bindings: resolved from the sub-stage call inside the
        mix body, using mix_bindings for pass-through params.
    """
    from ._cross_core_scanner import build_binding_map, scan_stage_accesses

    mix_fn = closure_vars.get(stage.func_name)
    mix_fd = _try_get_funcdef(mix_fn)

    # Bindings for the mix stage itself, from its main-loop call site.
    mix_bindings = {}
    if mix_fd is not None:
        mix_bindings = build_binding_map(mix_fd, stage.args, {}, kernel_slot_to_buffer, cross_buffers)

    sub_accesses = {}
    for sub_name in stage.sub_stages:
        sub_fd = stage_func_defs.get(sub_name)
        if sub_fd is None:
            continue
        # Find this sub-stage's call node inside the mix body to get its actual args.
        sub_call_args = _find_sub_stage_call_args(mix_fd, sub_name) if mix_fd else None
        acc_list = []
        scan_stage_accesses(
            sub_fd,
            cross_buffers,
            vf_func_defs,
            all_mem,
            cross_access_out=acc_list,
            call_args=sub_call_args,
            caller_bindings=mix_bindings,
            kernel_slot_map=kernel_slot_to_buffer,
            stage_func_defs=stage_func_defs,
        )
        sub_accesses[sub_name] = acc_list

    # Tag each access with the section_kind of the sub-stage it happens in, so
    # outer-buffer sync (set/wait around the mix call) can be wrapped in the
    # right section (the bare mix call has no outer section of its own).
    if mix_fd is not None:
        for sub_name, accesses in sub_accesses.items():
            sec = _find_sub_stage_section_in_body(mix_fd, sub_name)
            if sec is not None:
                for acc in accesses:
                    acc.section_kind = sec

    # Classify: inner (both W+R within mix) vs outer (one side outside)
    buf_roles: dict[str, set] = {}
    buf_access_map: dict[str, list] = {}
    for accesses in sub_accesses.values():
        for acc in accesses:
            buf_roles.setdefault(acc.buffer_name, set()).add(acc.role)
            buf_access_map.setdefault(acc.buffer_name, []).append(acc)

    for buf_name, roles in buf_roles.items():
        if AccessRole.WRITE in roles and AccessRole.READ in roles:
            stage.inner_buffers.add(buf_name)
            _record_inner_consumer(buf_name, buf_access_map.get(buf_name, []), sub_accesses, stage, closure_vars, info)
        else:
            stage.outer_accesses.extend(buf_access_map.get(buf_name, []))
    stage.cross_access = stage.outer_accesses


def _record_inner_consumer(
    buf_name: str, accesses: list, sub_accesses: dict, stage, closure_vars: dict, info: PipelineInfo
):
    """Record inner consumer info (section_kind, pipe) for pre-fire.

    Only buffers with backward ids need pre-fire, so skip recording if the buffer
    has no bwd_ids (its consumer info would never be consumed by _build_prefire).
    """
    buf = info.sync.buffers.get(buf_name)
    if buf is None or buf.bwd_ids_node is None or buf.bwd_slot_count <= 0:
        return
    for acc in accesses:
        if acc.role == AccessRole.READ:
            mix_fn = closure_vars.get(stage.func_name)
            mix_fd = _try_get_funcdef(mix_fn)
            consumer_section = None
            if mix_fd is not None:
                for sn, accs in sub_accesses.items():
                    if acc in accs:
                        consumer_section = _find_sub_stage_section_in_body(mix_fd, sn)
                        break
            if consumer_section is None:
                consumer_section = "vector"
            info.sync.inner_consumer_map[buf_name] = (consumer_section, acc.last_pipe)
            break


def _try_get_funcdef(fn) -> ast.FunctionDef | None:
    """Get the ast.FunctionDef for a Python function object, or None."""
    import inspect
    import textwrap

    if fn is None:
        return None
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        mod = ast.parse(src)
        for node in mod.body:
            if isinstance(node, ast.FunctionDef):
                return node
    except (OSError, TypeError, SyntaxError):
        return None
    return None


def _record_pipeline_loop_info(stmt: ast.For, stmts: list[ast.stmt], info: PipelineInfo, all_loop_vars: set) -> None:
    """Record loop metadata after the pipeline loop has been found."""
    # L7: loop variable must be a simple Name
    if not isinstance(stmt.target, ast.Name):
        raise ValueError(
            "pipeline: the pipeline loop variable must be a simple name "
            "(e.g. `for ki in pl.range(...)`); tuple unpacking is not supported."
        )
    info.inner_loop_var = stmt.target.id
    # L6: loop must be pl.range(...) with extractable end bound
    info.inner_loop_range_end = None
    if isinstance(stmt.iter, ast.Call):
        args = stmt.iter.args
        if len(args) >= 2:
            info.inner_loop_range_end = args[1]
        elif len(args) == 1:
            info.inner_loop_range_end = args[0]
    if info.inner_loop_range_end is None:
        raise ValueError(
            f"pipeline: pipeline loop `for {info.inner_loop_var} in ...` must iterate "
            f"over pl.range(start, end[, step]) so the end bound can be extracted "
            f"for the is_valid guard; got an unsupported loop iterable."
        )
    info.loop_changing_vars = _collect_changing_vars(stmts, all_loop_vars)
    # Collect mid-loop assignments (name -> rhs AST) from the pipeline loop body.
    # These are used by _derive_ctx_fields: if a changing arg is a mid-loop
    # assigned variable, ctx fill uses its rhs expression (which only depends on
    # loop vars/constants available at loop start) instead of the variable name.
    info.mid_loop_assigns = _collect_mid_loop_assigns(stmt.body)


def _collect_mid_loop_assigns(stmts: list[ast.stmt]) -> dict[str, ast.expr]:
    """Collect mid-loop scalar assignments in the pipeline loop body: name -> rhs AST.

    Records every simple `name = expr` assignment (single Name target, excluding
    method-call results like `slot = buf.next()`). Self-referencing accumulators
    (e.g. `tick = tick + 1`) are included — the caller uses the `changing` set and
    self-reference check to decide how to handle each entry.

    Only the LAST assignment to a given name is kept (matching runtime semantics).
    Recurses into section blocks and if/else branches (not into nested for-loops,
    which are not part of the pipeline loop body proper).
    """
    result: dict[str, ast.expr] = {}
    for stmt in stmts:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            # Exclude method-call results (e.g. cur_k = k_l1_db.next())
            if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute):
                continue
            result[target.id] = stmt.value
        elif isinstance(stmt, ast.With):
            result.update(_collect_mid_loop_assigns(stmt.body))
        elif isinstance(stmt, ast.If):
            result.update(_collect_mid_loop_assigns(stmt.body))
            result.update(_collect_mid_loop_assigns(stmt.orelse))
    return result


def _nested_search_body(stmt: ast.stmt) -> list[ast.stmt] | None:
    """Return the nested body that can contain a pipeline loop."""
    if isinstance(stmt, (ast.For, ast.With)):
        return stmt.body
    return None


def _find_pipeline_loop(
    stmts: list[ast.stmt], info: PipelineInfo, stage_func_names: set, closure_vars: dict, all_loop_vars: set
):
    """Recursively find the innermost for-loop containing stage calls."""
    for stmt in stmts:
        if isinstance(stmt, ast.For):
            stages_found = _extract_stages_from_loop(stmt, info, stage_func_names)
            if stages_found:
                _record_pipeline_loop_info(stmt, stmts, info, all_loop_vars)
                return True
        nested_body = _nested_search_body(stmt)
        if nested_body is not None and _find_pipeline_loop(
            nested_body, info, stage_func_names, closure_vars, all_loop_vars
        ):
            return True
    return False


def _split_stage_section_body(body: list[ast.stmt], stage_func_names: set):
    """Find one stage call in a section body and split pre/post statements."""
    pre_stmts = []
    post_stmts = []
    stage_call = None
    stage_func_name = None

    for body_stmt in body:
        if stage_call is None:
            if isinstance(body_stmt, ast.Expr) and isinstance(body_stmt.value, ast.Call):
                func_name = _get_call_func_name(body_stmt.value)
                if func_name in stage_func_names:
                    stage_call = body_stmt.value
                    stage_func_name = func_name
                    continue
            pre_stmts.append(body_stmt)
        else:
            post_stmts.append(body_stmt)
    return stage_func_name, stage_call, pre_stmts, post_stmts


def _extract_stages_from_loop(for_stmt: ast.For, info: PipelineInfo, stage_func_names: set) -> bool:
    """Extract stage calls from a for-loop body (expecting interleaved sections)."""
    found_any = False

    for stmt in for_stmt.body:
        # Bare stage call: could be a mix stage (section_kind inferred later)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            fn = _get_call_func_name(stmt.value)
            if fn in stage_func_names:
                info.stages.append(
                    StageCall(
                        func_name=fn,
                        section_kind="",
                        args=list(stmt.value.args),
                        delay=0,
                    )
                )
                found_any = True
                continue
        if isinstance(stmt, ast.With):
            section_kind = _get_section_kind(stmt)
            if section_kind is None:
                _check_unsupported_section(stmt, stage_func_names)
                continue
            stage_call_info = _extract_stage_from_section(stmt, section_kind, stage_func_names)
            if stage_call_info is not None:
                info.stages.append(stage_call_info)
                found_any = True

    return found_any


def _check_unsupported_section(stmt: ast.With, stage_func_names: set):
    """C1: raise if an unsupported section type contains a stage call."""
    for body_stmt in stmt.body:
        if (
            isinstance(body_stmt, ast.Expr)
            and isinstance(body_stmt.value, ast.Call)
            and _get_call_func_name(body_stmt.value) in stage_func_names
        ):
            raise ValueError(
                f"pipeline: stage call "
                f"'{_get_call_func_name(body_stmt.value)}' is inside an "
                f"unsupported `with` block. Stage calls must be wrapped in "
                f"`with pl.section_cube()` or `with pl.section_vector()`."
            )


def _extract_stage_from_section(stmt: ast.With, section_kind: str, stage_func_names: set) -> StageCall | None:
    """Extract a stage call from a section block, with L2 validation."""
    stage_func_name, stage_call, pre_stmts, post_stmts = _split_stage_section_body(stmt.body, stage_func_names)
    if stage_call is None:
        return None
    # L2: check for a second stage call in same section
    for body_stmt in post_stmts:
        if (
            isinstance(body_stmt, ast.Expr)
            and isinstance(body_stmt.value, ast.Call)
            and _get_call_func_name(body_stmt.value) in stage_func_names
        ):
            raise ValueError(
                f"pipeline: section block contains multiple stage calls "
                f"('{stage_func_name}' and "
                f"'{_get_call_func_name(body_stmt.value)}'). Each "
                f"`with pl.section_*()` block must contain exactly one stage call."
            )
    return StageCall(
        func_name=stage_func_name,
        section_kind=section_kind,
        args=stage_call.args,
        delay=0,
        pre_stmts=pre_stmts,
        post_stmts=post_stmts,
    )


def _get_section_kind(with_stmt: ast.With) -> str | None:
    """Extract section kind ('cube' or 'vector') from a with statement."""
    if not with_stmt.items:
        return None
    ctx = with_stmt.items[0].context_expr
    if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute):
        if ctx.func.attr == "section_cube":
            return "cube"
        elif ctx.func.attr == "section_vector":
            return "vector"
    return None


def _is_vf_function(func_def: ast.FunctionDef) -> bool:
    """Check if a function is a VF helper (``@pl.vector_function`` decorated)."""
    for dec in func_def.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr == "vector_function":
            return True
        if isinstance(dec, ast.Name) and dec.id == "vector_function":
            return True
    return False


def _calls_any_vf_function(func_def: ast.FunctionDef, vf_func_defs: dict[str, ast.FunctionDef]) -> bool:
    """Check if a function body calls any known VF function (one level of indirection)."""
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in vf_func_defs:
                return True
    return False


def _get_call_func_name(call: ast.Call) -> str:
    """Get the function name from a Call node."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _assigned_vars_from_assign(stmt: ast.Assign) -> set[str]:
    """Collect simple assignment target names, excluding method-call results."""
    assigned = set()
    for target in stmt.targets:
        if not isinstance(target, ast.Name):
            continue
        if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute):
            continue
        assigned.add(target.id)
    return assigned


def _assigned_vars_from_stmt(stmt: ast.stmt) -> set[str]:
    """Collect simple assigned variable names from one statement."""
    if isinstance(stmt, ast.Assign):
        return _assigned_vars_from_assign(stmt)
    assigned = set()
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
        assigned.add(stmt.target.id)
        return assigned
    if isinstance(stmt, ast.For):
        if isinstance(stmt.target, ast.Name):
            assigned.add(stmt.target.id)
        assigned.update(_collect_assigned_vars(stmt.body))
        return assigned
    if isinstance(stmt, ast.With):
        assigned.update(_collect_assigned_vars(stmt.body))
        return assigned
    if isinstance(stmt, ast.If):
        assigned.update(_collect_assigned_vars(stmt.body))
        assigned.update(_collect_assigned_vars(stmt.orelse))
    return assigned


def _collect_assigned_vars(stmts: list[ast.stmt]) -> set[str]:
    """Collect all variable names assigned in a list of statements (recursively).

    Only includes simple scalar assignments (not method call results like .current()).
    """
    assigned = set()
    for stmt in stmts:
        assigned.update(_assigned_vars_from_stmt(stmt))
    return assigned


def _collect_loop_vars(stmts: list[ast.stmt]) -> set[str]:
    """Collect all for-loop variables in a statement list (recursively).
    These are the seeds of 'changing' variables."""
    loop_vars: set[str] = set()
    for stmt in stmts:
        if isinstance(stmt, ast.For):
            if isinstance(stmt.target, ast.Name):
                loop_vars.add(stmt.target.id)
            loop_vars.update(_collect_loop_vars(stmt.body))
        elif isinstance(stmt, ast.With):
            loop_vars.update(_collect_loop_vars(stmt.body))
        elif isinstance(stmt, ast.If):
            loop_vars.update(_collect_loop_vars(stmt.body))
            loop_vars.update(_collect_loop_vars(stmt.orelse))
    return loop_vars


def _names_in_expr(node: ast.expr) -> set[str]:
    """All Name ids referenced (Load) inside an expression."""
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _collect_assignments(stmts: list[ast.stmt]) -> list[tuple[str, set[str]]]:
    """Collect (lhs_name, rhs_referenced_names) for every scalar assignment.
    Method-call results excluded; AugAssign includes target in rhs (loop-carried)."""
    out: list[tuple[str, set[str]]] = []
    for stmt in stmts:
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute):
                continue  # exclude slot accessor results
            rhs = _names_in_expr(stmt.value)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    out.append((target.id, rhs))
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name):
                rhs = _names_in_expr(stmt.value) | {stmt.target.id}
                out.append((stmt.target.id, rhs))
        elif isinstance(stmt, ast.For):
            out.extend(_collect_assignments(stmt.body))
        elif isinstance(stmt, ast.With):
            out.extend(_collect_assignments(stmt.body))
        elif isinstance(stmt, ast.If):
            out.extend(_collect_assignments(stmt.body))
            out.extend(_collect_assignments(stmt.orelse))
    return out


def _collect_changing_vars(stmts: list[ast.stmt], loop_var_seeds: set[str]) -> set[str]:
    """Collect variables that change across iterations (transitive closure).

    A variable is 'changing' if it depends (directly or transitively) on a loop
    variable, or is loop-carried (x = x + 1). Constants are excluded.
    """
    changing = set(loop_var_seeds)
    assignments = _collect_assignments(stmts)
    for lhs, rhs in assignments:
        if lhs in rhs:
            changing.add(lhs)
    # Transitive closure
    grew = True
    while grew:
        grew = False
        for lhs, rhs in assignments:
            if lhs in changing:
                continue
            if rhs & changing:
                changing.add(lhs)
                grew = True
    return changing


def _is_scalar_arith_expr(node: ast.expr) -> bool:
    """True if node is a pure scalar arithmetic expression (Name/Constant/BinOp/UnaryOp)."""
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.BinOp):
        return _is_scalar_arith_expr(node.left) and _is_scalar_arith_expr(node.right)
    if isinstance(node, ast.UnaryOp):
        return _is_scalar_arith_expr(node.operand)
    return False


def _derive_ctx_fields(info: PipelineInfo, closure_vars: dict):
    """Determine which stage arguments are 'changing' (need ctx) vs 'fixed'.

    A parameter goes into ctx if it varies across iterations:
    - a simple Name in loop_changing_vars (the changing-vars set), or
    - a pure scalar arithmetic expression referencing a changing var (e.g. `qi*TS`).

    Otherwise it's 'fixed' (kernel-level object, constant, or loop-invariant expr).

    stage_arg_mapping stores per-stage, per-arg: None (fixed) or (field_name, fill_expr).
    - field_name: the ctx struct field name (e.g. "ki" or "_pl_arg_compute_qk_1")
    - fill_expr: the AST expression to fill into the ctx each iteration
      (for plain Names: ast.Name(id=field_name); for expressions: the original expr AST)
    """
    ctx_field_set = set()
    info.stage_arg_mapping = []
    changing = info.loop_changing_vars

    for stage in info.stages:
        arg_map = []
        for argpos, arg in enumerate(stage.args):
            if isinstance(arg, ast.Name):
                if arg.id in changing:
                    ctx_field_set.add(arg.id)
                    # If this variable is a mid-loop assignment (e.g. p_offset = ki),
                    # use its rhs expression as fill_expr (the rhs only depends on
                    # loop vars/constants available at loop start). Otherwise use the
                    # variable name itself (it's a loop var or outer-scope variable).
                    if arg.id in info.mid_loop_assigns:
                        fill_expr = copy.deepcopy(info.mid_loop_assigns[arg.id])
                    else:
                        fill_expr = ast.Name(id=arg.id, ctx=ast.Load())
                    arg_map.append((arg.id, fill_expr))
                else:
                    arg_map.append(None)
                continue
            # Non-Name argument: check if it references a changing variable
            refs = _names_in_expr(arg)
            if refs & changing:
                if not _is_scalar_arith_expr(arg):
                    raise ValueError(
                        f"pipeline: stage '{stage.func_name}' arg #{argpos} is an "
                        f"unsupported expression that depends on a loop variable. Only "
                        f"plain scalar arithmetic (e.g. `ki + 1`, `qi * TS`) is allowed; "
                        f"subscripts/calls/attribute access are not supported."
                    )
                field_name = f"_pl_arg_{stage.func_name}_{argpos}"
                ctx_field_set.add(field_name)
                arg_map.append((field_name, arg))
            else:
                arg_map.append(None)
        info.stage_arg_mapping.append(arg_map)

    # Always include is_valid as the first field.
    # _pl_task_id is a framework field holding the task_id of the data this ctx
    # slot carries; used by auto-sync to index cross-core event_id tuples with
    # the correct (delayed) task index, independent of any user counter.
    info.ctx_fields = ["is_valid"] + sorted(ctx_field_set) + ["_pl_task_id"]


def _identify_mix_stages(info: PipelineInfo, stage_func_names: set, closure_vars: dict):
    """Identify mix stages: a stage whose body calls other @stage functions.

    For mix stages:
      - is_mix = True
      - sub_stages = list of sub-stage func_names in source order
      - section_kind = last sub-stage's section_kind (inferred from its section block)
    """
    for stage in info.stages:
        fn = closure_vars.get(stage.func_name)
        fd = _try_get_funcdef(fn)
        if fd is None:
            continue
        sub_stages = _find_sub_stage_calls(fd, stage_func_names)
        if sub_stages:
            stage.is_mix = True
            stage.sub_stages = sub_stages
            last_section = _infer_last_sub_stage_section(fd, stage_func_names)
            if last_section:
                stage.section_kind = last_section


def _find_sub_stage_calls(func_def: ast.FunctionDef, stage_func_names: set) -> list[str]:
    """Find all @stage function calls inside a function body (in source order)."""
    found = []
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in stage_func_names:
            if node.func.id not in found:
                found.append(node.func.id)
    return found


def _infer_last_sub_stage_section(func_def: ast.FunctionDef, stage_func_names: set) -> str | None:
    """Find the section_kind of the last sub-stage call in a mix stage body."""
    holder = [None]
    _scan_sections_for_last(func_def.body, stage_func_names, holder)
    return holder[0]


def _scan_sections_for_last(stmts: list, stage_func_names: set, holder: list):
    """Recursively scan for the last section block containing a sub-stage call."""
    for stmt in stmts:
        if isinstance(stmt, ast.With):
            section_kind = _get_section_kind(stmt)
            if section_kind is not None:
                for inner in stmt.body:
                    if not (isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Call)):
                        continue
                    if isinstance(inner.value.func, ast.Name) and inner.value.func.id in stage_func_names:
                        holder[0] = section_kind
            _scan_sections_for_last(stmt.body, stage_func_names, holder)
        elif isinstance(stmt, ast.For):
            _scan_sections_for_last(stmt.body, stage_func_names, holder)
        elif isinstance(stmt, ast.If):
            _scan_sections_for_last(stmt.body, stage_func_names, holder)
            _scan_sections_for_last(stmt.orelse, stage_func_names, holder)


def _find_sub_stage_section_in_body(mix_fd: ast.FunctionDef, sub_name: str) -> str | None:
    """Find the section_kind that a sub-stage call lives in within a mix func body."""
    for node in ast.walk(mix_fd):
        if not isinstance(node, ast.With):
            continue
        section_kind = _get_section_kind(node)
        if section_kind is None:
            continue
        for inner in node.body:
            if not (isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Call)):
                continue
            if isinstance(inner.value.func, ast.Name) and inner.value.func.id == sub_name:
                return section_kind
    return None


def _find_sub_stage_call_args(mix_fd: ast.FunctionDef, sub_name: str) -> list | None:
    """Find the call-site args (list of AST nodes) of a sub-stage call inside a mix
    func body. Returns None if not found."""
    for node in ast.walk(mix_fd):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == sub_name:
            return node.args
    return None


def _check_alternating(info: PipelineInfo):
    """C5: verify the stage chain strictly alternates cube/vector."""
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


def _check_single_producer_consumer(info: PipelineInfo):
    """C7: at most one producer (W) and one consumer (R) stage per cross-core buffer."""
    producers: dict[str, list[str]] = {}
    consumers: dict[str, list[str]] = {}
    for stage in info.stages:
        for acc in stage.cross_access:
            tbl = producers if acc.role == AccessRole.WRITE else consumers
            tbl.setdefault(acc.buffer_name, []).append(stage.func_name)
    for buf, prods in producers.items():
        if len(prods) > 1:
            raise ValueError(
                f"pipeline: cross-core buffer '{buf}' is produced (written) by "
                f"multiple stages {prods}. Only one producer stage per cross-core "
                f"buffer is supported."
            )
    for buf, cons in consumers.items():
        if len(cons) > 1:
            raise ValueError(
                f"pipeline: cross-core buffer '{buf}' is consumed (read) by multiple "
                f"stages {cons}. Only one consumer stage per cross-core buffer is "
                f"supported."
            )
