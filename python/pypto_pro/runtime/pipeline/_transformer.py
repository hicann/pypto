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

"""Transformer: convert serial kernel AST to preload pipeline AST."""
from __future__ import annotations

import ast
import copy

from ._analyzer import PipelineInfo, analyze_pipeline
from .config import PipelineConfig
from ._cross_core_scanner import AccessRole


def _snake_visit_name(node: ast.AST) -> str:
    """Return the snake_case visitor handler for an AST node type."""
    chars: list[str] = []
    for char in type(node).__name__:
        if char.isupper() and chars:
            chars.append("_")
        chars.append(char.lower())
    return f"visit_{''.join(chars)}"


class _CtxFieldReplacer(ast.NodeTransformer):
    """Replace ctx-field Name references with ctx_var.field attribute access.

    Used to rewrite sync statements (pre/post stmts around a stage) so that
    references to ctx fields (e.g. `tick`) become `ctx_var.tick`, where ctx_var
    is the delayed ctx for the stage. This makes the sync event_id index use the
    delayed task's value, matching the data the stage actually processes.
    """

    def __init__(self, ctx_var: str, fields: set):
        self.ctx_var = ctx_var
        self.fields = fields

    def visit(self, node):
        method = getattr(self, _snake_visit_name(node), None)
        if method is not None:
            return method(node)
        return super().visit(node)

    def visit_name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.fields:
            return ast.copy_location(
                ast.Attribute(
                    value=ast.Name(id=self.ctx_var, ctx=ast.Load()),
                    attr=node.id, ctx=ast.Load(),
                ),
                node,
            )
        return node


def _replace_ctx_fields(node, ctx_var: str, fields: set):
    """Apply ctx-field replacement to an AST node (in place) and return it."""
    if not fields:
        return node
    return _CtxFieldReplacer(ctx_var, fields).visit(node)


class _ParamReplacer(ast.NodeTransformer):
    """Replace formal-parameter Name references with the call-site actual-arg AST.

    Used when flattening (inlining) a mix stage body into the main loop: each
    reference to a formal param (e.g. `ki`) becomes the actual argument expression
    passed at the call site (e.g. `_pl_ctx_0.ki` in pipeline mode, or a plain Name
    in sync_only mode). Only Load references are replaced — a param being assigned
    (Store) inside the body would shadow it, so we stop replacing it from then on.
    """

    def __init__(self, mapping: dict):
        # mapping: param_name -> actual-arg ast.expr
        self.mapping = dict(mapping)

    def visit(self, node):
        method = getattr(self, _snake_visit_name(node), None)
        if method is not None:
            return method(node)
        return super().visit(node)

    def visit_name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return ast.copy_location(copy.deepcopy(self.mapping[node.id]), node)
        # A local assignment to a param name shadows the param: stop replacing it.
        if isinstance(node.ctx, ast.Store) and node.id in self.mapping:
            del self.mapping[node.id]
        return node


def _flatten_mix_body(stage, call_expr: ast.Call, info) -> list[ast.stmt]:
    """Flatten (inline) a synced mix function body into the call site.

    On first call for a given mix stage, builds the synced mix body (inserts inner
    cross-core sync) and caches it in info.mix_synced_funcs. Subsequent calls reuse
    the cached version. Then builds the formal-param -> actual-arg mapping and
    param-replaces the body. Returns the body statement list to splice in place of
    the call.

    Keeping the mix body in the main kernel AST (instead of leaving a function
    call) ensures the parser does not inline it by re-reading the original,
    sync-free source — so the inner sync survives into codegen. See design §12.12.
    """
    # Lazily build and cache the synced mix func body.
    mix_fd = info.mix_synced_funcs.get(stage.func_name)
    if mix_fd is None and stage.inner_buffers:
        mix_fd = _build_mix_synced_func(stage, info, info.closure_vars)
        if mix_fd is not None:
            info.mix_synced_funcs[stage.func_name] = mix_fd
    if mix_fd is None:
        # No inner sync needed — fall back to getting the raw func body.
        from ._analyzer import _try_get_funcdef
        raw_fn = info.closure_vars.get(stage.func_name)
        mix_fd = _try_get_funcdef(raw_fn)
        if mix_fd is None:
            return [ast.Expr(value=call_expr, lineno=0)]

    params = [a.arg for a in mix_fd.args.args if a.arg != "self"]
    args = call_expr.args
    mapping = {p: args[i] for i, p in enumerate(params) if i < len(args)}
    body = copy.deepcopy(mix_fd.body)
    replacer = _ParamReplacer(mapping)
    return [replacer.visit(stmt) for stmt in body]



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
            value=ast.Attribute(
                value=ast.Name(id="pl", ctx=ast.Load()),
                attr="system", ctx=ast.Load()),
            attr=fn_name, ctx=ast.Load()),
        args=[],
        keywords=[
            ast.keyword(arg="pipe", value=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id="pl", ctx=ast.Load()),
                    attr="PipeType", ctx=ast.Load()),
                attr=pipe_name, ctx=ast.Load())),
            ast.keyword(arg="event_id", value=event_id),
        ],
    )
    return ast.Expr(value=call, lineno=0)


def _build_sync_call(fn_name: str, pipe_name: str,
                     ids_node: ast.expr, slot_count: int, index_expr: ast.expr) -> ast.stmt:
    """Build `pl.system.<fn_name>(pipe=pl.PipeType.<pipe>, event_id=<ids>[idx])`."""
    event_id = _build_event_id_index(ids_node, slot_count, index_expr)
    return _build_system_sync_stmt(fn_name, pipe_name, event_id)


def _build_cross_core_sync_one(acc, index_expr: ast.expr) -> tuple[list, list]:
    """Build (pre_sync, post_sync) for a SINGLE cross-core access (see
    _build_cross_core_sync for the W/R rules)."""
    pre, post = [], []
    buf = acc.buffer
    if acc.role == AccessRole.WRITE:
        if buf.bwd_ids_node is not None:
            pre.append(_build_sync_call(
                "wait_cross_core", acc.first_pipe, buf.bwd_ids_node, buf.bwd_slot_count, index_expr))
        if buf.fwd_ids_node is not None:
            post.append(_build_sync_call(
                "set_cross_core", acc.last_pipe, buf.fwd_ids_node, buf.fwd_slot_count, index_expr))
    else:  # READ
        if buf.fwd_ids_node is not None:
            pre.append(_build_sync_call(
                "wait_cross_core", acc.first_pipe, buf.fwd_ids_node, buf.fwd_slot_count, index_expr))
        if buf.bwd_ids_node is not None:
            post.append(_build_sync_call(
                "set_cross_core", acc.last_pipe, buf.bwd_ids_node, buf.bwd_slot_count, index_expr))
    return pre, post


def _build_cross_core_sync(stage, index_expr: ast.expr) -> tuple[list, list]:
    """Build (pre_sync, post_sync) statement lists for a stage's cross-core
    accesses, based on W/R role:

      W (producer): pre wait(bwd), post set(fwd)
      R (consumer): pre wait(fwd), post set(bwd)

    Pipe selection for multi-op accesses (different pipes):
      - pre (wait, before stage): uses acc.first_pipe (gates the first access)
      - post (set, after stage):  uses acc.last_pipe  (fires after the last access)

    Each emitted only if the corresponding id tuple is configured on the buffer.
    event_id index = index_expr % slot_count. index_expr is the iteration index
    base: in pipeline mode `_pl_ctx_negD._pl_task_id`, in sync_only mode `_pl_sync_id`.
    """
    pre, post = [], []
    for acc in stage.cross_core:
        p, q = _build_cross_core_sync_one(acc, index_expr)
        pre.extend(p)
        post.extend(q)
    return pre, post


def transform_pipeline(func_def: ast.FunctionDef, closure_vars: dict,
                       config: PipelineConfig) -> ast.FunctionDef:
    """Transform a serial kernel function into a preload pipeline version.

    Args:
        func_def: The kernel function's AST (will not be mutated; a copy is made)
        closure_vars: Closure variables (contains stage functions, module constants)
        config: Pipeline configuration (depth, dump_generated, etc.)

    Returns:
        New ast.FunctionDef with pipeline transformation applied
    """
    # Work on a deep copy to avoid mutating the original
    new_func = copy.deepcopy(func_def)

    # Analyze the serial structure (raises if no usable stages / pipeline loop)
    info = analyze_pipeline(new_func, closure_vars)

    if config.sync_only:
        # Validation mode: keep the serial loop, only auto-insert cross-core sync.
        # Mix calls are flattened in-place inside _insert_sync_into_loop_body.
        _transform_sync_only(new_func.body, info)
        ast.fix_missing_locations(new_func)
        return new_func

    # Compute delays based on preload
    _compute_delays(info, config.preload)

    # Check framework var names don't clash with user names
    _check_name_collisions(new_func, info)

    # Defensive: verify delay monotonicity along dependency chain
    _check_pipeline_consistency(info)

    # ctx ring buffer depth = max_delay + 1
    max_delay = max((s.delay for s in info.stages), default=0)
    depth = max_delay + 1
    extra = depth - 1

    # Find and replace the pipeline loop in the function body. Mix stages are
    # flattened (inlined) into the loop body during this pass.
    _transform_body(new_func.body, info, depth, extra, closure_vars)

    ast.fix_missing_locations(new_func)
    return new_func


def _build_mix_synced_func(stage, info: PipelineInfo, closure_vars: dict):
    """Build a mix stage's function def with inner cross-core sync inserted.

    Returns a deep-copied ast.FunctionDef whose body has wait/set inserted around
    each sub-stage that touches an inner buffer, or None if there is no inner sync
    to insert. Called lazily by _flatten_mix_body (and cached there); the result is
    then flattened (inlined) into the main loop.
    """
    from ._analyzer import _try_get_funcdef, _is_vf_function
    from ._cross_core_scanner import scan_stage_accesses

    mix_fn = closure_vars.get(stage.func_name)
    mix_fd = _try_get_funcdef(mix_fn)
    if mix_fd is None:
        return None
    mix_fd = copy.deepcopy(mix_fd)

    # Scan sub-stages for inner buffer accesses
    stage_names = {s.func_name for s in info.stages}
    vf_func_defs = {}
    for name, val in closure_vars.items():
        if callable(val) and name not in stage_names:
            fd = _try_get_funcdef(val)
            if fd is not None and _is_vf_function(fd):
                vf_func_defs[name] = fd

    all_mem = info.sync.all_memory
    sub_sync_info = {}
    for sub_name in stage.sub_stages:
        sub_fn = closure_vars.get(sub_name)
        sub_fd = _try_get_funcdef(sub_fn)
        if sub_fd is None:
            continue
        accesses = scan_stage_accesses(sub_fd, info.sync.buffers, vf_func_defs, all_mem)
        inner_accs = [a for a in accesses if a.buffer_name in stage.inner_buffers]
        if inner_accs:
            sub_sync_info[sub_name] = inner_accs

    if not sub_sync_info:
        return None

    # Insert sync into mix function body. The counter _pl_inner_sync_id is
    # incremented after the LAST sub-stage's section (one +1 per "round" of the
    # full inner chain), regardless of control flow structure (for/if/bare).
    last_sub = stage.sub_stages[-1]
    index_expr = ast.Name(id="_pl_inner_sync_id", ctx=ast.Load())
    _insert_inner_sync_recursive(mix_fd.body, sub_sync_info, index_expr, last_sub)

    ast.fix_missing_locations(mix_fd)
    return mix_fd


def _insert_inner_sync_recursive(stmts: list[ast.stmt], sub_sync_info: dict,
                                  index_expr: ast.expr, last_sub: str) -> bool:
    """Recursively find section blocks with sub-stage calls, insert sync.

    After the last sub-stage's section block (identified by `last_sub`), a
    `_pl_inner_sync_id += 1` is inserted at the same level (sibling), so that
    each "round" of the inner chain advances the counter once — regardless of
    whether the stages sit inside a for-loop, an if/else branch, or bare code.

    Returns True if any sync was inserted."""
    found_any = False
    i = 0
    while i < len(stmts):
        stmt = stmts[i]
        if isinstance(stmt, ast.With):
            for j, inner in enumerate(stmt.body):
                if isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Call):
                    fname = _get_call_name(inner.value)
                    if fname in sub_sync_info:
                        class _H:
                            cross_core = sub_sync_info[fname]
                        pre, post = _build_cross_core_sync(_H(), index_expr)
                        stmt.body[j:j + 1] = pre + [inner] + post
                        found_any = True
                        # If this is the last sub-stage, insert counter incr
                        # AFTER this with block (sibling level in stmts).
                        if fname == last_sub:
                            stmts.insert(i + 1, _build_counter_incr("_pl_inner_sync_id"))
                            i += 1  # skip over the just-inserted incr
                        break
            else:
                _insert_inner_sync_recursive(stmt.body, sub_sync_info, index_expr, last_sub)
        elif isinstance(stmt, ast.For):
            if _insert_inner_sync_recursive(stmt.body, sub_sync_info, index_expr, last_sub):
                found_any = True
        elif isinstance(stmt, ast.If):
            r1 = _insert_inner_sync_recursive(stmt.body, sub_sync_info, index_expr, last_sub)
            r2 = _insert_inner_sync_recursive(stmt.orelse, sub_sync_info, index_expr, last_sub)
            if r1 or r2:
                found_any = True
        i += 1
    return found_any


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


def _transform_sync_only(stmts: list[ast.stmt], info: PipelineInfo,
                         _insert_decls_here: bool = True) -> bool:
    """Validation mode: keep the serial loop structure, only auto-insert
    cross-core sync around each stage, plus a per-iteration `_pl_sync_id`
    counter and a pre-fire block before the OUTERMOST loop.

    Recursively finds the pipeline loop (matching info.inner_loop_var); inside its
    body each section block's stage call gets wait/set wrapped around it,
    indexed by `_pl_sync_id % slot_count`. No ctx / guard / delay / extra.

    `_pl_sync_id` and pre-fire are placed before the outermost loop so the
    counter advances continuously across outer-loop iterations (matching the
    cross-core buffer cursor, which does not reset) and pre-fire fires once.
    """
    # Map stage func_name -> StageCall (has cross_core summary)
    stage_by_name = {s.func_name: s for s in info.stages}

    for i, stmt in enumerate(stmts):
        if isinstance(stmt, ast.For):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == info.inner_loop_var:
                # Found the pipeline loop: insert sync into its body + counter incr.
                _insert_sync_into_loop_body(stmt.body, stage_by_name, info)
                stmt.body.append(_build_counter_incr("_pl_sync_id"))
                if _insert_decls_here:
                    stmts[i:i] = _build_sync_only_decls(info)
                return True
            else:
                # Outer loop — recurse into its body. Declarations go BEFORE this
                # outer loop (continuous counter + single pre-fire), not inside it.
                if _transform_sync_only(stmt.body, info, _insert_decls_here=False):
                    stmts[i:i] = _build_sync_only_decls(info)
                    return True
        elif isinstance(stmt, ast.With):
            if _transform_sync_only(stmt.body, info, _insert_decls_here):
                return True
    return False


def _build_sync_only_decls(info: PipelineInfo) -> list[ast.stmt]:
    """Build lifted-id decls + pre-fire + `_pl_sync_id = 0` to place before the
    outermost loop."""
    decls = []
    decls.extend(_build_lifted_id_decls(info))
    decls.extend(_build_prefire(info))
    decls.append(ast.Assign(
        targets=[ast.Name(id="_pl_sync_id", ctx=ast.Store())],
        value=ast.Constant(value=0), lineno=0))
    return decls


def _insert_sync_into_loop_body(body: list[ast.stmt], stage_by_name: dict,
                                info: PipelineInfo) -> None:
    """Insert wait(before)/set(after) cross-core sync around each stage call in
    the loop body. Index base is `_pl_sync_id` (current iteration, no delay).

    Two cases:
      - leaf stage: call is inside a `with pl.section_*()` block -> wrap sync
        inside that same section.
      - mix stage: bare call (not in a section) -> flatten (inline) the synced mix
        body in place of the call, then wrap each outer-buffer sync in the
        sub-stage's own section via _build_mix_outer_sync.
    """
    index_expr = ast.Name(id="_pl_sync_id", ctx=ast.Load())
    for i, stmt in enumerate(body):
        # Mix stage: bare call directly in the loop body.
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            stage = stage_by_name.get(_get_call_name(stmt.value))
            if stage is not None and stage.is_mix:
                mix_body = _flatten_mix_body(stage, stmt.value, info)
                body[i:i + 1] = _build_mix_outer_sync(stage, mix_body, index_expr)
            continue
        if not isinstance(stmt, ast.With):
            continue
        # Leaf stage: call inside a section block.
        for j, inner in enumerate(stmt.body):
            if (isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Call)):
                fname = _get_call_name(inner.value)
                stage = stage_by_name.get(fname)
                if stage is not None:
                    pre, post = _build_cross_core_sync(stage, index_expr)
                    stmt.body[j:j + 1] = pre + [inner] + post
                    break


def _get_call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
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


def _framework_var_names(info: PipelineInfo) -> set[str]:
    """All framework-introduced variable names this transform will generate."""
    names = {"_pl_ctx_arr", "_pl_task_id"}
    for stage in info.stages:
        names.add("_pl_ctx_0" if stage.delay == 0 else f"_pl_ctx_neg{stage.delay}")
    return names


def _check_name_collisions(func_def: ast.FunctionDef, info: PipelineInfo) -> None:
    """Raise if any framework-introduced name collides with a user name."""
    fw_names = _framework_var_names(info)
    user_names = {n.id for n in ast.walk(func_def) if isinstance(n, ast.Name)}
    clash = sorted(fw_names & user_names)
    if clash:
        raise ValueError(
            f"pipeline transform: framework variable name(s) {clash} collide with "
            f"user-defined names in the kernel. The '_pl_' prefix is reserved for "
            f"the pipeline transform — please rename the conflicting user variable(s)."
        )


def _check_pipeline_consistency(info: PipelineInfo) -> None:
    """Defensive check: consumer delay must not be less than producer delay.

    Equality (same delay = same-tick dispatch) is perfectly legal — the cross-core
    sync serializes them. Only delay(consumer) < delay(producer) is a formula bug.
    """
    producer = {}
    consumer = {}
    for stage in info.stages:
        for acc in stage.cross_core:
            if acc.role == AccessRole.WRITE:
                producer[acc.buffer_name] = stage
            elif acc.role == AccessRole.READ:
                consumer[acc.buffer_name] = stage

    for buf, cons in consumer.items():
        prod = producer.get(buf)
        if prod is None:
            continue
        if cons.delay < prod.delay:
            raise ValueError(
                f"pipeline (internal): cross-core buffer '{buf}' consumer "
                f"'{cons.func_name}' (delay={cons.delay}) runs earlier than producer "
                f"'{prod.func_name}' (delay={prod.delay}). This indicates a delay-"
                f"assignment bug — please report."
            )


def _transform_body(stmts: list[ast.stmt], info: PipelineInfo,
                    depth: int, extra: int, closure_vars: dict,
                    _insert_decls_here: bool = True) -> bool:
    """Recursively find and transform the pipeline loop in the statement list.

    When _insert_decls_here is True, declarations (_pl_ctx_arr, _pl_task_id) are inserted
    at the current level before the loop. When False, they are inserted by the caller.
    """
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, ast.For):
            # Check if this is our pipeline loop (has the right loop var)
            if isinstance(stmt.target, ast.Name) and stmt.target.id == info.inner_loop_var:
                # Found it - replace with transformed version
                new_stmts = _build_pipeline_loop(stmt, info, depth, extra)
                if _insert_decls_here:
                    # Insert declarations at this level (before the loop)
                    decls = _build_declarations(info, depth)
                    stmts[i:i + 1] = decls + new_stmts
                else:
                    # Just replace the loop, declarations handled by caller
                    stmts[i:i + 1] = new_stmts
                return True
            else:
                # Not our target loop - recurse into its body.
                # Declarations should go BEFORE the OUTERMOST loop, only once.
                if _transform_body(stmt.body, info, depth, extra, closure_vars,
                                   _insert_decls_here=False):
                    if _insert_decls_here:
                        decls = _build_declarations(info, depth)
                        stmts[i:i] = decls
                    return True
        elif isinstance(stmt, ast.With):
            if _transform_body(stmt.body, info, depth, extra, closure_vars,
                               _insert_decls_here):
                return True
    return False


def _build_lifted_id_decls(info: PipelineInfo) -> list[ast.stmt]:
    """Declare lifted literal fwd_ids/bwd_ids as variables, plus `_pl_inner_sync_id`
    if any mix stage has inner buffers.

    Shared by both the full-pipeline path (_build_declarations) and the
    sync_only path (_build_sync_only_decls):
      - lifted ids: the buffer's ids_node was rewritten to a Name reference, so
        sync code uses `var[idx]` instead of unsupported `(1,2)[idx]` subscript.
      - `_pl_inner_sync_id`: counter for mix-stage inner-buffer sync.
    """
    stmts = []
    for var_name, literal_node in info.sync.lifted_ids:
        stmts.append(ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=copy.deepcopy(literal_node),
            lineno=0,
        ))
    if any(s.is_mix and s.inner_buffers for s in info.stages):
        stmts.append(ast.Assign(
            targets=[ast.Name(id="_pl_inner_sync_id", ctx=ast.Store())],
            value=ast.Constant(value=0),
            lineno=0,
        ))
    return stmts


def _build_declarations(info: PipelineInfo, depth: int) -> list[ast.stmt]:
    """Build _pl_ctx_arr and _pl_task_id declarations."""
    stmts = []

    keywords = [ast.keyword(arg=f, value=ast.Constant(value=0)) for f in info.ctx_fields]
    ctx_call = ast.Call(
        func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()),
                           attr="struct_array", ctx=ast.Load()),
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
        targets=[ast.Name(id="_pl_task_id", ctx=ast.Store())],
        value=ast.Constant(value=0),
        lineno=0,
    )
    stmts.append(_pl_task_id_assign)

    # Lifted ids + _pl_inner_sync_id (shared with sync_only path).
    stmts.extend(_build_lifted_id_decls(info))

    # Pre-fire: release all backward slots before the loop (consumer side)
    # so the first-round producer wait(bwd) doesn't deadlock. Each pre-fire set
    # uses the consumer's section/pipe (returns one block per consumer section).
    stmts.extend(_build_prefire(info))

    return stmts


def _consumer_sync_for_buffer(info: PipelineInfo, buffer_name: str):
    """Return (section_kind, pipe) of the consumer (role 'R') of a cross-core
    buffer, or None if no consumer found. Backward release is emitted by the
    consumer in steady state, so pre-fire must match the consumer's
    section/pipe to land in the same synchronization domain."""
    for stage in info.stages:
        for acc in stage.cross_core:
            if acc.buffer_name == buffer_name and acc.role == AccessRole.READ:
                return stage.section_kind, acc.last_pipe
    return None


def _build_prefire(info: PipelineInfo) -> list[ast.stmt]:
    """Build section block(s) that pre-fire all backward cross-core slots
    (set_cross_core for each bwd id of each cross-core buffer), so the
    first-round producer wait(bwd) doesn't deadlock.

    Handles both:
      - outer buffers: consumer found among outer stages (info.stages)
      - inner buffers (mix): consumer found among mix sub-stages

    The pipe and section of each pre-fire set MATCH THE BUFFER'S CONSUMER
    (the stage with role 'R' on that buffer), because backward release is
    emitted by the consumer in steady state. Sets are grouped by the
    consumer's section_kind (one `with pl.section_*()` block per kind).
    Returns [] if no buffer has backward ids.
    """
    # section_kind -> list of (pipe_name, bwd_ids_node, slot)
    by_section: dict[str, list] = {}

    for buf in info.sync.buffers.values():
        if buf.bwd_ids_node is None or buf.bwd_slot_count <= 0:
            continue
        # Try outer consumer first
        consumer = _consumer_sync_for_buffer(info, buf.name)
        if consumer is None:
            # Try inner consumer (mix stage's inner buffer)
            consumer = _inner_consumer_sync_for_buffer(info, buf.name)
        if consumer is None:
            section_kind, pipe_name = "vector", "V"  # fallback
        else:
            section_kind, pipe_name = consumer
        for slot in range(buf.bwd_slot_count):
            by_section.setdefault(section_kind, []).append(
                (pipe_name, buf.bwd_ids_node, slot))

    if not by_section:
        return []

    result: list[ast.stmt] = []
    for section_kind, entries in by_section.items():
        body = []
        for pipe_name, ids_node, slot in entries:
            event_id = ast.Subscript(
                value=copy.deepcopy(ids_node),
                slice=ast.Constant(value=slot),
                ctx=ast.Load(),
            )
            body.append(_build_system_sync_stmt("set_cross_core", pipe_name, event_id))
        section_attr = f"section_{section_kind}"
        section_call = ast.Call(
            func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()),
                               attr=section_attr, ctx=ast.Load()),
            args=[], keywords=[],
        )
        result.append(ast.With(
            items=[ast.withitem(context_expr=section_call, optional_vars=None)],
            body=body,
            lineno=0,
        ))
    return result


def _inner_consumer_sync_for_buffer(info: PipelineInfo, buffer_name: str):
    """Find (section_kind, pipe) of the inner consumer of a mix stage's buffer.
    Uses pre-computed inner_consumer_map from analyzer."""
    return info.sync.inner_consumer_map.get(buffer_name)


def _build_pipeline_loop(original_for: ast.For, info: PipelineInfo,
                         depth: int, extra: int) -> list[ast.stmt]:
    """Build the transformed pipeline for-loop."""

    # Modify loop range: pl.range(0, end) -> pl.range(0, end + extra)
    new_for = copy.deepcopy(original_for)
    _extend_loop_range(new_for, extra)

    # Collect user statements from original loop body that are NOT stage-related.
    # Two categories:
    #   - "ctx-taken": changing mid-loop assignments whose value is already captured
    #     by ctx fill (via their rhs expression). These are dead after transform → skip.
    #   - "preserved": other user statements (e.g. tick = tick + 1) → keep at loop end.
    #
    # A statement is "ctx-taken" if: it's a simple assignment `name = expr` where
    # `name` is in ctx_fields AND the rhs does NOT reference itself (i.e. not a
    # loop-carried accumulator like `tick = tick + 1`).
    stage_func_names = {s.func_name for s in info.stages}
    ctx_field_names = set(info.ctx_fields)
    pre_stmts = []       # user stmts that must run BEFORE ctx fill (non-changing mid-loop assigns)
    preserved_stmts = [] # user stmts that run AFTER all stages (loop-carried accumulators etc.)
    for stmt in original_for.body:
        if isinstance(stmt, ast.With):
            continue  # section blocks are regenerated
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            fname = _get_call_name(stmt.value)
            if fname in stage_func_names:
                continue  # bare stage calls (mix) are regenerated
        # Check if this is a ctx-taken mid-loop assignment (dead after transform):
        # its value is already captured by ctx fill via the rhs expression.
        # Only delete if it's in ctx_fields (meaning it was changing and ctx-backed).
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and target.id in info.mid_loop_assigns:
                if target.id in ctx_field_names:
                    continue  # ctx-taken changing assign → skip (dead code)
                else:
                    # Non-changing mid-loop assign (e.g. p_x = 3): stage references
                    # it directly by name, so it must exist before stage calls.
                    pre_stmts.append(copy.deepcopy(stmt))
                    continue
        preserved_stmts.append(copy.deepcopy(stmt))

    # Build new loop body
    new_body = []

    # 0. Non-changing mid-loop assigns that stages reference by name (e.g. p_x = 3)
    #    must exist before ctx fill / stage calls.
    new_body.extend(pre_stmts)

    # 1. Fill current ctx
    new_body.extend(_build_ctx_fill(info, depth))

    # 2. Stage calls with delay and is_valid guard
    for stage_idx, stage in enumerate(info.stages):
        new_body.extend(_build_stage_call(stage, stage_idx, info, depth))

    # 3. task_id increment
    new_body.append(ast.Assign(
        targets=[ast.Name(id="_pl_task_id", ctx=ast.Store())],
        value=ast.BinOp(
            left=ast.Name(id="_pl_task_id", ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(value=1),
        ),
        lineno=0,
    ))

    # 4. Preserve user's non-stage statements (e.g. tick = tick + 1)
    new_body.extend(preserved_stmts)

    new_for.body = new_body
    return [new_for]


def _extend_loop_range(for_stmt: ast.For, extra: int):
    """Extend the loop range by 'extra' iterations for drain.

    Transforms pl.range(0, end) -> pl.range(0, end + extra)
    """
    if not isinstance(for_stmt.iter, ast.Call):
        return
    call = for_stmt.iter
    # Find the 'end' argument (second positional arg for pl.range(start, end, ...))
    if len(call.args) >= 2:
        end_arg = call.args[1]
    elif len(call.args) == 1:
        end_arg = call.args[0]
    else:
        return

    # Replace end with end + extra (inline the constant)
    new_end = ast.BinOp(
        left=end_arg,
        op=ast.Add(),
        right=ast.Constant(value=extra),
    )
    if len(call.args) >= 2:
        call.args[1] = new_end
    else:
        call.args[0] = new_end


def _build_ctx_fill(info: PipelineInfo, depth: int) -> list[ast.stmt]:
    """Build statements to fill the current ctx slot."""
    stmts = []

    _pl_ctx_0_assign = ast.Assign(
        targets=[ast.Name(id="_pl_ctx_0", ctx=ast.Store())],
        value=ast.Subscript(
            value=ast.Name(id="_pl_ctx_arr", ctx=ast.Load()),
            slice=ast.BinOp(
                left=ast.Name(id="_pl_task_id", ctx=ast.Load()),
                op=ast.Mod(),
                right=ast.Constant(value=depth),
            ),
            ctx=ast.Load(),
        ),
        lineno=0,
    )
    stmts.append(_pl_ctx_0_assign)

    loop_var = info.inner_loop_var
    # Use the original range end (before we extended it)
    range_end = (
        copy.deepcopy(info.inner_loop_range_end)
        if info.inner_loop_range_end
        else ast.Name(id="skv_tiles", ctx=ast.Load())
    )

    is_valid_if = ast.If(
        test=ast.Compare(
            left=ast.Name(id=loop_var, ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[range_end],
        ),
        body=[ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id="_pl_ctx_0", ctx=ast.Load()),
                attr="is_valid", ctx=ast.Store())],
            value=ast.Constant(value=1),
            lineno=0,
        )],
        orelse=[ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id="_pl_ctx_0", ctx=ast.Load()),
                attr="is_valid", ctx=ast.Store())],
            value=ast.Constant(value=0),
            lineno=0,
        )],
        lineno=0,
    )
    stmts.append(is_valid_if)

    # _pl_ctx_0.field = value for each ctx field (except is_valid).
    # Build field_name -> fill_expr mapping from stage_arg_mapping.
    fill_values: dict[str, ast.expr] = {}
    for arg_map in info.stage_arg_mapping:
        for item in arg_map:
            if item is not None:
                fname, fexpr = item
                if fname not in fill_values:
                    fill_values[fname] = fexpr

    for field_name in info.ctx_fields:
        if field_name == "is_valid":
            continue
        if field_name in fill_values:
            value_node = copy.deepcopy(fill_values[field_name])
        else:
            # Framework fields like _pl_task_id: fill with same-named var
            value_node = ast.Name(id=field_name, ctx=ast.Load())

        assign = ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id="_pl_ctx_0", ctx=ast.Load()),
                attr=field_name, ctx=ast.Store())],
            value=value_node,
            lineno=0,
        )
        stmts.append(assign)

    return stmts


def _build_stage_ctx_lookup(stage, depth: int) -> tuple[str, list[ast.stmt]]:
    """Return the ctx variable name and optional delayed ctx lookup statements."""
    delay = stage.delay
    if delay == 0:
        return "_pl_ctx_0", []

    ctx_var = f"_pl_ctx_neg{delay}"
    ctx_assign = ast.Assign(
        targets=[ast.Name(id=ctx_var, ctx=ast.Store())],
        value=ast.Subscript(
            value=ast.Name(id="_pl_ctx_arr", ctx=ast.Load()),
            slice=ast.BinOp(
                left=ast.BinOp(
                    left=ast.Name(id="_pl_task_id", ctx=ast.Load()),
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


def _build_stage_args(stage, arg_mapping: list, ctx_var: str) -> list[ast.expr]:
    """Build stage call args, replacing ctx-backed args with ctx.field references."""
    new_args = []
    for i, orig_arg in enumerate(stage.args):
        if i < len(arg_mapping) and arg_mapping[i] is not None:
            field_name = arg_mapping[i][0]  # (field_name, fill_expr) tuple
            new_args.append(ast.Attribute(
                value=ast.Name(id=ctx_var, ctx=ast.Load()),
                attr=field_name, ctx=ast.Load(),
            ))
        else:
            new_args.append(copy.deepcopy(orig_arg))
    return new_args


def _wrap_in_section(section_kind: str, body: list[ast.stmt]) -> ast.With:
    """Wrap statements in `with pl.section_<kind>():`."""
    section_call = ast.Call(
        func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()),
                           attr=f"section_{section_kind}", ctx=ast.Load()),
        args=[], keywords=[],
    )
    return ast.With(
        items=[ast.withitem(context_expr=section_call, optional_vars=None)],
        body=body, lineno=0,
    )


def _build_mix_outer_sync(stage, mix_body: list[ast.stmt], index_expr: ast.expr) -> list[ast.stmt]:
    """Wrap a mix stage's flattened body with its outer-buffer sync.

    A mix stage is flattened (inlined) into the loop — `mix_body` is its
    param-replaced body (with its own sections + inner sync inside). Each
    outer-buffer sync must be wrapped in the section of the sub-stage that touches
    the buffer, so set/wait fire on the correct core. pre (wait) goes in its own
    section block before the body; post (set) in its own block after. Returns the
    statement list [pre-sync sections..., mix body..., post-sync sections...].

    Shared by the full-pipeline path (under an is_valid guard) and sync_only.
    """
    body = []
    for acc in stage.cross_core:
        pre, _ = _build_cross_core_sync_one(acc, index_expr)
        if pre:
            body.append(_wrap_in_section(acc.section_kind or stage.section_kind, pre))
    body.extend(mix_body)
    for acc in stage.cross_core:
        _, post = _build_cross_core_sync_one(acc, index_expr)
        if post:
            body.append(_wrap_in_section(acc.section_kind or stage.section_kind, post))
    return body


def _build_guarded_stage_body(stage, call_expr: ast.Call, ctx_var: str, info: PipelineInfo) -> list[ast.stmt]:
    """Build sync + stage-call statements that run under the ctx.is_valid guard."""
    ctx_field_set = set(f for f in info.ctx_fields if f != "is_valid")
    index_expr = ast.Attribute(
        value=ast.Name(id=ctx_var, ctx=ast.Load()),
        attr="_pl_task_id", ctx=ast.Load(),
    )

    if stage.is_mix:
        # Flatten (inline) the synced mix body in place of a bare call, so the
        # inner sync survives codegen (the parser would otherwise inline the call
        # by re-reading the original, sync-free source). _flatten_mix_body builds
        # and caches the synced body on first use.
        mix_body = _flatten_mix_body(stage, call_expr, info)
        return _build_mix_outer_sync(stage, mix_body, index_expr)

    auto_pre, auto_post = _build_cross_core_sync(stage, index_expr)

    guarded_body = []
    guarded_body.extend(auto_pre)
    for s in stage.pre_stmts:
        guarded_body.append(_replace_ctx_fields(copy.deepcopy(s), ctx_var, ctx_field_set))
    guarded_body.append(ast.Expr(value=call_expr, lineno=0))
    for s in stage.post_stmts:
        guarded_body.append(_replace_ctx_fields(copy.deepcopy(s), ctx_var, ctx_field_set))
    guarded_body.extend(auto_post)
    return guarded_body


def _wrap_stage_section(stage, guarded_body: list[ast.stmt], ctx_var: str) -> ast.With:
    """Wrap a guarded stage body in its original pl.section_* context."""
    guarded_call = ast.If(
        test=ast.Attribute(
            value=ast.Name(id=ctx_var, ctx=ast.Load()),
            attr="is_valid", ctx=ast.Load(),
        ),
        body=guarded_body,
        orelse=[],
        lineno=0,
    )
    # Mix stage: no outer section wrapper (has its own sections inside)
    if stage.is_mix:
        return guarded_call
    section_attr = f"section_{stage.section_kind}"
    section_call = ast.Call(
        func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()),
                           attr=section_attr, ctx=ast.Load()),
        args=[], keywords=[],
    )
    return ast.With(
        items=[ast.withitem(context_expr=section_call, optional_vars=None)],
        body=[guarded_call],
        lineno=0,
    )


def _build_stage_call(stage, stage_idx: int, info: PipelineInfo,
                      depth: int) -> list[ast.stmt]:
    """Build a single stage call with delay, ctx lookup, and is_valid guard.

    Preserves pre_stmts (e.g. wait sync) and post_stmts (e.g. set sync)
    from the original section block around the stage call.
    """
    ctx_var, stmts = _build_stage_ctx_lookup(stage, depth)

    # Build the stage function call with args replaced
    arg_mapping = info.stage_arg_mapping[stage_idx]
    call_expr = ast.Call(
        func=ast.Name(id=stage.func_name, ctx=ast.Load()),
        args=_build_stage_args(stage, arg_mapping, ctx_var),
        keywords=[],
    )

    # Build the if-guarded body: pre_stmts (sync) + stage_call + post_stmts (sync)
    # all INSIDE the is_valid guard. With delay-adjusted index (ctx_var.tick),
    # sync must be guarded so warmup/drain iterations (is_valid=0) don't execute
    # wait/set and steal tokens belonging to later valid iterations.
    # Inside the sync, ctx-field references (e.g. `tick`) are replaced with
    # `ctx_var.field` so the event_id index uses the delayed task's value
    # (matching the data this stage actually processes).
    guarded_body = _build_guarded_stage_body(stage, call_expr, ctx_var, info)
    stmts.append(_wrap_stage_section(stage, guarded_body, ctx_var))

    return stmts
