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

"""Frontend-test conftest with optional parallel pre-compilation.

Normally each frontend test compiles its ``@pl.jit`` kernel lazily on first launch, so the
whole suite pays for kernel compilation (codegen + bisheng ``.so`` build) serially, interleaved
with device launches. The compile step is host/CPU bound and independent of the runtime tensor
values, so it parallelizes well; the launch step must stay serial (one NPU).

Set ``PARALLEL_COMPILE=1`` to split the run into three phases:

  1. discovery — run every test with kernel launches stubbed out, recording each
     ``(kernel, tilingkey, datatype)`` combo;
  2. compile   — compile every recorded specialization in parallel across a thread pool;
     results land in each kernel's in-process cache;
  3. launch    — run the tests normally; every launch now hits the warm cache and only executes
     on device.

Anything the discovery pass misses (e.g. a kernel reached only after an earlier assertion, or a
non-``_TileJitKernel`` path) simply falls back to lazy compilation in phase 3, so behavior is
always correct — parallel compilation is a pure speed-up.

Tunables:
  PARALLEL_COMPILE       — ``1`` to enable the three-phase flow (default: off).
  PARALLEL_COMPILE_JOBS  — thread-pool size (default: min(#combos, os.cpu_count())).
"""

import logging
import os


def _parallel_compile_enabled() -> bool:
    return os.environ.get("PARALLEL_COMPILE") == "1"


def pytest_runtestloop(session):
    """Three-phase loop (discover → parallel compile → serial launch) when enabled.

    Returning ``None`` when disabled defers to pytest's default runtestloop. Any failure in the
    discover/compile phases is swallowed so the run degrades gracefully to lazy compilation.
    """
    if not _parallel_compile_enabled():
        return None  # default behavior

    if session.testsfailed and not session.config.option.continue_on_collection_errors:
        raise session.Interrupted(
            f"{session.testsfailed} error{'s' if session.testsfailed != 1 else ''} during collection"
        )
    if session.config.option.collectonly:
        return None  # let the default loop handle --collect-only

    from concurrent.futures import ThreadPoolExecutor
    import importlib

    from _pytest.runner import runtestprotocol

    # NB: ``pypto_pro.runtime`` re-exports the ``jit`` *function*, which shadows the submodule
    # under ``from pypto_pro.runtime import jit``; import the module object explicitly.
    jit = importlib.import_module("pypto_pro.runtime.jit")

    tr = session.config.pluginmanager.get_plugin("terminalreporter")

    def info(msg):
        if tr is not None:
            tr.write_line(f"[PARALLEL_COMPILE] {msg}")
        else:  # pragma: no cover - only when terminal reporter is absent
            logging.info("[PARALLEL_COMPILE] %s", msg)

    items = session.items

    # ---- Phase 1: discovery -- record every variant that misses the cache, launches stubbed.
    #
    # The seam is _compile_variant, the single funnel every cache miss goes through, rather
    # than _ensure_compiled. _compile_variant receives exactly what compiling needs and
    # nothing else -- no runtime argument tuple to replay, no keys to re-resolve -- and its
    # arguments are the cache key itself, so the dedup below is the real variant identity
    # instead of an approximation of it. It is still a private method, so the mismatch risk
    # has not vanished; what has changed is that phase 1 now reports when it goes wrong.
    records = []  # ordered, de-duplicated (kernel, concrete_key, spec, bound_sig, static_sig)
    seen = set()
    placeholder = jit.CompiledKernel(lib_path="", param_specs=[])
    tile_jit_kernel_cls = getattr(jit, "_TileJitKernel")
    orig_compile_variant = getattr(tile_jit_kernel_cls, "_compile_variant")
    orig_launch = getattr(jit, "_launch")

    def discover_compile_variant(self, concrete_key, spec, bound_signature, static_signature):
        # Record and hand back a placeholder without compiling and without touching the
        # cache, so the next launch of the same variant misses again and the dedup set --
        # not the cache -- decides what is new.
        tilingkey_packed, _, _, dtype_hash = spec
        dedup = (id(self), static_signature, dtype_hash, tilingkey_packed)
        if dedup not in seen:
            seen.add(dedup)
            records.append((self, concrete_key, spec, bound_signature, static_signature))
        return placeholder

    def noop_launch(*_args, **_kwargs):
        return None

    info(f"phase 1/3: discovering kernels across {len(items)} tests...")
    # Only the first failure is ever reported, so only the first is looked at, and even then
    # through reprcrash -- the one-line message pytest already built. Stringifying the whole
    # longrepr instead formats the entire traceback with source context, about 90 ms per test,
    # and phase 1 stubs the launch so every test that asserts on kernel output fails by
    # construction: doing it for all of them cost 19 s of the 25 s this phase took on a
    # 210-test directory. `any()` below short-circuits without touching the repr at all.
    first_failure = None

    def _crash_message(report) -> str:
        longrepr = getattr(report, "longrepr", None)
        crash = getattr(longrepr, "reprcrash", None)
        return getattr(crash, "message", None) or str(longrepr)[-200:]
    try:
        setattr(tile_jit_kernel_cls, "_compile_variant", discover_compile_variant)
        setattr(jit, "_launch", noop_launch)
        for i, item in enumerate(items):
            nextitem = items[i + 1] if i + 1 < len(items) else None
            try:
                # log=False: run setup/call/teardown for fixture/param handling without emitting
                # reports, so this phase does not affect the pass/fail tally.
                reports = runtestprotocol(item, nextitem=nextitem, log=False)
            except Exception as exc:  # noqa: BLE001 - discovery is best effort
                if first_failure is None:
                    first_failure = (item.nodeid, repr(exc))
                continue
            if first_failure is None and any(report.failed for report in reports):
                failed = next(report for report in reports if report.failed)
                first_failure = (item.nodeid, _crash_message(failed))
    finally:
        setattr(tile_jit_kernel_cls, "_compile_variant", orig_compile_variant)
        setattr(jit, "_launch", orig_launch)

    # Most tests "fail" in this phase by construction: the launch is stubbed, so anything
    # that asserts on kernel output sees an untouched tensor. That makes the per-test result
    # useless as a health signal. What is not normal is finishing discovery with nothing
    # recorded while tests were doing work -- that is precisely what a stub that no longer
    # matches the real method looks like (96b03fb84 added a keyword the stub did not accept;
    # every discovery call raised, every exception was swallowed as a test failure, and the
    # suite quietly fell back to serial lazy compilation for months). Report that case, and
    # carry a failure along so there is something to diagnose from.
    if not records and items:
        detail = f" First failure: {first_failure[0]} -> {first_failure[1]}" if first_failure else ""
        info(
            f"phase 1/3: WARNING -- discovered no kernels across {len(items)} test(s). "
            f"Parallel pre-compilation is doing nothing; check that the _compile_variant stub "
            f"still matches the real signature.{detail}"
        )

    # ---- Phase 2: compile every recorded kernel in parallel.
    if records:
        jobs = int(os.environ.get("PARALLEL_COMPILE_JOBS") or 0)
        if jobs <= 0:
            jobs = min(len(records), os.cpu_count() or 4)
        info(f"phase 2/3: compiling {len(records)} kernel(s) with {jobs} worker(s)...")

        def build(record):
            kernel, concrete_key, spec, bound_signature, static_signature = record
            try:
                # Codegen + compile once; _compile_variant installs the result in the
                # kernel's cache itself, which is what phase 3 then hits.
                orig_compile_variant(kernel, concrete_key, spec, bound_signature, static_signature)
            except Exception:  # noqa: BLE001 - phase 3 owns test error reporting
                # Pre-compilation is best-effort. The normal test phase retries the kernel
                # and either handles an expected error or reports an unexpected one.
                pass

        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            list(pool.map(build, records))
    else:
        info("phase 2/3: no JIT kernels discovered; nothing to pre-compile.")

    # ---- Phase 3: launch tests one by one (default protocol, warm compile cache).
    info(f"phase 3/3: launching {len(items)} tests...")
    for i, item in enumerate(items):
        nextitem = items[i + 1] if i + 1 < len(items) else None
        item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
        if session.shouldfail:
            raise session.Failed(session.shouldfail)
        if session.shouldstop:
            raise session.Interrupted(session.shouldstop)
    return True
