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
"""Golden-text helpers shared by the sibling ``test_*_goldens.py`` suites, so neither suite has to load
the other by filename to reach them.
"""

from __future__ import annotations

import importlib.util
import pathlib as _pl

_UT_DIR = _pl.Path(__file__).resolve().parents[1]
GOLDEN_SUFFIX = ".golden"


def golden_text(raw: str) -> str:
    """The on-disk form of an emitter's output: terminated with exactly one newline if it has none.

    Goldens must be newline-terminated or pre-commit's end-of-file-fixer rewrites them and the test
    desyncs from the emitter. A fragment emitter (a block spliced into a larger TU) legitimately ends
    without one, so this adds it and the comparison applies the same function to the fresh output.
    The one property it therefore cannot pin is whether a fragment ends in zero or one newline, which
    no file in this repo could pin anyway.
    """
    return raw if raw.endswith("\n") else raw + "\n"


def _load(unique_name: str, filename: str):
    """Load a sibling ``*.py`` sample module by path (works without a parent package)."""
    spec = importlib.util.spec_from_file_location(unique_name, _UT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def claimed_cases():
    """Every sibling golden suite's case names, unioned, and how many suites contributed them.

    The goldens of all suites share one directory, so a per-suite count of it means nothing; the
    sibling suites are the only definition of what belongs there.
    """
    suites = sorted(_UT_DIR.glob("test_*_goldens.py"))
    claimed = set()
    for suite in suites:
        mod = _load("golden_claim_" + suite.stem, suite.name)
        claimed |= set(mod.GOLDEN_CASES)
    return claimed, len(suites)
