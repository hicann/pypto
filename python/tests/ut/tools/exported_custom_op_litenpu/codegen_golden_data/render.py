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
"""Rewrite every golden in this directory from the sibling case tables, and delete the unclaimed ones.

Each ``test_*_goldens.py`` beside this directory contributes a ``GOLDEN_CASES`` mapping of case name to
a zero-argument builder. A case name carries its own language extension (``add.cpp``), and the golden is
that name plus ``GOLDEN_SUFFIX``.
"""

import glob
import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_UT_DIR = os.path.dirname(_DATA_DIR)
GOLDEN_SUFFIX = ".golden"


def _load_suites():
    """Import every sibling golden suite by path and return them in a stable order."""
    suites = []
    for path in sorted(glob.glob(os.path.join(_UT_DIR, "test_*_goldens.py"))):
        name = "golden_render_" + os.path.basename(path)[:-3]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        suites.append(mod)
    return suites


def _reject_hook_hostile(case, text):
    """Refuse to write a golden pre-commit would rewrite, which would desync it from its emitter.

    trailing-whitespace carries no file filter, so it applies to this directory too, and the emitter is
    the only place that can fix a trailing space. The final newline is handled by the suite's
    ``golden_text``, since a fragment emitter legitimately has none.
    """
    for i, line in enumerate(text.split("\n"), 1):
        if line != line.rstrip():
            raise SystemExit(f"{case}: line {i} has trailing whitespace; fix the emitter, not the golden")
    if "\r" in text:
        raise SystemExit(f"{case}: carries CR; the emitter must write LF only")
    if not text.endswith("\n"):
        raise SystemExit(f"{case}: golden_text did not terminate the text; the suite's normalizer is wrong")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    claimed = set()
    for suite in _load_suites():
        for case, build in sorted(suite.GOLDEN_CASES.items()):
            text = suite.golden_text(build())
            _reject_hook_hostile(case, text)
            if case.endswith(".py"):
                compile(text, case, "exec")
            path = os.path.join(_DATA_DIR, case + GOLDEN_SUFFIX)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            claimed.add(os.path.basename(path))
            logger.info("wrote %s %d bytes", os.path.basename(path), len(text.encode("utf-8")))
    for stale in sorted(glob.glob(os.path.join(_DATA_DIR, "*" + GOLDEN_SUFFIX))):
        if os.path.basename(stale) not in claimed:
            os.remove(stale)
            logger.info("removed %s", os.path.basename(stale))


if __name__ == "__main__":
    main()
