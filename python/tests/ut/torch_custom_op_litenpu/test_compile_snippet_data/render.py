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
"""Rewrite every golden in this directory from ``GOLDEN_CASES`` and delete the ones no case claims.

The case table lives in the sibling ``test_compile_snippet.py`` as ``GOLDEN_CASES``, so the goldens and
the comparison share one definition of each case. Run this from an environment where ``import pypto``
succeeds: the tracer classifies ``pypto`` as a pre-imported module only when it can import it, so a
snippet rendered without pypto available packs references that belong in the header.
"""

import glob
import importlib.util
import os
import sys

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_SUITE_PATH = os.path.join(os.path.dirname(_DATA_DIR), "test_compile_snippet.py")


def _load_suite():
    """Import the sibling test module by path (it needs no parent package)."""
    spec = importlib.util.spec_from_file_location("test_compile_snippet_render", _SUITE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    import pypto  # noqa: F401 - fail loudly here rather than render a mis-classified snippet

    suite = _load_suite()
    for case, build in sorted(suite.GOLDEN_CASES.items()):
        text = build()
        compile(text, case, "exec")
        path = os.path.join(_DATA_DIR, case + suite.GOLDEN_SUFFIX)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote", os.path.basename(path), len(text.encode("utf-8")), "bytes")
    for stale in sorted(glob.glob(os.path.join(_DATA_DIR, "*" + suite.GOLDEN_SUFFIX))):
        if os.path.basename(stale)[:-len(suite.GOLDEN_SUFFIX)] not in suite.GOLDEN_CASES:
            os.remove(stale)
            print("removed", os.path.basename(stale))


if __name__ == "__main__":
    main()
