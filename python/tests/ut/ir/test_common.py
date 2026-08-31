# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Common helpers shared across ir.Pass tests (merge_stmts_into_if and friends)."""

import difflib
import logging
import os
from pathlib import Path
from typing import Any

from pypto import ir, pil


def check_snapshot(func: Any, golden_path: Path) -> None:
    if os.environ.get("PYPTO_RENDER_IR"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(str(func).strip() + "\n")
        print(f"Updated: {golden_path}")
        return

    actual = str(func)
    golden = golden_path.read_text()
    if golden.strip() != actual.strip():
        diff = "".join(
            difflib.unified_diff(
                golden.splitlines(keepends=True),
                actual.splitlines(keepends=True),
            )
        )
        raise AssertionError("IR snapshot mismatch in %s:\n%s" % (func.name, diff))


def _ssa_verify(verifier, prog, name):
    diagnostic = verifier.verify(prog)
    if diagnostic:
        print(f"{prog}\n")
        print(ir.IRVerifier.generate_report(diagnostic))
        raise SyntaxError(f"IR verification failed after {name}")

def ssa_verify(func, desc: str = ""):
    verifier = ir.IRVerifier.create_default()
    b = ir.IRBuilder()
    prog = b.create_program([func], "main", ir.Span.unknown())
    _ssa_verify(verifier, prog, desc)


def run_merge_pass(func, *args, create_new_logical_tensor=True):
    """Compile a kernel and run canonicalize + dce + merge_stmts_into_if, stopping before lowering
    so the resulting if-tree (func.body) is inspectable. Mirrors compile_new_ir's first half.
    """
    b = ir.IRBuilder()
    func = pil.compile(func, *args, create_new_logical_tensor=create_new_logical_tensor)
    prog = b.create_program([func], "main", ir.Span.unknown())
    logging.info("\ninitial:\n%s" % func)
    dce = ir.Pass.aggressive_dce()
    canonical = ir.Pass.canonicalize()
    merge = ir.Pass.merge_stmts_into_if()
    verifier = ir.IRVerifier.create_default()
    _ssa_verify(verifier, prog, "original")
    prog = canonical(prog)
    _ssa_verify(verifier, prog, "canonical")
    prog = dce(prog)
    _ssa_verify(verifier, prog, "dce")
    prog = canonical(merge(prog))
    _ssa_verify(verifier, prog, "merged")
    func = prog.functions[func.name]
    logging.info("\nmerged:\n%s" % func.body)
    return func
