#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Helpers for lowering PIL-compiled functions through the new IR pipeline."""

from .. import ir, pil
from ..logging import log_debug


def _dump_program(name, program):
    log_debug(f"=========================={name}=========================\n{program}")


def _build_default_pipeline():
    dce = ir.Pass.aggressive_dce()
    canonicalize = ir.Pass.canonicalize()
    merge_stmts = ir.Pass.merge_stmts_into_if()
    create_root_functions = ir.Pass.create_root_functions()
    finalize = ir.Pass.finalize_dynamic_function()

    return [
        ("first_canonicalize_dce", lambda p: dce(canonicalize(p))),
        ("second_canonicalize_dce", lambda p: dce(canonicalize(p))),
        ("canonicalize(merge_stmts)", lambda p: canonicalize(merge_stmts(p))),
        ("create_root_functions", create_root_functions),
        ("finalize", finalize),
    ]


def _run_pipeline(program, pipeline):
    _dump_program("initial", program)
    for name, transform in pipeline:
        program = transform(program)
        _dump_program(f"after {name}", program)
    return program


def compile_new_ir(pyfunc, *args, pipeline=None, **kwargs):
    """Compile a Python function and run the complete new-IR lowering pipeline."""
    builder = ir.IRBuilder()
    func = pil.compile(pyfunc, *args, **kwargs)
    program = builder.create_program([func], "main", ir.Span.unknown())

    if pipeline is None:
        pipeline = _build_default_pipeline()

    program = _run_pipeline(program, pipeline)
    return program.functions[func.name]
