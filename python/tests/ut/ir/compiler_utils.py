# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pypto import ir, pil


def compile_new_ir(pyfunc, *args, **kwargs):
    """Compile a Python function and run the complete new-IR lowering pipeline."""
    builder = ir.IRBuilder()
    func = pil.compile(pyfunc, *args, **kwargs)
    program = builder.create_program([func], "main", ir.Span.unknown())

    dce = ir.Pass.aggressive_dce()
    canonicalize = ir.Pass.canonicalize()
    merge_stmts = ir.Pass.merge_stmts_into_if()
    create_root_functions = ir.Pass.create_root_functions()
    token_pass = ir.Pass.token_pass()

    program = token_pass(program)
    program = canonicalize(program)
    program = dce(program)
    program = dce(canonicalize(program))
    program = canonicalize(merge_stmts(program))
    program = create_root_functions(program)
    return program.functions[func.name]
