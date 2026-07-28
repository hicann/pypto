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
import logging

import pypto_pro.language as pl


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets
    from pypto_pro.runtime.kernel import KernelDef

    kernel_def = kernel if isinstance(kernel, KernelDef) else kernel.to_kernel_def()
    cube, vector = _parse_and_codegen_targets(kernel_def, "a5", "")
    return _assemble_cv_source(cube, vector).content


@pl.jit()
def ssbuf_copy_kernel(x: pl.Tensor[[1], pl.DT_INT32]):
    message = pl.struct("Message", batch=0, block=0, offset=0)
    with pl.section_vector():
        message.batch = 8
        message.block = 1
        message.offset = 32768
        sub_id = pl.get_subblock_idx()
        if sub_id == 0:
            pl.ssbuf_store(message, 0)
            pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=15)

    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=15, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        pl.ssbuf_load(message, 0)
        pl.printf("Get ssbuf mssage: batch=%d, block=%d, offset=%d", message.batch, message.block, message.offset)


def test_ssbuf_store_and_load_codegen():
    cpp = _compile_to_cce(ssbuf_copy_kernel)
    logging.info("%s", cpp)
    assert "class Message" in cpp
    assert "reinterpret_cast<__ssbuf__ uint32_t*>((uint64_t)(0))" in cpp
    assert "reinterpret_cast<const uint32_t*>(&message_0)" in cpp
    assert "reinterpret_cast<uint32_t*>(&message_0)" in cpp
    assert "sizeof(message_0) / sizeof(uint32_t)" in cpp
    assert cpp.count("wait_intra_block(PIPE_S, 15);") == 1
    assert "wait_intra_block(PIPE_S, 31);" not in cpp
