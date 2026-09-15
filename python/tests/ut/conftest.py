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
"""UT 公共 conftest: gcov fork 子进程覆盖率数据 flush.

--forked 模式下 pytest-forked 通过 os._exit() 退出子进程, 跳过 atexit,
导致 gcov 的 __gcov_exit 不被调用, .gcda 不写入.
在 teardown 中手动调用 __gcov_dump() 将内存计数器写入 .gcda.
"""

import ctypes
import os


def duration_estimate(seconds: float):
    """Annotate a test case with an estimated duration (seconds)."""

    def decorator(func):
        func.duration_estimate = seconds
        return func

    return decorator


_gcov_dump = None
_gcov_resolved = False


def _resolve_gcov_dump():
    global _gcov_dump, _gcov_resolved
    if _gcov_resolved:
        return _gcov_dump
    _gcov_resolved = True
    try:
        lib = ctypes.CDLL(None)
        _gcov_dump = getattr(lib, "__gcov_dump", None)
    except Exception:
        pass
    return _gcov_dump


def pytest_runtest_teardown(item, nextitem):
    if not os.environ.get("PYPTO_GCOV_FLUSH"):
        return
    dump = _resolve_gcov_dump()
    if dump is not None:
        dump()
