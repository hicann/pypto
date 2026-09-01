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
""" """

import itertools

from .error import FeError


class CompileState:
    _loop_idx_generator = itertools.count(0)
    in_function = False
    _atomic_scope_iter_id = 0

    @classmethod
    def init_atomic_scope_iter(cls):
        cls._atomic_scope_iter_id = 0

    @classmethod
    def next_loop_idx(cls) -> int:
        return next(cls._loop_idx_generator)

    @classmethod
    def bump_atomic_scope_iter(cls):
        cls._atomic_scope_iter_id += 1

    @classmethod
    def get_atomic_scope_iter(cls) -> int:
        return cls._atomic_scope_iter_id

    @classmethod
    def begin_function(cls):
        if cls.in_function:
            raise FeError(RuntimeError("function nested is not allowed"))
        cls.in_function = True

    @classmethod
    def end_function(cls):
        cls.in_function = False
        cls._loop_idx_generator = itertools.count(0)
