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
"""Whole-text goldens for the generated C++, CMake and embedded-Python artifacts.

Each file is one emitter's complete output for one case, pinned byte-for-byte. The case tables live in
the sibling ``test_*_goldens.py`` modules, so a golden and its comparison share one definition; run
``render.py`` in this directory to rewrite them all after an intended emitter change.

The ``.golden`` suffix is load-bearing: pre-commit's clang-format hook selects on a C/C++ extension at
end of name, so a golden named ``x.cpp`` would be reformatted out from under its test.
"""
