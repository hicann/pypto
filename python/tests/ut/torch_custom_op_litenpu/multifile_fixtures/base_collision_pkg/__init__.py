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
"""Package for the canonical base-collision test: two DIFFERENT helpers whose sanitized
``<module>__<qualname>`` bases collide because of a ``__`` inside a module segment.

* ``base_collision_pkg.a__b.f``  -> base ``base_collision_pkg__a__b__f``
* ``base_collision_pkg.a.b__f``  -> base ``base_collision_pkg__a__b__f``
  (same base -> the ``__2`` counter disambiguates)
"""
