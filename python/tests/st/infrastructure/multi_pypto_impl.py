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
"""验证看护多 pypto_impl 在线编译
"""
import sys
import logging
from datetime import datetime, timezone


def _import_pypto() -> int:
    ts = datetime.now(tz=timezone.utc)
    import pypto
    duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
    return duration


def test_multi_pypto_impl_import():
    verinfo = sys.version_info
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    # 1st
    duration = _import_pypto()
    logging.info("Python %s.%s 1st import, Duration %s secs.", verinfo.major, verinfo.minor, duration)
    # 2nd
    duration = _import_pypto()
    logging.info("Python %s.%s 1st import, Duration %s secs.", verinfo.major, verinfo.minor, duration)
