#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import os
import pytest


def pytest_configure_node(node):
    """每个worker进程启动时配置环境变量"""
    worker_id = node.workerinput["workerid"]  # gw0, gw1...
    worker_num = int(worker_id.lstrip("gw"))
    os.environ.pop("TILE_FWK_DEVICE_ID", None)
    os.environ["TILE_FWK_DEVICE_ID"] = str(worker_num)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """fork子进程前强制刷新环境变量"""
    worker_id = item.config.workerinput.get("workerid", "master") if hasattr(item.config, "workerinput") else "master"
    if worker_id.startswith("gw"):
        worker_num = int(worker_id.lstrip("gw")) if worker_id.lstrip("gw").isdigit() else 0
        os.environ["TILE_FWK_DEVICE_ID"] = str(worker_num)