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
"""Pytest 配置控制
"""
import os
from typing import List, Optional

import pytest


def _set_process_desc(desc: str):
    try:
        import setproctitle
        setproctitle.setproctitle(desc)
    except ModuleNotFoundError:
        pass


def pytest_addoption(parser: pytest.Parser):
    """向 pytest 注册自定义参数

    :param parser: pytest.Parser 类型
    """
    parser.addoption("--device", nargs="+", type=int,
                     help="Device ID, default 0")


def pytest_configure_node(node):
    """pytest-xdist 回调函数, 在 pytest 主进程 fork 出 worker 进程之前被调用.

    :param node: worker 节点
    """
    # 获取 DeviceId 列表, 当外部传入 --device 时, 是 STest 场景, 否则是 UTest 场景
    device_id_lst: Optional[List[int]] = node.config.getoption("--device")
    if device_id_lst:
        # 获取 WorkerIdx, 并获取 DeviceId
        worker_idx = int(str(node.gateway.id).lstrip("gw"))
        if worker_idx >= len(device_id_lst):
            raise ValueError(f"WorkerIdx[{worker_idx}] out of DeviceIdLst{device_id_lst} range.")
        device_id: int = device_id_lst[worker_idx]

        # 修改 worker 名称, 设置 worker 中的 DeviceId
        node.gateway.id = f"Device[{device_id}]"  # 体现在回显中
        node.gateway.remote_exec(f'import os; os.environ["TILE_FWK_DEVICE_ID"] = "{device_id}"')
    else:
        node.gateway.remote_exec(f'import os; os.environ.pop("TILE_FWK_DEVICE_ID", None)')


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    device_id: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID", None)
    if device_id is not None:
        _set_process_desc(f"Device[{device_id}]")
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """case 进程启动后被调用"""
    device_id: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID", None)
    if device_id is not None:
        case_name: str = str(item.name)
        _set_process_desc(f"Case(Device[{device_id}]::{case_name})")
    return None  # 继续执行默认的测试流程
