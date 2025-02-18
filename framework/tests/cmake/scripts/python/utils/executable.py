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
"""可执行文件执行辅助.
"""
import os
import shlex
import subprocess
import sys
from datetime import timedelta, datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Tuple


class Executable:
    """可执行文件
    """

    def __init__(self, file: Path, envs: Optional[Dict[str, str]] = None, timeout: Optional[int] = None):
        """
        :param file: 可执行文件路径
        :param envs: 可执行文件执行时额外指定环境变量
        """
        self.file: Path = Path(file).resolve()
        self.envs: Dict[str, str] = envs if envs is not None else {}
        self.timeout: Optional[int] = timeout

    @property
    def brief(self) -> str:
        asan: str = "ON" if "ASAN_OPTIONS" in self.envs.keys() else "OFF"
        ubsan: str = "ON" if "UBSAN_OPTIONS" in self.envs.keys() else "OFF"
        return f"({self.file.name}) XSAN(ASAN:{asan} UBSAN:{ubsan})"

    def run(self, gtest_filter: str,
            envs: Optional[Dict[str, str]] = None) -> Tuple[subprocess.CompletedProcess, str, timedelta]:
        """执行可执行文件

        :param gtest_filter: GTestFilter
        :param envs: 运行时额外需配置的环境变量
        :return: subprocess.CompletedProcess 执行结果
        :raise subprocess.TimeoutExpired: timeout 指定且执行超时
        """
        cmd: str = (f"{sys.executable} " if self.file.name.endswith(".py") else "./") + f"{self.file.name} "
        cmd += f"--gtest_filter={gtest_filter}"
        # 环境变量优先级: 函数参数指定 > 类内环境变量(命令行参数指定) > 系统内已有的
        envs = envs if envs is not None else {}
        act_env = os.environ.copy()  # 系统环境变量
        act_env.update(self.envs)  # 额外指定环境变量
        act_env.update(envs)  # 函数调用时指定的环境变量
        cwd: str = str(self.file.parent)
        ts = datetime.now(tz=timezone.utc)
        ret = subprocess.run(shlex.split(cmd), env=act_env, cwd=cwd, timeout=self.timeout,
                             capture_output=True, check=False, text=True, encoding='utf-8')
        return ret, cmd, datetime.now(tz=timezone.utc) - ts
