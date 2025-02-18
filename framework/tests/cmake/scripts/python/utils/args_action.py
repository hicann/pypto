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
"""Args处理辅助.
"""
import argparse
import subprocess
from typing import Sequence, Optional, Any, List


class ArgsEnvDictAction(argparse.Action):
    """解析命令行参数传入的环境变量字段(env)
    """

    def __call__(self, parser, namespace, values, option_string=None):
        env_dict = getattr(namespace, self.dest, {}) or {}
        for item in values:
            k, v = item.split('=', 1)
            env_dict[k] = v
        setattr(namespace, self.dest, env_dict)


class ArgsGTestFilterListAction(argparse.Action):
    """解析命令行参数传入的 GTestFilter 字段
    """

    def __init__(self, option_strings: Sequence[str], dest: str, nargs: Optional[int] = None, **kwargs: Any) -> None:
        # 确保 nargs 至少为 1
        if nargs is None:
            nargs = '+'
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    @staticmethod
    def parse_all_cases(binary: str) -> List[str]:
        """获取gtest ut测试用例
        """
        result = subprocess.run([binary, '--gtest_list_tests'],
                            capture_output=True, text=True)
        cases = []
        current_suite = ""
        for line in result.stdout.split('\n'):
            line = line.rstrip()
            if not line or "GoogleTestVerification" in line:
                continue
            if line.endswith('.'):
                current_suite = line[:-1]
            elif line.startswith('  '):
                test_name = line.strip()
                cases.append(f"{current_suite}.{test_name}")
        return cases

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: List[str],
                 option_string: Optional[str] = None) -> None:
        # 解析每个字符串，按冒号分隔并展平
        case_list: List[str] = []

        target = getattr(namespace, 'target')
        if (len(values) == 1 and values[0] == "*"):
            case_list = self.parse_all_cases(target[0])
        else:
            for value in values:
                # 分割每个字符串，并过滤空字符串
                cases = [cs.strip() for cs in value.split(':') if cs.strip()]
                case_list.extend(cases)
        # 将结果设置到命名空间
        setattr(namespace, self.dest, case_list)
