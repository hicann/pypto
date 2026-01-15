#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Args处理辅助(适配自定义参数--gtest_list_tests_with_meta)
"""
import argparse
import subprocess
import logging
from typing import Sequence, Optional, Any, List, Dict
import re


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
    """解析命令行参数传入的 GTestFilter 字段(适配自定义元信息参数)
    """

    def __init__(self, option_strings: Sequence[str], dest: str, nargs: Optional[int] = None, **kwargs: Any) -> None:
        # 确保 nargs 至少为 1
        if nargs is None:
            nargs = '+'
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: List[str],
                 option_string: Optional[str] = None) -> None:
        # 解析每个字符串，按冒号分隔并展平
        case_list = []

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

    @staticmethod
    def get_test_costs(binary: str) -> Dict[str, float]:
        """
        获取所有带耗时信息的测试用例(通过自定义参数--gtest_list_tests_with_meta)
        返回格式: { "TestCaseName.TestName": cost_seconds, ... }
        """
        cost_map = {}

        result = subprocess.run(
            [binary, '--gtest_list_tests_with_meta'],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        # 仅解析stdout(格式:TestCaseName.TestName|cost_seconds)
        pattern = re.compile(r'^([\w\.]+)\|(\d+\.?\d*)$', re.MULTILINE)
        matches = pattern.findall(result.stdout)
        for test_name, cost_str in matches:
            cost_map[test_name.strip()] = float(cost_str.strip())

        return cost_map

    @staticmethod
    def parse_all_cases(binary: str) -> List[str]:
        """
        获取 gtest ut 测试用例列表，并重排序：
          - 有耗时信息的排在前面
          - 无耗时信息的排在后面
        """
        # 1. 通过自定义参数获取带耗时的测试列表
        cost_map = ArgsGTestFilterListAction.get_test_costs(binary)

        # 2. 获取原生测试列表(无耗时的用例)
        cases = []
        result = subprocess.run([binary, '--gtest_list_tests'], capture_output=True, text=True)
        current_suite = ""

        for line in result.stdout.split('\n'):
            line = line.rstrip()
            if not line or line.startswith('#') or "GoogleTestVerification" in line:
                continue
            if line.endswith('.'):
                current_suite = line[:-1]
            elif line.startswith('  '):
                test_name = line.strip()
                full_name = f"{current_suite}.{test_name}"
                cases.append(full_name)


        # 3. 分类重排序
        cost_tests = []
        no_cost_tests = []
        for test in cases:
            if test in cost_map:
                cost_tests.append(test)
            else:
                no_cost_tests.append(test)

        cost_tests_sorted = sorted(cost_tests, key=lambda x: cost_map[x], reverse=True)

        # 4. 日志输出
        logging.info("Found %d tests with cost info, %d tests without.",
                    len(cost_tests_sorted), len(no_cost_tests))
        if cost_tests_sorted:
            logging.info("First few cost-aware tests (desc order): %s", cost_tests_sorted)

        return cost_tests_sorted + no_cost_tests
