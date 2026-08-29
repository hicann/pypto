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
"""STest Golden handler function registration management."""

import dataclasses
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union


@dataclasses.dataclass
class GoldenRegInfo:
    func: Optional[Callable]
    version: int = 0  # Golden implementation version
    timeout: Optional[int] = None  # Golden timeout duration


class GoldenRegister:
    # Global callback function registry
    _REG_MAP: Dict[str, GoldenRegInfo] = {}

    @classmethod
    def reg_golden_func(
        cls, case_names: Union[str, List[str]], version: int = 0, timeout: Optional[int] = None
    ) -> Callable:
        """
        Register callback function, supports two function prototypes:
            func(case_name: str, output: Path)
            func(case_name: str, output: Path, case_index: int)

        :param case_names: CaseName
        :param version: implementation version, controlled by Golden script. When the framework
            detects a version greater than the cached version, it triggers Golden regeneration
        :param timeout: timeout duration (in seconds), when the framework detects Golden files
            have exceeded the specified duration, it triggers Golden regeneration
        """

        def decorator(func: Callable) -> Callable:
            case_name_list = [case_names] if isinstance(case_names, str) else case_names
            for name in case_name_list:
                ori_func = cls._REG_MAP.get(name, None)
                if ori_func:
                    logging.debug("Case(%s) update func %s -> %s to %s", name, ori_func, func, hex(id(cls._REG_MAP)))
                else:
                    logging.debug("Case(%s) register func %s to %s", name, func, hex(id(cls._REG_MAP)))
                cls._REG_MAP[name] = GoldenRegInfo(func=func, version=version, timeout=timeout)
            return func

        return decorator

    @classmethod
    def get_golden_func(cls, case_name: str) -> Tuple[Optional[GoldenRegInfo], Optional[int]]:
        """Get callback function by name

        Supports the following case name formats:

        1. TEST/TEST_F scenario:
            TestSuiteName.TestCaseName
        2. TEST_P scenario:
            TestInstanceName/TestSuiteName.TestCaseName
            TestInstanceName/TestSuiteName.TestCaseName/
            TestInstanceName/TestSuiteName.TestCaseName/*
            TestInstanceName/TestSuiteName.TestCaseName*
            TestInstanceName/TestSuiteName.TestCaseName/numeric_index

        Corresponding registered case name supports the following scenarios:

        1. TEST/TEST_F scenario:
            TestSuiteName.TestCaseName
        2. TEST_P scenario:
            TestInstanceName/TestSuiteName.TestCaseName

        :param case_name: CaseName
        """
        # Normalize case name
        #   TestSuiteName.TestCaseName
        #   TestInstanceName/TestSuiteName.TestCaseName
        #   TestInstanceName/TestSuiteName.TestCaseName/{int}
        cs = case_name.replace("*", "")
        cs = cs[:-1] if cs.endswith("/") else cs

        # Extract case index (optional)
        cs_idx = None
        cs_split = cs.split("/")
        if cs_split[-1].isdigit():
            cs_idx = int(cs_split[-1])
            cs_split = cs_split[:-1]

        # Re-normalize case name
        #   TestSuiteName.TestCaseName
        #   TestInstanceName/TestSuiteName.TestCaseName
        cs = "/".join(cs_split)

        return cls._REG_MAP.get(cs, None), cs_idx

    @classmethod
    def get_golden_func_num(cls) -> int:
        return len(cls._REG_MAP)
