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
"""工具基类定义.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any


class ToolsAbc(ABC):

    def __init__(self, args):
        # 路径管理
        self.source_root: Path = Path(__file__).parent.parent.parent.parent.parent.parent.parent.resolve()

        # 执行控制
        self.clean_flg: bool = args.tools_output_clean
        self.intercept: bool = args.intercept

    @property
    def brief(self) -> List[Any]:
        datas = [["SourceRoot", self.source_root],
                 ["CleanFlag", self.clean_flg],
                 ["Intercept", self.intercept]]
        return datas

    @abstractmethod
    def clean(self) -> bool:
        pass

    @abstractmethod
    def prepare(self) -> bool:
        pass

    @abstractmethod
    def process(self) -> bool:
        pass

    @abstractmethod
    def post(self) -> bool:
        pass
